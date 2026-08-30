#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动对目录内文件用 AI（DeepSeek）分类，缺目录自动创建，分类目录最多两级。

用法：
    python3 classify_files.py [目录] [--dry-run]    # 分类
    python3 classify_files.py [目录] --rollback     # 回滚：文件移回顶层、删除子目录
    目录默认 ~/Downloads，只处理该目录的顶层文件（不递归）。

流程：
    1. 扫描目录顶层文件，跳过目录、隐藏文件、下载中的临时文件（.crdownload 等）。
    2. 读出「已有目录」，连同文件名清单批量发给 DeepSeek，
       返回「文件名 -> 目录名」映射（一级，或 一级/二级）。把已有目录写进提示词，
       是为了让 AI 优先复用现有目录，避免目录无限增长。
    3. 按映射 mkdir -p 建缺失目录并移动文件；识别不出或失败的兜底移入 other/。
       文件名有固定前缀、属于同一专辑/艺术家/剧集时，归入 一级/二级 子目录。
    4. 同名冲突自动重命名（file (1).ext）。
    5. --rollback：把子目录里的文件全部移回顶层，并删除子目录，用于重新分类。

配置项（文件顶部或环境变量）：
    DEEPSEEK_API_KEY       必填，DeepSeek API Key
    DEEPSEEK_BASE_URL      Anthropic 兼容端点，默认 https://api.deepseek.com/anthropic
    DEEPSEEK_MODEL         默认 deepseek-v4-flash
    DEEPSEEK_MAX_TOKENS    单次响应最大 token，默认 8192
    DEEPSEEK_BATCH_SIZE    每批发给 AI 的文件数，默认 300
    DEEPSEEK_TREE_LIMIT    提示词里目录树最大行数，默认 200
    OTHER_DIR              兜底目录名，默认 other
"""

import argparse
import json
import os
import re
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request

try:
    import certifi
except ImportError:
    certifi = None

# python.org 版 Python 自带 cert.pem 缺失，urllib 会报 CERTIFICATE_VERIFY_FAILED；
# 用 certifi 的 CA bundle 建 SSL 上下文来规避。
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where()) if certifi else ssl.create_default_context()

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic").rstrip("/")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
MAX_TOKENS = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "8192"))
BATCH_SIZE = int(os.environ.get("DEEPSEEK_BATCH_SIZE", "300"))
TREE_LIMIT = int(os.environ.get("DEEPSEEK_TREE_LIMIT", "200"))
OTHER_DIR = os.environ.get("OTHER_DIR", "other")

# 下载中的临时文件后缀，跳过不处理
IN_PROGRESS_SUFFIXES = (".crdownload", ".part", ".download", ".tmp", ".partial")

SYSTEM_PROMPT = (
    "你是一个文件整理助手。根据文件名和后缀，把每个文件归入一个目录。"
    "第一级是分类目录，例如 音乐、影视、文档、图片 等。"
    "当同一批里有多个文件名共享明显前缀、明显属于同一个专辑 / 艺术家 / 剧集时，"
    "把它们归入一个二级子目录，目录名取该专辑 / 艺术家 / 剧集名，"
    "返回形式为「一级/二级」，例如 音乐/十二月的肖邦、影视/琅琊榜。"
    "孤立文件（无法与其他文件成组）只归入一级目录，"
    "禁止为单个文件单独建目录（无论一级还是二级）。"
    "路径最多两级，禁止出现第三级。"
    "当文件名能明确识别出内容主题时，用主题名作为一级目录，例如 文学、小说、技术文档、"
    "技术数据、美食、哲学、足控、游戏、旅行 等；"
    "其中 小说 指虚构故事，文学 指散文/诗歌等非小说文学，技术文档 指手册/说明/规范，"
    "技术数据 指数据表/数据集/日志。"
    "否则按文件类型归类，例如 音频、视频、文档、图片、字体、压缩包。"
    "同类文件归同一个目录。"
    "优先复用「现有目录」里已有的目录；确属新类别时才新建一个目录。"
    "无法判断的，统一归入 other。"
    "只输出一个 JSON 对象，键为完整文件名，值为目录（一级，或 一级/二级），"
    "不要输出任何解释或多余文字。"
)


def list_files(source_dir):
    """返回顶层文件的文件名列表（跳过目录、隐藏文件、下载中文件）。"""
    names = []
    for name in os.listdir(source_dir):
        full = os.path.join(source_dir, name)
        if not os.path.isfile(full):
            continue
        if name.startswith("."):
            continue
        if name.lower().endswith(IN_PROGRESS_SUFFIXES):
            continue
        names.append(name)
    return sorted(names)


def list_directory_tree(source_dir):
    """列出已有目录（两级：一级分类，及一级下的二级子目录），用于喂给 AI 复用。"""
    entries = []
    for d in sorted(os.listdir(source_dir)):
        full = os.path.join(source_dir, d)
        if not os.path.isdir(full) or d.startswith("."):
            continue
        entries.append(d)
        for sub in sorted(os.listdir(full)):
            subfull = os.path.join(full, sub)
            if os.path.isdir(subfull) and not sub.startswith("."):
                entries.append(f"{d}/{sub}")
    if len(entries) > TREE_LIMIT:
        entries = entries[:TREE_LIMIT] + [f"...（其余 {len(entries) - TREE_LIMIT} 个目录省略）"]
    return entries


def call_deepseek(user_prompt):
    """调用 DeepSeek 的 Anthropic 兼容端点，返回模型输出文本。"""
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=180, context=SSL_CONTEXT) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    # Anthropic 返回 content 为块列表，取其中的 text 块（跳过 thinking 块）
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return ""


def _try_strict_json(text):
    """按标准 JSON 解析，仅当结果是 dict 时返回；失败或非 dict 返回 None。"""
    for candidate in (text, text.replace("“", '"').replace("”", '"')):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _find_value_after(text, needle):
    """在 text 中定位 needle（文件名），返回其后「: 值」里的目录值；找不到返回 None。"""
    idx = text.find(needle)
    if idx == -1:
        return None
    colon = text.find(":", idx + len(needle))
    if colon == -1:
        return None
    p = colon + 1
    while p < len(text) and text[p] in " \t\r\n":
        p += 1
    if p < len(text) and text[p] in ('"', "'"):
        quote = text[p]
        end = text.find(quote, p + 1)
        if end == -1:
            return None  # 值被截断，无闭合引号，丢弃
        return text[p + 1:end].strip()
    # 无引号值：读到逗号 / 换行 / 右花括号为止
    end = p
    while end < len(text) and text[end] not in ",\n\r}":
        end += 1
    return text[p:end].strip()


def _filename_variants(fname):
    """生成文件名常见的转义变体，用于在模型输出里兜底匹配（模型可能做了转义）。"""
    variants = []
    if '"' in fname:
        variants.append(fname.replace('"', '\\"'))
    if '\\' in fname:
        variants.append(fname.replace('\\', '\\\\'))
    return [v for v in variants if v != fname]


def _extract_by_anchor(texts, filenames):
    """回退解析：按已知文件名在原文里逐条定位并读取目录值。

    不依赖整段 JSON 合法，因此能容忍文件名里未转义的引号/反斜杠、
    单引号分隔、尾逗号/缺逗号、输出被截断等异常；模型漏掉的条目直接不收录。
    """
    mapping = {}
    for fname in filenames:
        value = None
        for text in texts:
            value = _find_value_after(text, fname)
            if value is None:
                for variant in _filename_variants(fname):
                    value = _find_value_after(text, variant)
                    if value is not None:
                        break
            if value is not None:
                break
        if value:
            mapping[fname] = value
    return mapping


def parse_json_response(text, filenames=None):
    """从模型输出里提取「文件名 -> 目录」映射，容错处理常见模型异常。

    有已知文件名时优先按锚点逐条提取（键一定与真实文件名一致，避免反斜杠
    等被 JSON 转义改坏），再用标准 JSON 解析结果补齐漏项；没有文件名清单时
    退回纯 JSON 解析。filenames 为本次请求已知的文件名清单。
    """
    raw = text.strip()
    # 去除可能的 markdown 代码块包裹
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    # 若周边有多余文字，截取第一个 { 到最后一个 }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]

    # 归一化全角冒号/逗号（不影响文件名里的全角引号），修复扩展名跑到引号外的情况
    normalized = raw.replace("：", ":").replace("，", ",")
    normalized = re.sub(
        r'("[^"]*")\.([A-Za-z0-9]{1,8})(\s*:)',
        lambda m: m.group(1)[:-1] + "." + m.group(2) + '"' + m.group(3),
        normalized,
    )

    parsed = _try_strict_json(normalized)

    if filenames:
        mapping = _extract_by_anchor([normalized, raw], filenames)
        # 用严格解析结果补齐锚点没定位到的文件（例如模型改写了文件名的情况）
        if parsed is not None:
            for fname in filenames:
                if fname not in mapping:
                    v = resolve_category(parsed, fname)
                    if v:
                        mapping[fname] = v
        if mapping:
            return mapping

    if parsed is not None:
        return parsed

    raise json.JSONDecodeError("无法从模型输出解析出有效映射", text, 0)


def resolve_category(mapping, fname):
    """从 AI 返回的映射里查文件名对应的目录，容忍键与文件名不完全一致。"""
    c = mapping.get(fname)
    if c:
        return c
    # 去掉文件名里常见的全角引号后比较，再比较去掉扩展名的主体，最后用包含关系兜底
    def norm(s):
        return re.sub(r"[“”‘’]", "", s)
    nf = norm(fname)
    for k, v in mapping.items():
        if not isinstance(k, str):
            continue
        nk = norm(k)
        if nk == nf:
            return v
        if os.path.splitext(nk)[0] == os.path.splitext(nf)[0]:
            return v
        if nk in nf or nf in nk:
            return v
    return ""


def sanitize_rel_path(rel):
    """清洗 AI 返回的路径并固定为最多两级目录；非法时返回空字符串。"""
    if not isinstance(rel, str):
        return ""
    rel = rel.replace("\\", "/").strip().strip("/")
    # 逐段切分，去掉空段，最多保留前两段
    parts = [p.strip() for p in rel.split("/") if p.strip()][:2]
    clean = []
    for p in parts:
        if p in (".", ".."):
            return ""  # 拒绝 "."、".."，防止逃逸出目标目录
        clean.append(p)
    if not clean:
        return ""
    return "/".join(clean)


def unique_dest(dest_dir, filename):
    """在 dest_dir 下为 filename 生成不冲突的完整路径（同名自动加序号）。"""
    base, ext = os.path.splitext(filename)
    candidate = filename
    i = 1
    while os.path.exists(os.path.join(dest_dir, candidate)):
        candidate = f"{base} ({i}){ext}"
        i += 1
    return os.path.join(dest_dir, candidate)


def rollback(source_dir, dry_run, assume_yes):
    """回滚：把子目录里的文件全部移回顶层，并删除子目录。"""
    subdirs = sorted(
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d)) and not d.startswith(".")
    )
    if not subdirs:
        print("没有需要回滚的子目录。")
        return

    # 先递归收集所有子目录里的普通文件（跳过隐藏文件），再统一移动
    files = []
    for sub in subdirs:
        for root, _dirs, names in os.walk(os.path.join(source_dir, sub)):
            for name in names:
                if not name.startswith("."):
                    files.append(os.path.join(root, name))

    print(f"回滚：将把 {len(files)} 个文件移回顶层，并删除 {len(subdirs)} 个子目录。")

    if not dry_run and not assume_yes:
        try:
            answer = input("确认回滚？此操作不可逆 [y/N]: ").strip().lower()
        except EOFError:
            answer = "n"
        if answer not in ("y", "yes"):
            print("已取消。")
            return

    for full in files:
        dest = unique_dest(source_dir, os.path.basename(full))
        if dry_run:
            print(f"[dry-run] {os.path.relpath(full, source_dir)} -> {os.path.basename(dest)}")
        else:
            shutil.move(full, dest)

    for sub in subdirs:
        if dry_run:
            print(f"[dry-run] 删除目录 {sub}/")
        else:
            shutil.rmtree(os.path.join(source_dir, sub))

    verb = "将" if dry_run else "已"
    print(f"\n{verb}移动 {len(files)} 个文件、{verb}删除 {len(subdirs)} 个子目录。")


def main():
    parser = argparse.ArgumentParser(description="用 AI 自动分类目录内文件（DeepSeek）")
    parser.add_argument(
        "path", nargs="?", default=os.path.expanduser("~/Downloads"),
        help="待分类目录，默认 ~/Downloads",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印将要执行的操作，不真正移动文件",
    )
    parser.add_argument(
        "--rollback", action="store_true",
        help="回滚：把子目录里的文件移回顶层，并删除子目录",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="回滚时跳过确认提示",
    )
    args = parser.parse_args()

    source_dir = os.path.abspath(os.path.expanduser(args.path))
    if not os.path.isdir(source_dir):
        print(f"错误：目录不存在: {source_dir}", file=sys.stderr)
        sys.exit(1)

    if args.rollback:
        rollback(source_dir, args.dry_run, args.yes)
        return

    if not API_KEY:
        print("错误：未设置 DEEPSEEK_API_KEY 环境变量。", file=sys.stderr)
        print('例如：export DEEPSEEK_API_KEY="sk-..."', file=sys.stderr)
        sys.exit(1)

    files = list_files(source_dir)
    if not files:
        print("没有需要分类的文件。")
        return

    tree = list_directory_tree(source_dir)
    tree_text = "\n".join(f"- {t}" for t in tree) if tree else "(空)"

    print(f"待分类目录: {source_dir}")
    print(f"待分类文件: {len(files)} 个；已有目录: {len(tree)} 个")
    if args.dry_run:
        print("（dry-run 模式，只打印不移动）\n")

    moved = 0
    to_other = 0
    dest_dirs = set()
    unmoved = []

    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i + BATCH_SIZE]
        user_prompt = (
            f"现有目录（优先复用，勿随意新建）：\n{tree_text}\n\n"
            f"待分类文件（共 {len(batch)} 个）：\n" + "\n".join(batch) + "\n\n"
            "请返回 JSON 对象，把每个文件映射到目录（一级，或 一级/二级）。"
        )

        mapping = None
        last_err = None
        last_raw = None
        for attempt in range(1, 4):
            try:
                content = call_deepseek(user_prompt)
                last_raw = content
                mapping = parse_json_response(content, batch)
                break
            except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as e:
                last_err = e
                if attempt < 3:
                    time.sleep(2 * attempt)
        if mapping is None:
            print(
                f"AI 调用失败（已重试 3 次），本批 {len(batch)} 个文件保持不动：{last_err}",
                file=sys.stderr,
            )
            if isinstance(last_err, json.JSONDecodeError) and last_raw:
                print(f"模型原始输出：\n{last_raw!r}", file=sys.stderr)
            unmoved.extend(batch)
            continue

        if not mapping:
            print(f"AI 返回空映射，本批 {len(batch)} 个文件保持不动。", file=sys.stderr)
            unmoved.extend(batch)
            continue

        for fname in batch:
            rel = sanitize_rel_path(resolve_category(mapping, fname)) or OTHER_DIR
            dest_dir = os.path.join(source_dir, *rel.split("/"))
            dest = unique_dest(dest_dir, fname)
            dest_dirs.add(dest_dir)

            if rel == OTHER_DIR:
                to_other += 1

            if args.dry_run:
                print(f"[dry-run] {fname} -> {rel}/")
            else:
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(os.path.join(source_dir, fname), dest)
            moved += 1

    verb = "将" if args.dry_run else "已"
    print(
        f"\n完成：{verb}处理 {moved} 个文件，"
        f"{verb}创建 {len(dest_dirs)} 个目录，兜底 other {to_other} 个。"
    )
    if unmoved:
        print(f"\n以下 {len(unmoved)} 个文件因 AI 调用失败未处理：")
        for f in unmoved:
            print(f"  {f}")


if __name__ == "__main__":
    main()
