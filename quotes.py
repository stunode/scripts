#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fetch Tencent (0700.HK) + Kweichow Moutai (600519.SH) quotes and format a report.

Sources:
- Sina HQ: https://hq.sinajs.cn/list=hk00700,sh600519 (GBK)
- FX HKD->CNY: tries multiple free endpoints

Output: human-friendly text by default, JSON with --json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import urllib.request


def shanghai_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def run(cmd: str, timeout: int = 30) -> str:
    # Keep deterministic and quiet.
    return subprocess.check_output(
        cmd,
        shell=True,
        text=True,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def fetch_sina_hq() -> dict[str, str]:
    # Sina sometimes stalls; keep strict timeouts so cron never hangs.
    # Also try both https and http in case of transient TLS/DNS issues.
    urls = [
        "https://hq.sinajs.cn/list=hk00700,sh600519",
        "http://hq.sinajs.cn/list=hk00700,sh600519",
    ]

    last_err: Exception | None = None
    for url in urls:
        cmd = (
            "curl -s "
            "--noproxy '*' "
            "--connect-timeout 5 "
            "--max-time 15 "
            "--retry 2 "
            "--retry-delay 1 "
            f"'{url}' "
            "-H 'Referer: https://finance.sina.com.cn' "
            "-H 'User-Agent: Mozilla/5.0' "
            "| iconv -f gbk -t utf-8"
        )
        try:
            raw = run(cmd, timeout=25)
        except Exception as e:
            last_err = e
            continue

        out: dict[str, str] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r'var hq_str_(\w+)="(.*)";', line)
            if m:
                out[m.group(1)] = m.group(2)
        if "hk00700" in out and "sh600519" in out:
            return out

        last_err = RuntimeError(f"Unexpected Sina payload: keys={list(out.keys())}")

    raise RuntimeError(f"Failed to fetch Sina quotes: {last_err}")


def parse_tencent_hk(payload: str) -> dict:
    # Example:
    # TENCENT,腾讯控股,455.000,456.400,457.800,445.800,449.400,-7.000,-1.534,...,2026/05/18,16:02
    parts = payload.split(",")
    if len(parts) < 10:
        raise RuntimeError(f"Tencent payload too short: {payload}")
    name_zh = parts[1]
    open_p = float(parts[2])
    prev_close = float(parts[3])
    high = float(parts[4])
    low = float(parts[5])
    last = float(parts[6])
    chg = float(parts[7])
    chg_pct = float(parts[8])
    date = parts[-2]
    time = parts[-1]
    return {
        "symbol": "0700.HK",
        "name": name_zh,
        "currency": "HKD",
        "last": last,
        "open": open_p,
        "prev_close": prev_close,
        "high": high,
        "low": low,
        "change": chg,
        "change_pct": chg_pct,
        "market_time": f"{date} {time}",
    }


def parse_moutai_cn(payload: str) -> dict:
    # Example (note the trailing comma):
    # 贵州茅台,...,2026-05-18,15:00:01,00,
    parts = payload.split(",")
    if len(parts) < 10:
        raise RuntimeError(f"Moutai payload too short: {payload}")

    name = parts[0]
    open_p = float(parts[1])
    prev_close = float(parts[2])
    last = float(parts[3])
    high = float(parts[4])
    low = float(parts[5])

    # Robustly locate date/time near the end (Sina payload sometimes ends with ",00,").
    m_date = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", payload)
    m_time = re.findall(r"\b\d{2}:\d{2}:\d{2}\b", payload)
    date = m_date[-1] if m_date else ""
    time = m_time[-1] if m_time else ""

    chg = last - prev_close
    chg_pct = (chg / prev_close * 100.0) if prev_close else math.nan
    return {
        "symbol": "600519.SH",
        "name": name,
        "currency": "CNY",
        "last": last,
        "open": open_p,
        "prev_close": prev_close,
        "high": high,
        "low": low,
        "change": chg,
        "change_pct": chg_pct,
        "market_time": (f"{date} {time}".strip() if (date or time) else ""),
    }


def http_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    return json.loads(data)


def fetch_hkd_cny_rate() -> tuple[float | None, str | None]:
    """Return (rate, source). Rate is 1 HKD -> CNY."""
    # Try multiple public endpoints; keep them simple and cache-free.
    candidates: list[tuple[str, callable]] = [
        (
            "open.er-api.com",
            lambda: (lambda j: float(j["rates"]["CNY"]))(
                http_json("https://open.er-api.com/v6/latest/HKD")
            ),
        ),
        (
            "exchangerate.host",
            lambda: (lambda j: float(j["rates"]["CNY"]))(
                http_json("https://api.exchangerate.host/latest?base=HKD&symbols=CNY")
            ),
        ),
        (
            "frankfurter.app",
            lambda: (lambda j: float(j["rates"]["CNY"]))(
                http_json("https://api.frankfurter.app/latest?from=HKD&to=CNY")
            ),
        ),
    ]

    last_err = None
    for src, fn in candidates:
        try:
            rate = fn()
            if rate and rate > 0:
                return rate, src
        except Exception as e:
            last_err = e
            continue
    return None, (str(last_err) if last_err else None)


def format_line(name: str, symbol: str, last: float, ccy: str, chg: float, chg_pct: float) -> str:
    return f"{name} {symbol}: {last:.2f} {ccy} ({chg:+.2f}, {chg_pct:+.3f}%)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="output JSON")
    args = ap.parse_args()

    now = shanghai_now()

    hq = fetch_sina_hq()
    tencent = parse_tencent_hk(hq["hk00700"])
    moutai = parse_moutai_cn(hq["sh600519"])

    fx_rate, fx_src = fetch_hkd_cny_rate()
    if fx_rate:
        tencent_cny = tencent["last"] * fx_rate
    else:
        tencent_cny = None

    payload = {
        "asof": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source": "sina_hq",
        "fx": {
            "pair": "HKD/CNY",
            "rate": fx_rate,
            "source": fx_src if fx_rate else None,
            "error": None if fx_rate else fx_src,
        },
        "tencent": {
            **tencent,
            "last_cny": tencent_cny,
        },
        "moutai": moutai,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    lines = []
    lines.append(f"腾讯 + 茅台 当前价格（{now.strftime('%m-%d %H:%M')} 更新；数据或有延时）")
    lines.append("")

    lines.append(
        "1) "
        + format_line(
            tencent["name"],
            tencent["symbol"],
            tencent["last"],
            "HKD",
            tencent["change"],
            tencent["change_pct"],
        )
    )
    lines.append(f"   高/低: {tencent['high']:.2f} / {tencent['low']:.2f}；行情时间: {tencent['market_time']}")
    if fx_rate and tencent_cny is not None:
        lines.append(f"   折合人民币: ≈ {tencent_cny:.2f} CNY（按 1 HKD = {fx_rate:.4f} CNY，源: {fx_src}）")
    else:
        lines.append("   折合人民币: （获取汇率失败，暂不可用）")

    lines.append("")

    lines.append(
        "2) "
        + format_line(
            moutai["name"],
            moutai["symbol"],
            moutai["last"],
            "CNY",
            moutai["change"],
            moutai["change_pct"],
        )
    )
    lines.append(f"   高/低: {moutai['high']:.2f} / {moutai['low']:.2f}；行情时间: {moutai['market_time']}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
