#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rain check for Xuhui, Shanghai using wttr.in (free, no API key).

Uses weather *description* words (e.g. "Light rain", "Drizzle", "Thunder")
plus chance-of-rain + precipitation amount to decide whether it actually
rains, instead of a naive probability>=50% threshold that fires every day.

Outputs a rain warning only when real rain is forecast, otherwise a clear
no-rain message. Nothing on stdout means nothing to report.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone, timedelta

# Unset proxy to bypass corporate proxy
for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(var, None)

CST = timezone(timedelta(hours=8))
CITY = "Shanghai"

# Description keywords that indicate actual rain (ordered, most specific first)
RAIN_KEYWORDS = [
    "thunder", "thundery", "thunderstorm",
    "sleet", "freezing rain", "ice pellets",
    "heavy rain", "moderate rain", "light rain", "rain",
    "drizzle",
    "shower",
    "rain nearby", "patchy rain",
]

# Keywords that indicate NO rain (cloudy/clear conditions)
DRY_KEYWORDS = [
    "sunny", "clear", "cloudy", "overcast", "mist", "fog",
    "snow", "blizzard",  # snow isn't rain; ignore for a rain check
]

# English -> Chinese mapping for common descriptions
DESC_ZH = {
    "sunny": "晴",
    "clear": "晴",
    "partly cloudy": "多云",
    "cloudy": "多云",
    "overcast": "阴",
    "mist": "薄雾",
    "fog": "雾",
    "freezing fog": "冻雾",
    "patchy rain nearby": "局部小雨",
    "patchy light rain": "局部小雨",
    "light rain": "小雨",
    "moderate rain": "中雨",
    "heavy rain": "大雨",
    "light rain shower": "小阵雨",
    "moderate or heavy rain shower": "中到大阵雨",
    "patchy light drizzle": "局部毛毛雨",
    "light drizzle": "毛毛雨",
    "thundery outbreaks possible": "可能有雷暴",
    "patchy light rain with thunder": "局部雷阵雨",
}


def translate(desc: str) -> str:
    d = desc.strip().lower()
    # exact match first
    if d in DESC_ZH:
        return DESC_ZH[d]
    # partial match (longest key first)
    for key in sorted(DESC_ZH, key=len, reverse=True):
        if key in d:
            return DESC_ZH[key]
    return desc.strip()


def is_rain(desc: str, chance: float, precip_mm: float) -> bool:
    """Decide whether a slot is actual rain.

    Uses the description word as primary signal, corroborated by chance/amount.
    Avoids the false-positive where probability>=50% but 0.0mm falls.
    """
    d = desc.strip().lower()

    # Snow / ice / hail are not rain for this check
    if any(k in d for k in ("snow", "blizzard", "ice pellets", "hail")):
        return False

    # Explicit rain words -> rain, but require some support (chance or amount)
    for kw in RAIN_KEYWORDS:
        if kw in d:
            return chance >= 30 or precip_mm >= 0.5

    # No rain word -> dry, unless a significant amount is forecast
    if precip_mm >= 1.0 and chance >= 40:
        return True

    return False


def build_url() -> str:
    return f"https://wttr.in/{CITY}?format=j1&lang=zh"


def fetch() -> dict:
    req = urllib.request.Request(build_url(), headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def main() -> int:
    try:
        data = fetch()
    except Exception as e:
        print(f"获取天气数据失败: {e}", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(CST)
    date_str = now.strftime("%Y-%m-%d")

    current = data["current_condition"][0]
    today = data["weather"][0]
    max_temp = today.get("maxtempC", "?")
    min_temp = today.get("mintempC", "?")

    cur_desc = (current.get("lang_zh") or current.get("weatherDesc") or [{}])[0].get("value", "")
    cur_temp = current.get("temp_C", "?")
    cur_feels = current.get("FeelsLikeC", "?")

    # Collect future rain slots from hourly forecast
    rain_slots = []
    for h in today.get("hourly", []):
        try:
            hour = int(h["time"]) // 100  # wttr.in time is HHMM (e.g. 1300)
        except (KeyError, ValueError):
            continue

        desc = (h.get("lang_zh") or h.get("weatherDesc") or [{}])[0].get("value", "")
        chance = int(h.get("chanceofrain", 0) or 0)
        precip = float(h.get("precipMM", 0) or 0)

        # Only current + future hours
        if hour < now.hour:
            continue

        if is_rain(desc, chance, precip):
            rain_slots.append({
                "hour": hour,
                "desc": translate(desc),
                "chance": chance,
                "precip": precip,
            })

    # Today's summary description for the header
    today_desc = ""

    if not rain_slots:
        print(
            f"☀️ 今日上海无雨\n\n"
            f"📅 日期: {date_str}\n"
            f"当前: {translate(cur_desc)}，{cur_temp}°C（体感 {cur_feels}°C）\n"
            f"今日无降水预报，放心出行！\n\n"
            f"🌡️ 全天温度: {min_temp}°C ~ {max_temp}°C"
        )
        sys.exit(0)

    slots_text = "\n".join(
        f"  ⏰ {s['hour']:02d}:00 - {s['desc']}（概率 {s['chance']}%，降水量 {s['precip']}mm）"
        for s in rain_slots
    )

    print(
        f"🌧️ 今日上海有雨提醒\n\n"
        f"📅 日期: {date_str}\n"
        f"当前: {translate(cur_desc)}，{cur_temp}°C（体感 {cur_feels}°C）\n"
        f"{slots_text}\n"
        f"☔ 建议: 出门记得带伞！\n\n"
        f"🌡️ 全天温度: {min_temp}°C ~ {max_temp}°C"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
