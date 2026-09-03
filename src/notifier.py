"""ntfy.sh push notification publisher."""

from __future__ import annotations

import logging
import os
import re

import requests

logger = logging.getLogger(__name__)

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "rent591-scout")
NTFY_URL = os.environ.get("NTFY_URL", "https://ntfy.sh")


def _header_safe(value: str) -> str:
    """HTTP header values are latin-1 encoded by urllib3; drop anything else."""
    return str(value or "").encode("latin-1", "ignore").decode("latin-1").strip()


def _post(message: str, headers: dict, proxy: str | None = None) -> bool:
    """POST to ntfy. With a proxy configured: tunnel-first, then direct fallback.

    The devtunnel path is the only way to reach ntfy.sh from the GPU server
    while the PC is up; the direct fallback covers the probe-said-live-but-
    tunnel-died race. When the PC is truly powered off neither path can
    deliver — the alert is best-effort by design (the queue persists).
    """
    attempts = [proxy, None] if proxy else [None]
    for px in attempts:
        try:
            resp = requests.post(
                f"{NTFY_URL}/{NTFY_TOPIC}", data=message.encode("utf-8"),
                headers=headers, timeout=15,
                proxies={"http": px, "https": px} if px else None,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning("ntfy push failed via %s: %s", px or "direct", e)
    return False


_SECTION_EN = {
    "汐止區": "Xizhi", "三重區": "Sanchong", "南港區": "Nangang", "內湖區": "Neihu",
    "北投區": "Beitou", "大同區": "Datong", "士林區": "Shilin", "蘆洲區": "Luzhou",
    "淡水區": "Tamsui", "板橋區": "Banqiao",
    "台北市": "Taipei", "新北市": "New Taipei",
}
_KIND_EN = {
    "套房": "private suite", "整層住家": "whole flat", "分租套房": "suite in shared house",
    "雅房": "partition room", "店面": "storefront", "事務所": "office",
}
_SHAPE_EN = {
    "電梯大樓": "elevator building", "公寓": "walk-up", "透天厝": "townhouse",
    "大廈": "multistory building", "別墅": "villa",
}
_WARN_EN = [
    ("衛浴獨立性未確認", "Bathroom privacy unconfirmed"),
    ("頂樓加蓋疑慮", "Suspected illegal rooftop addition"),
    ("無窗/採光不足", "No window / poor light"),
    ("採光不足", "Poor light"),
    ("共用卫浴", "Shared bathroom"),
    ("共用衛浴", "Shared bathroom"),
    ("無窗", "No window"),
    ("頂樓：可能炎熱/漏水", "Top floor: heat/leak risk"),
    ("一樓：注意採光與隱私", "1st floor: light & privacy concerns"),
    ("水電/管理費另計", "Utilities/management fees extra"),
    ("要求半年付", "6-month upfront payment required"),
    ("要求季付", "Quarterly upfront payment required"),
    ("付款規則", "Unusual payment rule"),
    ("電費超過 5 元/度", "Electricity billed above 5 NTD/kWh"),
    ("禁止養寵", "No pets allowed"),
    ("5樓以上無電梯", "5F+ walk-up, no elevator"),
    ("需追垃圾車", "Manual trash disposal (garbage-truck chasing)"),
    ("共享/投幣洗衣", "Shared / coin-operated laundry"),
]


def _en(text: str) -> str:
    """Best-effort translation of the pipeline's fixed Chinese warning vocabulary."""
    text = str(text or "").replace("：", ": ")
    for zh, en in _WARN_EN:
        if zh in text:
            return text.replace(zh, en)
    return text


def _en_line(*parts) -> str:
    return " · ".join(str(p) for p in parts if p)


def _summary(listing: dict) -> str:
    """Structured English one-liner (591 titles/addresses are Chinese and untranslatable here)."""
    layout = str(listing.get("layout") or "").replace("房", "BR ").replace("廳", "LR ").replace("衛浴", "BA").replace("衛", "BA")
    floor = str(listing.get("floor") or "").replace("頂樓", "top floor").replace("頂層", "top floor").replace("樓", "F")
    section = _SECTION_EN.get(listing.get("section") or "", listing.get("section") or "")
    region = _SECTION_EN.get(listing.get("region") or "", "")
    where = _en_line(section, region)
    area = listing.get("area")
    return _en_line(
        _KIND_EN.get(listing.get("kind_name") or "", listing.get("kind_name")),
        layout.strip(), floor, _SHAPE_EN.get(listing.get("shape") or "", listing.get("shape")),
        f"{float(area):g} ping" if area else None, where,
    )


_POST_MONTH_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_POST_REL_RE = re.compile(r"(\d+)\s*(分鐘|分钟|小時|小时|天|週|周)\s*前")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_REL_EN = {"分鐘": "min", "分钟": "min", "小時": "hour", "小时": "hour", "天": "day", "週": "week", "周": "week"}


def _posted_en(refresh_time) -> str:
    """591 refresh_time is Chinese free text ('此房屋在8月18日發佈') -> English label or ''."""
    text = str(refresh_time or "")
    verb = "Updated" if "更新" in text else "Posted"
    m = _POST_MONTH_RE.search(text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{verb} {_MONTHS[month - 1]} {day}"
    m = _POST_REL_RE.search(text)
    if m:
        n, unit = int(m.group(1)), _REL_EN[m.group(2)]
        return f"{verb} {n} {unit}{'s' if n != 1 else ''} ago"
    return ""


def send_ntfy_alert(listing: dict, predicted_score: float | None = None, threshold: float = 3.5,
                    proxy: str | None = None) -> bool:
    """Push an ntfy.sh alert for a high-scoring match. Non-fatal: returns False on failure.

    proxy: forward through the PC devtunnel (ntfy.sh is blocked from the GPU
    server directly; the tunnel path works while the PC is online).
    """
    if predicted_score is None:
        predicted_score = float(listing.get("predicted_score") or 0.0)
    if predicted_score < threshold:
        return False
    warnings = [_en(w) for w in (listing.get("qwen_warnings") or [])]
    warnings_str = " | ".join(warnings) if warnings else "No major issues"
    url = str(listing.get("url") or "")
    price = listing.get("price") or "—"
    posted = _posted_en(listing.get("refresh_time"))
    price_line = f"NT${price}/mo | Rating {predicted_score:.2f}/5"
    if posted:
        price_line += f" | {posted}"
    message = (
        f"{_summary(listing)}\n"
        f"{price_line}\n"
        f"⚠️ {warnings_str}\n"
        f"{url}"
    )
    headers = {
        "Title": _header_safe(f"Apartment Match ({predicted_score:.2f}/5)"),
        "Click": _header_safe(url),
        "Tags": "house,bathroom",
    }
    return _post(message, headers, proxy=proxy)


def send_proxy_request_alert(pending_count: int, proxy: str | None = None) -> bool:
    """Ask the user to power on the PC and start the devtunnel so queued
    listing images can be fetched. Non-fatal: returns False on failure."""
    if pending_count <= 0:
        return False
    message = (
        f"🏠 [591 Monitor] {pending_count} new listings queued for vision processing. "
        "Connect PC proxy (port 8999) to process images."
    )
    headers = {
        "Title": _header_safe("PC proxy needed"),
        "Tags": "house,warning",
        "Priority": "high",
    }
    return _post(message, headers, proxy=proxy)
