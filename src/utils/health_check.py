"""Listing liveness probes: expired-delisting detection and dead-link filtering.

Accepts either a 591 detail URL (live HTTP DOM probe) or a payload dict captured
during ingestion (offline inspection). Fail-open on transport errors: a transient
network failure is not evidence of delisting (mirrors ingestion's detail_failed
semantics), so only positive delisting signals mark a listing inactive.
"""

from __future__ import annotations

import logging
import os
import sqlite3

import requests

logger = logging.getLogger(__name__)

# DOM strings 591 renders when a listing has been pulled down or never existed.
DELISTED_MARKERS = ("物件已下架", "此物件已下架", "物件不存在", "找不到您要找的網頁")
DEAD_STATUS_CODES = (404, 410)
DEAD_STATUSES = ("closed", "sold", "delisted", "expired", "removed")
HTTP_TIMEOUT = int(os.environ.get("HEALTH_CHECK_TIMEOUT", "15"))
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "zh-TW,zh;q=0.9",
}


def _payload_inactive(payload: dict) -> bool:
    if "is_active" in payload:
        return not bool(payload["is_active"])
    code = payload.get("status_code") or payload.get("http_status")
    if code in DEAD_STATUS_CODES:
        return True
    if str(payload.get("status") or "").lower() in DEAD_STATUSES:
        return True
    for key in ("html", "detail_html", "body", "text"):
        blob = payload.get(key)
        if isinstance(blob, str) and any(m in blob for m in DELISTED_MARKERS):
            return True
    return False


def is_listing_active(url_or_payload: str | dict) -> bool:
    """Return False only on positive evidence the listing is dead/expired.

    dict payload  -> inspect status/http code/DOM markers, no network call.
    str url       -> HTTP GET and scan status code + rendered DOM text.
    Unreachable URLs / probe errors are treated as ACTIVE (fail-open).
    """
    if isinstance(url_or_payload, dict):
        return not _payload_inactive(url_or_payload)
    url = str(url_or_payload or "").strip()
    if not url:
        return False
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        logger.warning("health probe unreachable %s: %s -> assume active", url, e)
        return True
    if resp.status_code in DEAD_STATUS_CODES:
        return False
    try:
        body = resp.text or ""
    except Exception:  # pragma: no cover - malformed body decoding
        body = ""
    return not any(m in body for m in DELISTED_MARKERS)


def mark_listing_inactive(conn: sqlite3.Connection, listing_id: str) -> bool:
    """Flag a stored listing as delisted (is_active = FALSE). Returns True if a row changed."""
    cur = conn.execute(
        "UPDATE listings SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP WHERE listing_id = ?",
        (str(listing_id),),
    )
    conn.commit()
    return cur.rowcount > 0
