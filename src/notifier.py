"""ntfy.sh push notification publisher."""

from __future__ import annotations

import logging
import os

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
    warnings = listing.get("qwen_warnings") or []
    warnings_str = " | ".join(str(w) for w in warnings) if warnings else "No major issues"
    url = str(listing.get("url") or "")
    message = (
        f"{listing.get('title') or 'Untitled listing'}\n"
        f"NT${listing.get('price') or '—'} | Rating {predicted_score:.1f}/5\n"
        f"⚠️ {warnings_str}\n"
        f"{url}"
    )
    headers = {
        "Title": _header_safe(f"Apartment Match ({predicted_score:.1f}/5)"),
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
