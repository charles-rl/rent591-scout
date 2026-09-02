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


def send_ntfy_alert(listing: dict, predicted_score: float | None = None, threshold: float = 3.5) -> bool:
    """Push an ntfy.sh alert for a high-scoring match. Non-fatal: returns False on failure."""
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
    try:
        resp = requests.post(f"{NTFY_URL}/{NTFY_TOPIC}", data=message.encode("utf-8"), headers=headers, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("ntfy push failed: %s", e)
        return False
