"""ntfy.sh push notification publisher."""

from __future__ import annotations

import os

import requests

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "rent591-scout")
NTFY_URL = os.environ.get("NTFY_URL", "https://ntfy.sh")


def send_ntfy_alert(listing: dict, predicted_score: float, threshold: float = 3.5) -> bool:
    if predicted_score < threshold:
        return False
    warnings = listing.get("qwen_warnings") or []
    warnings_str = " | ".join(warnings) if warnings else "No major issues"
    message = f"NT${listing.get('price')} - {listing.get('title')}\n⚠️ {warnings_str}"
    headers = {
        "Title": f"Apartment Match ({predicted_score:.1f}/5)",
        "Click": listing.get("url") or "",
        "Tags": "house,bathroom",
    }
    try:
        resp = requests.post(f"{NTFY_URL}/{NTFY_TOPIC}", data=message.encode("utf-8"), headers=headers, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[notifier] ntfy push failed: {e}")
        return False
