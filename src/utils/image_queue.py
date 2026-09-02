"""Pending-image drain through the PC devtunnel proxy (hybrid mode).

Listings ingested while the PC proxy was offline carry image_status='pending'.
Once the tunnel is back up, process_pending_images() downloads their photos
via the proxy, re-encodes them to WebP q=85 under data/images/<listing_id>/,
flips image_status to 'completed'/'failed', and hands the fresh image rows
back to the caller so the vision/scoring pass can run.

591's CDN answers 403 to the *stripped* original URLs through the tunnel and
only serves resize variants, so bare img*.591.com.tw URLs get a
PROXY_IMAGE_SUFFIX (default "!fit.1000x.water2.jpg" — 1000px watermarked,
served as WebP) appended before fetching.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .. import database
from ..ingestion import IMAGES_DIR, _valid_webp

logger = logging.getLogger(__name__)

DEFAULT_PROXY_URL = os.environ.get("PROXY_URL", "http://127.0.0.1:8999")
PROXY_IMAGE_SUFFIX = os.environ.get("PROXY_IMAGE_SUFFIX", "!fit.1000x.water2.jpg")
VERIFY_SSL = os.environ.get("PROXY_SSL_VERIFY", "0").lower() in ("1", "true", "yes")
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
DOWNLOAD_TIMEOUT = int(os.environ.get("PROXY_DOWNLOAD_TIMEOUT", "30"))
# CDN/tunnel throttling (502 storms): back off instead of treating it as a
# dead proxy, and stop politely so the next cron pass resumes where we left off.
RATE_LIMIT_BACKOFF = int(os.environ.get("PROXY_RATE_LIMIT_BACKOFF", "20"))
RATE_LIMIT_MAX_STREAK = int(os.environ.get("PROXY_RATE_LIMIT_MAX_STREAK", "3"))
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    ),
    # img1/img2.591.com.tw enforce hotlink protection: image GETs need the site referer.
    "Referer": "https://rent.591.com.tw/",
    "Accept": "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}


def _make_session(proxy_url: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(_HEADERS)
    session.proxies.update({"http": proxy_url, "https": proxy_url})
    session.verify = VERIFY_SSL
    # Transient hiccups (429/5xx/connect) retry; 403/404 are persistent and don't.
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=2, connect=2, read=2, status=2, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"}),
    )))
    return session


def _fetch_url(url: str) -> str:
    """Bare 591 CDN originals 403 through the tunnel; request the resize variant."""
    if PROXY_IMAGE_SUFFIX and "!" not in url and "//img" in url and url.lower().endswith((".jpg", ".jpeg", ".png")):
        return url + PROXY_IMAGE_SUFFIX
    return url


PLACEHOLDER_MAX_BYTES = int(os.environ.get("PLACEHOLDER_MAX_BYTES", "4096"))


def looks_placeholder(path: Path) -> bool:
    """True for synthesized solid-color WebPs left by PLACEHOLDER_IMAGES runs.

    Fixture placeholders are 800x600 flat colors (~1KB, <=16 distinct colors);
    real photos are far larger and full-color. Used so the download cache never
    skips a fake in place of a real CDN fetch.
    """
    try:
        if path.stat().st_size > PLACEHOLDER_MAX_BYTES:
            return False
        with Image.open(path) as img:
            colors = img.convert("RGB").getcolors(maxcolors=16)
        return colors is not None and len(colors) <= 16
    except Exception:
        return True


def _download_one(session: requests.Session, url: str, dest: Path) -> bool:
    """Fetch url through the proxy and save as WebP q=85. Returns True on success.

    Raises requests.ConnectionError (via retry adapter exhaustion) when the
    tunnel itself died mid-run — caller distinguishes that from persistent
    per-image HTTP errors (403/404).
    """
    if dest.exists() and _valid_webp(dest) and not looks_placeholder(dest):
        return True
    try:
        resp = session.get(_fetch_url(url), timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
    except requests.HTTPError as e:
        # 403/404 from the CDN: persistent for this image, the tunnel is fine.
        logger.warning("image fetch rejected %s: %s", url, e)
        return False
    if "image" not in (resp.headers.get("content-type", "") or "").lower():
        logger.warning("unexpected content-type %r for %s", resp.headers.get("content-type"), url)
        return False
    if len(resp.content) > MAX_IMAGE_BYTES:
        logger.warning("payload too large (%d bytes) for %s", len(resp.content), url)
        return False
    try:
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.warning("unreadable image %s: %s", url, e)
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, "WEBP", quality=85)
    finally:
        img.close()
    return True


def _process_listing(conn, row, session: requests.Session) -> tuple[str, list[dict], str | None]:
    """Drain one pending listing's images.

    Returns (outcome, image_rows, stop_reason). Outcome is 'completed' or
    'skipped' (ready for the vision pass), 'failed', or 'pending' (left in
    queue). stop_reason: 'tunnel_down' (PC offline -> stop the run),
    'rate_limited' (CDN 5xx exhaustion -> caller backs off and continues),
    or None.
    """
    lid = str(row["listing_id"])
    try:
        urls = json.loads(row["image_urls"] or "[]")
    except (TypeError, ValueError):
        urls = []
    if not urls:
        # Nothing to download: text-only listing, straight to vision.
        database.set_image_status(conn, lid, "skipped")
        return "skipped", [], None

    out_dir = IMAGES_DIR / lid
    rows: list[dict] = []
    ok_count = 0
    stop_reason: str | None = None
    for ordinal, url in enumerate(urls):
        dest = out_dir / f"{ordinal:02d}.webp"
        try:
            ok = _download_one(session, url, dest)
        except requests.exceptions.RetryError as e:
            # CDN/tunnel exhaustion (e.g. "too many 502") — the proxy is alive
            # but throttling; leave pending and let the caller back off.
            logger.warning("rate-limited mid-download of %s/%s: %s -> leaving pending", lid, ordinal, e)
            stop_reason = "rate_limited"
            break
        except (requests.exceptions.ProxyError, requests.exceptions.SSLError,
                requests.exceptions.ConnectionError) as e:
            logger.warning("proxy transport failure on %s/%s: %s", lid, ordinal, e)
            stop_reason = "tunnel_down"
            break
        except requests.Timeout as e:
            logger.warning("proxy timeout on %s/%s: %s -> leaving pending", lid, ordinal, e)
            stop_reason = "rate_limited"
            break
        if ok:
            ok_count += 1
            rows.append({"ordinal": ordinal, "image_url": url, "image_path": str(dest)})
        else:
            rows.append({"ordinal": ordinal, "image_url": url, "image_path": None})

    if stop_reason:
        return "pending", [], stop_reason
    if ok_count == 0:
        logger.warning("%s: all %d image downloads failed -> image_status=failed", lid, len(urls))
        database.set_image_status(conn, lid, "failed")
        return "failed", [], None

    for r in rows:
        r.setdefault("dino_embedding", None)
    database.replace_images(conn, lid, rows)
    database.set_image_status(conn, lid, "completed", image_paths=[r["image_path"] for r in rows])
    logger.info("%s: %d/%d images downloaded -> image_status=completed", lid, ok_count, len(urls))
    return "completed", rows, None


def process_pending_images(conn, proxy_url: str = DEFAULT_PROXY_URL) -> list[tuple[str, list[dict]]]:
    """Download queued images for active pending listings via the proxy.

    Returns [(listing_id, image_rows)] for listings that reached
    'completed'/'skipped' in this call, ready for the vision/scoring pass.
    """
    pending = database.get_pending_image_listings(conn)
    if not pending:
        logger.info("image queue empty: no pending listings")
        return []
    logger.info("image queue: %d pending listings via %s", len(pending), proxy_url)
    session = _make_session(proxy_url)
    results: list[tuple[str, list[dict]]] = []
    throttle_streak = 0
    for row in pending:
        try:
            outcome, rows, stop_reason = _process_listing(conn, row, session)
        except Exception:
            logger.exception("image drain failed for %s -> image_status=failed", row["listing_id"])
            database.set_image_status(conn, str(row["listing_id"]), "failed")
            continue
        if outcome in ("completed", "skipped"):
            results.append((str(row["listing_id"]), rows))
            throttle_streak = 0
        elif stop_reason == "rate_limited":
            throttle_streak += 1
            if throttle_streak >= RATE_LIMIT_MAX_STREAK:
                logger.warning("throttled %d listings in a row -> stopping; resumes next run",
                               throttle_streak)
                break
            logger.info("rate limited -> backing off %ss", RATE_LIMIT_BACKOFF)
            time.sleep(RATE_LIMIT_BACKOFF)
        elif stop_reason == "tunnel_down":
            break
    return results
