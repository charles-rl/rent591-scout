#!/usr/bin/env python3
"""Availability refresh + XGBoost ranking of unrated listings.

Phase 1: live-probe every active unrated listing URL through the PC devtunnel
proxy; positive delisting evidence (404/410 or DOM markers) marks it
is_active=0 in the DB. Transport errors fail open (listing stays active, listed
as unverified). Skipped entirely when --no-probe is passed or the proxy probe
fails (offline PC: stored is_active flags are trusted as-is).

Phase 2: re-score all unrated listings with the current XGBoost head
(scoring.score_all_unrated) and print top/middle/bottom N of what's available.

Usage:
    .venv/bin/python scripts/rank_unrated.py             # probe + score + report
    .venv/bin/python scripts/rank_unrated.py --no-probe  # skip liveness check (offline PC ok)
    .venv/bin/python scripts/rank_unrated.py --retrain   # retrain the head first
                                                         # (after new ratings below threshold
                                                         # or a FEATURE_NAMES width change)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# devtunnel MITMs TLS with a self-signed CA (verify=False is intentional);
# silence the per-request InsecureRequestWarning noise.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from src import database, scoring
from src.utils.health_check import DELISTED_MARKERS, mark_listing_inactive
from src.utils.proxy_check import is_proxy_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rank_unrated")

PROXY_URL = os.environ.get("PROXY_URL", "http://127.0.0.1:8999")
VERIFY_SSL = os.environ.get("PROXY_SSL_VERIFY", "0").lower() in ("1", "true", "yes")
WORKERS = int(os.environ.get("RANK_PROBE_WORKERS", "8"))
PROBE_TIMEOUT = int(os.environ.get("RANK_PROBE_TIMEOUT", "25"))


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/126.0 Safari/537.36"),
        "Accept-Language": "zh-TW,zh;q=0.9",
        # img*.591.com.tw hotlink rules expect a site referer even for page GETs.
        "Referer": "https://rent.591.com.tw/",
    })
    session.proxies.update({"http": PROXY_URL, "https": PROXY_URL})
    session.verify = VERIFY_SSL
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=2, connect=2, read=2, status=2, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"}))))
    return session


def refresh_availability(conn) -> dict:
    """Probe active unrated listings through the proxy; mark positive delisting evidence inactive."""
    rows = conn.execute(
        "SELECT listing_id, url FROM listings WHERE is_active=1 AND IFNULL(user_rated,0)!=1"
    ).fetchall()
    if not rows:
        logger.info("availability: nothing to probe")
        return {"probed": 0, "alive": 0, "delisted": [], "unverified": []}

    session = _session()

    def probe(row) -> tuple[str, str]:
        url = (row["url"] or "").strip()
        if not url:
            return row["listing_id"], "no_url"
        for attempt in range(3):
            try:
                resp = session.get(url, timeout=PROBE_TIMEOUT)
            except requests.RequestException as e:
                if attempt == 2:
                    return row["listing_id"], f"error:{type(e).__name__}"
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code in (404, 410):
                return row["listing_id"], "dead_status"
            content = resp.content
            body = content.decode("utf-8", "ignore") if isinstance(content, (bytes, bytearray)) else ""
            if any(m in body for m in DELISTED_MARKERS):
                return row["listing_id"], "delisted_marker"
            return row["listing_id"], "alive"

    stats = {"probed": len(rows), "alive": 0, "delisted": [], "unverified": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, (lid, reason) in enumerate(pool.map(probe, rows), 1):
            if reason == "alive":
                stats["alive"] += 1
            elif str(reason).startswith("error") or reason == "no_url":
                logger.warning("%s liveness unverified (%s) -> left active", lid, reason)
                stats["unverified"].append(lid)
            else:
                mark_listing_inactive(conn, lid)
                stats["delisted"].append(lid)
            if i % 25 == 0 or i == len(rows):
                logger.info("availability: %d/%d probed", i, len(rows))
    return stats


def _warn_brief(raw) -> str:
    try:
        items = [str(w)[:60] for w in json.loads(raw or "[]")]
    except (TypeError, ValueError):
        return ""
    return " | ".join(items[:3])


def report(conn, n: int = 5) -> None:
    rows = conn.execute(
        """SELECT listing_id, title, region, section, price, area, floor, kind_name,
                  predicted_score, score_source, is_duplicate, qwen_warnings
           FROM listings
           WHERE is_active=1 AND IFNULL(user_rated,0)!=1 AND predicted_score IS NOT NULL
           ORDER BY predicted_score ASC"""
    ).fetchall()
    unscored = conn.execute(
        "SELECT COUNT(*) FROM listings WHERE is_active=1 AND IFNULL(user_rated,0)!=1"
        " AND predicted_score IS NULL"
    ).fetchone()[0]

    def block(name: str, picked) -> None:
        print(f"\n{name} (of {len(rows)} available unrated)")
        for rank, r in enumerate(picked, 1):
            dup = " [DUP]" if r["is_duplicate"] else ""
            price = f"NT${r['price']:,.0f}" if r["price"] else "NT$?"
            print(f"{rank}. {r['listing_id']}  {r['predicted_score']:.2f} ({r['score_source']}){dup}")
            loc = "/".join(x for x in (r["region"], r["section"]) if x)
            print(f"   {loc or 'unknown location'}  {price} / {r['area']}ping @ {r['floor']}  {r['kind_name'] or '?'}")
            print(f"   {(r['title'] or '').strip()[:70]}")
            warn = _warn_brief(r["qwen_warnings"])
            if warn:
                print(f"   warnings: {warn}")

    if not rows:
        print("\nno available unrated listings with a predicted score")
        return
    mid = (len(rows) - 1) // 2
    block("TOP", list(reversed(rows[-n:])))
    block("MIDDLE", rows[max(0, mid - n // 2): max(0, mid - n // 2) + n])
    block("BOTTOM", rows[:n])
    if unscored:
        print(f"\nnote: {unscored} available unrated listings still have no predicted score")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-probe", action="store_true", help="skip the live availability probe")
    ap.add_argument("--retrain", action="store_true", help="retrain the XGBoost head before scoring")
    ap.add_argument("--n", type=int, default=5, help="listings per top/middle/bottom block (default 5)")
    args = ap.parse_args()

    conn = database.connect()
    try:
        if not args.no_probe:
            if is_proxy_available(PROXY_URL):
                print(f"proxy {PROXY_URL}: LIVE -> probing availability")
                stats = refresh_availability(conn)
                print(f"\navailability: probed={stats['probed']} alive={stats['alive']}"
                      f" delisted_marked_inactive={len(stats['delisted'])}"
                      f" unverified_left_active={len(stats['unverified'])}")
                if stats["delisted"]:
                    print("  delisted: " + ", ".join(sorted(stats["delisted"])))
            else:
                logger.warning("proxy %s offline -> skipping availability probe, trusting stored is_active",
                               PROXY_URL)
        else:
            print("--no-probe: skipping availability check")

        if args.retrain:
            try:
                scoring.train_and_save(conn)
            except RuntimeError as e:
                logger.warning("retrain skipped: %s (scoring with the saved head)", e)
        backfilled = scoring.score_all_unrated(conn)
        print(f"\nxgboost: re-scored {backfilled} unrated listings\n")

        report(conn, args.n)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
