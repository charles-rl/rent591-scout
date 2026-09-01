"""Unified ingestion: mcp-591 API (primary) + 591scraper DOM fallback + WebP image pipeline.

API-first with full raw payload retention (zero data loss), see docs/data-maxification.md.
Supports an offline `--fixtures` mode replaying captured 591 responses for testing.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .client591 import Client591
from .constants591 import REGIONS, RENT_KINDS, SECTIONS, SECTIONS_BY_REGION

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(os.environ.get("FIXTURES_DIR", ROOT / "external" / "mcp-591" / "tests" / "fixtures"))
IMAGES_DIR = Path(os.environ.get("IMAGES_DIR", ROOT / "data" / "images"))
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36")
VERIFY_SSL = os.environ.get("RENT591_SSL_VERIFY", "1").lower() not in ("0", "false", "no")
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": _UA,
    # img1/img2.591.com.tw enforce hotlink protection: image GETs need the site referer.
    "Referer": "https://rent.591.com.tw/",
    "Accept": "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
})
_SESSION.verify = VERIFY_SSL
_IMAGE_RETRY = Retry(
    total=2, connect=2, read=2, status=2, backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
)
_SESSION.mount("https://", HTTPAdapter(max_retries=_IMAGE_RETRY))


# --------------------------------------------------------------------------
# Query config
# --------------------------------------------------------------------------
def _resolve_region_sections(region_name: str, section_name: str) -> tuple[int, list[int]]:
    region_id = next((rid for rid, name in REGIONS.items() if name == region_name), None)
    if region_id is None:
        raise ValueError(f"unknown region: {region_name!r}; available: {list(REGIONS.values())}")
    if section_name:
        section_ids = [
            sid for sid, (sname, rid) in SECTIONS.items() if sname == section_name and rid == region_id
        ]
        if not section_ids:
            raise ValueError(f"unknown section {section_name!r} in {region_name}")
    else:
        section_ids = list(SECTIONS_BY_REGION[region_id].keys())
    return region_id, section_ids


def _resolve_kind(kind_name: str | None) -> int | None:
    if not kind_name:
        return None
    match = next((k for k, v in RENT_KINDS.items() if v == kind_name), None)
    if match is None:
        raise ValueError(f"unknown kind {kind_name!r}; available: {list(RENT_KINDS.values())}")
    return match


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------
def fetch_raw_listings(fixtures: bool = False, limit: int = -1) -> list[dict]:
    """Return list of {"raw_search": item, "raw_metadata": data|None}."""
    region_name = os.environ.get("X591_REGION", "台北市")
    section_name = os.environ.get("X591_SECTION", "")
    kind_name = os.environ.get("X591_KIND", "整層住家")
    price_str = os.environ.get("X591_PRICE_STR", "15000_25000")
    first_pages = int(os.environ.get("X591_FIRST_PAGES", "1"))

    if fixtures:
        return _fetch_from_fixtures(limit)

    region_id, section_ids = _resolve_region_sections(region_name, section_name)
    kind = _resolve_kind(kind_name)
    client = Client591()
    items: list[dict] = []
    first_row = 0
    for _ in range(first_pages):
        result = client.search_rent(
            region_id=region_id, section_ids=section_ids, kind=kind,
            price_str=price_str, first_row=first_row,
        )
        data = result.get("data", {})
        page_items = data.get("items", [])
        items.extend(page_items)
        if not page_items:
            break
        next_row = data.get("firstRow")
        try:
            next_row = int(next_row)
        except (TypeError, ValueError):
            logger.warning("unexpected firstRow=%r; stopping pagination", data.get("firstRow"))
            break
        if next_row == first_row:
            break
        first_row = next_row

    out: list[dict] = []
    for item in items:
        pid = str(item.get("id"))
        meta = None
        detail_failed = False
        try:
            resp = client.get_rent_detail(pid)
            d = resp.get("data")
            meta = d if isinstance(d, dict) and d else None
        except Exception as e:
            detail_failed = True
            logger.warning("detail fetch failed for %s: %s", pid, e)
        out.append({"raw_search": item, "raw_metadata": meta, "detail_failed": detail_failed})
        if limit > 0 and len(out) >= limit:
            break
        if not detail_failed:
            time.sleep(random.uniform(0.5, 2.0))
    return out


def _fetch_from_fixtures(limit: int) -> list[dict]:
    with open(FIXTURES_DIR / "search_rent.json") as f:
        items = json.load(f)["data"]["items"]
    with open(FIXTURES_DIR / "rent_detail.json") as f:
        detail = json.load(f)["data"]

    fixture_id = str(items[0]["id"]) if items else None
    out = []
    for i, item in enumerate(items):
        has_detail = fixture_id and str(item["id"]) == fixture_id
        meta = detail if has_detail else None
        out.append({"raw_search": item, "raw_metadata": meta, "detail_failed": not has_detail})
        if limit > 0 and len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def _int_price(raw: str | int | None) -> int:
    if isinstance(raw, int):
        return raw
    if not raw:
        return 0
    m = re.search(r"\d[\d,]*", str(raw))
    return int(m.group(0).replace(",", "")) if m else 0


def _float_price(raw: str | int | float | None) -> float | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if not raw:
        return None
    m = re.search(r"\d[\d.,]*", str(raw))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _info_map(data: dict) -> dict:
    return {x.get("key"): x.get("value") for x in data.get("info", [])}


def normalize_listing(raw_search: dict, raw_metadata: dict | None,
                      detail_failed: bool = False) -> dict:
    item = raw_search or {}
    data = raw_metadata or {}
    info = _info_map(data)
    gtm = data.get("gtm_detail_data", {})
    addr = data.get("address", {}) or {}
    cost = {x.get("key"): x.get("value") for x in (data.get("cost") or {}).get("data", [])}

    raw_id = item.get("id")
    pid = str(raw_id) if raw_id is not None else str(gtm.get("item_id", "")).removeprefix("R")
    title = item.get("title") or data.get("title") or ""
    price = _int_price(item.get("price") or data.get("price"))
    lat = None
    lng = None
    if addr.get("lat"):
        lat, lng = float(addr["lat"]), float(addr["lng"])
    else:
        pos = data.get("positionRound") or {}
        if pos.get("lat") is not None:
            lat, lng = float(pos["lat"]), float(pos["lng"])

    area_raw = item.get("area") or info.get("area")
    area = None
    if isinstance(area_raw, (int, float)):
        area = float(area_raw)
    elif area_raw:
        m = re.search(r"[\d.]+", str(area_raw))
        if m:
            area = float(m.group(0))

    description = (data.get("remark") or {}).get("content") or ""
    contain = data.get("containCost") or []
    item_tags = [t.get("value") if isinstance(t, dict) else t for t in (item.get("tags") or [])]
    detail_tags = []
    if isinstance(data.get("tags"), list):
        detail_tags = [t.get("value", "") for t in data["tags"] if isinstance(t, dict)]
    tags = list(dict.fromkeys(str(t).strip() for t in item_tags + detail_tags if t))

    facility = data.get("service") or {}
    facilities = [
        f.get("name") for f in facility.get("facility", [])
        if isinstance(f, dict) and f.get("active")
    ]
    social_house = item.get("social_house") or (data.get("favData") or {}).get("socialHouse")
    social_house = bool(int(social_house)) if isinstance(social_house, (int, str)) else None

    if data:
        status = data.get("status") or "open"
        active = True
    elif detail_failed:
        status = None  # unknown — network/rate-limit failure, not evidence of delisting
        active = True  # keep processing; COALESCE preserves previously stored values
    else:
        status = "closed"
        active = False

    return {
        "listing_id": pid,
        "title": title,
        "price": price,
        "price_unit": item.get("price_unit") or data.get("priceUnit") or "元/月",
        "url": f"https://rent.591.com.tw/{pid}",
        "status": status,
        "is_active": active,
        "region": gtm.get("region_name") or (REGIONS.get(item.get("regionid")) if item.get("regionid") else None),
        "section": gtm.get("section_name"),
        "address": addr.get("data") or item.get("address") or "",
        "lat": lat,
        "lng": lng,
        "community_name": item.get("community_name") or (data.get("positionRound") or {}).get("communityName"),
        "community_id": item.get("community_id") or (data.get("positionRound") or {}).get("communityId"),
        "layout": item.get("layoutStr") or info.get("layout"),
        "area": area,
        "floor": item.get("floor_name") or info.get("floor"),
        "shape": info.get("shape"),
        "kind_name": item.get("kind_name") or gtm.get("kind_name"),
        "deposit": data.get("deposit") or cost.get("deposit") or "",
        "rent_per": _float_price(item.get("price_per")),
        "rent_per_unit": item.get("price_per_unit") or "元/坪/月",
        "browse_count": item.get("browse_count"),
        "refresh_time": item.get("refresh_time") or (data.get("publish") or {}).get("postTime"),
        "tags": tags,
        "contain_cost": contain,
        "social_house": social_house,
        "facilities": facilities,
        "description": description or "",
        "raw_search": item,
        "raw_metadata": data,
        "scraper_raw": None,
    }


def fetch_image_urls(raw_search: dict, raw_metadata: dict | None) -> list[str]:
    """Primary photoList (resize suffix stripped), fallback cover/ogimage."""
    urls = list(raw_search.get("photoList") or [])
    if raw_metadata:
        meta = raw_metadata.get("meta", {}) or {}
        fav = (raw_metadata.get("favData") or {}).get("thumb")
        for u in (raw_metadata.get("cover"), meta.get("ogimage"), fav):
            if u:
                urls.append(u)
    seen: list[str] = []
    for u in urls:
        s = _strip_suffix(u)
        if s and s not in seen:
            seen.append(s)
    return seen


def _strip_suffix(url: str) -> str:
    if "!" in url:
        return url.split("!")[0]
    return url


def _valid_webp(path: Path) -> bool:
    """Reject zero-byte / truncated cache files so they get re-downloaded."""
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _grab_webp(url: str, dest: Path, placeholder: bool = False, key: str = "") -> None:
    """Fetch url (or synthesize a solid-color WebP when placeholder=True), save at q=85."""
    if placeholder:
        color = int(hashlib.md5(key.encode()).hexdigest()[:6], 16)
        img = Image.new("RGB", (800, 600), (color >> 16 & 255, color >> 8 & 255, color & 255))
    else:
        resp = _SESSION.get(url, timeout=30)
        resp.raise_for_status()
        if "image" not in (resp.headers.get("content-type", "") or "").lower():
            raise ValueError(f"unexpected content-type: {resp.headers.get('content-type')}")
        if len(resp.content) > MAX_IMAGE_BYTES:
            raise ValueError(f"payload too large: {len(resp.content)} bytes")
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    try:
        img.save(dest, "WEBP", quality=85)
    finally:
        img.close()


def download_images(listing_id: str, urls: list[str], placeholder: bool = False) -> list[dict]:
    """Download originals, re-encode to WebP quality=85, return image row dicts.

    When placeholder=True (offline fixtures testing), generate solid-color WebP
    placeholders instead of fetching from the blocked 591 CDN.

    Rows are appended for every ordinal (failed downloads carry image_path=None)
    so image_urls/image_paths/listing_images stay aligned (data-maxification.md:105).
    """
    if not urls:
        return []
    out_dir = IMAGES_DIR / listing_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for ordinal, url in enumerate(urls):
        path = out_dir / f"{ordinal:02d}.webp"
        try:
            if not path.exists() or not _valid_webp(path):
                _grab_webp(url, path, placeholder=placeholder, key=f"{listing_id}:{ordinal}")
            rows.append({"ordinal": ordinal, "image_url": url, "image_path": str(path)})
        except Exception as e:
            logger.warning("image %s for %s failed: %s", ordinal, listing_id, e)
            rows.append({"ordinal": ordinal, "image_url": url, "image_path": None})
    return rows


# --------------------------------------------------------------------------
# 591scraper DOM fallback
# --------------------------------------------------------------------------
def scraper_fallback(listing_id: str) -> dict | None:
    """DrissionPage detail scrape. Returns DOM dict or None on any failure."""
    try:
        from DrissionPage import ChromiumPage, ChromiumOptions  # non-commercial license
    except Exception:
        return None
    try:
        co = ChromiumOptions()
        binary = os.environ.get("CHROME_BINARY")
        if binary:
            co.set_browser_path(binary)
        profile = os.environ.get("BROWSER_PROFILE", str(ROOT / "browser_profile"))
        co.set_paths(user_data_path=profile)
        co.set_argument("--disable-blink-features=AutomationControlled")
        co.set_argument("--no-first-run")
        page = ChromiumPage(co)
        try:
            page.get(f"https://rent.591.com.tw/home/{listing_id}")
            page.wait.eles_loaded("css:div.title", timeout=10)

            result: dict = {"id": listing_id}
            h1 = page.ele("css:.title h1", timeout=3)
            result["title"] = h1.text.strip() if h1 else ""
            addr_el = page.ele("css:div.address div", timeout=3)
            result["addr"] = addr_el.text.strip() if addr_el else ""
            complex_el = page.ele("css:div.address p a", timeout=2)
            if complex_el:
                result["社區"] = complex_el.text.strip()
            price_el = page.ele("css:div.house-price", timeout=3)
            result["price"] = _scrape_price(price_el.text if price_el else "")
            desc_el = page.ele("css:div.house-condition-content", timeout=3)
            result["desc"] = desc_el.text.strip() if desc_el else ""
            poster_el = page.ele("css:p.base-info-pc", timeout=3)
            result["poster"] = re.sub(r"\s+", " ", poster_el.text.strip()) if poster_el else ""

            service_el = page.ele("css:section.service", timeout=2)
            result["養寵物"] = (
                "No" if service_el and "不可養寵物" in (service_el.text or "") else ("Yes" if service_el else None)
            )
            for label_name in ("租金含", "車位費", "管理費"):
                label_el = page.ele(f"text={label_name}", timeout=1)
                if label_el:
                    parent = label_el.parent()
                    text_el = parent.ele("css:div.text", timeout=1) if parent else None
                    result[label_name] = text_el.text.strip() if text_el else ""
                else:
                    result[label_name] = ""
            facility_el = page.ele("css:div.service-facility", timeout=2)
            if facility_el:
                items = facility_el.eles("css:dl:not(.del) dd")
                result["提供設備"] = ", ".join(i.text.strip() for i in items if i.text)
            else:
                result["提供設備"] = ""
            return result
        finally:
            try:
                page.quit()
            except Exception:
                pass
    except Exception as e:
        logger.warning("scraper fallback %s failed: %s", listing_id, e)
        return None


def _scrape_price(price_str: str) -> int:
    if not price_str or "--" in price_str or "無" in price_str:
        return 0
    m = re.match(r"^([\d,]+)", price_str)
    return int(m.group(1).replace(",", "")) if m else 0


def apply_scraper(listing: dict, scraper: dict | None) -> dict:
    if not scraper:
        return listing
    listing["scraper_raw"] = scraper
    if not listing.get("description"):
        listing["description"] = scraper.get("desc", "")
    if not listing.get("address"):
        listing["address"] = scraper.get("addr", "")
    if not listing.get("title"):
        listing["title"] = scraper.get("title", "")
    if not listing.get("price"):
        listing["price"] = scraper.get("price", 0)
    if not listing.get("community_name"):
        listing["community_name"] = scraper.get("社區", "")
    fac = list(listing.get("facilities") or [])
    if scraper.get("提供設備"):
        for item in str(scraper["提供設備"]).replace("，", ",").split(","):
            item = item.strip()
            if item and item not in fac:
                fac.append(item)
    pet = scraper.get("養寵物")
    if pet == "Yes" and "可養寵物" not in fac:
        fac.append("可養寵物")
    elif pet == "No" and "不可養寵物" not in fac:
        fac.append("不可養寵物")
    listing["facilities"] = fac
    if not listing.get("contain_cost") and scraper.get("租金含"):
        listing["contain_cost"] = [scraper["租金含"]]
    return listing


# --------------------------------------------------------------------------
# GitHub Actions cron relay: dump raw payloads + WebP images to data/incoming/
# --------------------------------------------------------------------------
# The GPU server cannot reach rent.591.com.tw / ntfy.sh (egress firewall), but
# github.com is whitelisted. This relay mode runs on a GitHub Actions runner:
# it scrapes 591, writes per-listing raw JSON + WebP images under an output
# dir, and the workflow commits them back to the repo. The GPU server then
# `git pull`s and runs main.py --incoming fully offline.
#
#   Layout:
#     <output_dir>/listings/<id>.json   raw_search + raw_metadata + manifest
#     <output_dir>/images/<id>/<ord>.webp
#
# Files are written atomically and skipped when the payload hash is unchanged,
# so repeated runs produce no git churn.
def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def dump_relay_payloads(output_dir: str | Path, fixtures: bool = False,
                        limit: int = -1, placeholder: bool = False) -> int:
    """Scrape (or replay fixtures) and persist raw JSON + WebP for git relay."""
    out = Path(output_dir)
    (out / "listings").mkdir(parents=True, exist_ok=True)
    skip_images = os.environ.get("RELAY_SKIP_IMAGES", "").lower() in ("1", "true", "yes")
    entries = fetch_raw_listings(fixtures=fixtures, limit=limit)
    written = skipped = 0
    for entry in entries:
        item = entry.get("raw_search") or {}
        pid = str(item.get("id") or "")
        if not pid or pid == "None":
            continue
        urls = fetch_image_urls(item, entry.get("raw_metadata"))
        webp_blobs: dict[int, bytes] = {}
        if not skip_images:
            for ordinal, url in enumerate(urls):
                dest = out / "images" / pid / f"{ordinal:02d}.webp"
                try:
                    if dest.exists() and _valid_webp(dest):
                        webp_blobs[ordinal] = dest.read_bytes()
                        continue
                    with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tf:
                        tmp_path = Path(tf.name)
                    try:
                        _grab_webp(url, tmp_path, placeholder=placeholder, key=f"{pid}:{ordinal}")
                        webp_blobs[ordinal] = tmp_path.read_bytes()
                    finally:
                        tmp_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning("relay image %s/%s failed: %s", pid, ordinal, e)
        manifest = {
            "listing_id": pid,
            "detail_failed": bool(entry.get("detail_failed")),
            "image_urls": urls,
            "images": sorted(webp_blobs),
            "raw_search": item,
            "raw_metadata": entry.get("raw_metadata"),
        }
        blob = json.dumps(
            {k: manifest[k] for k in ("detail_failed", "image_urls", "images", "raw_search", "raw_metadata")},
            sort_keys=True, ensure_ascii=False, default=str,
        ).encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()
        lp = out / "listings" / f"{pid}.json"
        if lp.exists():
            try:
                if json.loads(lp.read_text(encoding="utf-8")).get("payload_sha256") == digest:
                    skipped += 1
                    continue
            except Exception:
                pass
        manifest["payload_sha256"] = digest
        manifest["scraped_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _atomic_write_bytes(lp, json.dumps(manifest, ensure_ascii=False, default=str).encode("utf-8"))
        for ordinal, data in webp_blobs.items():
            _atomic_write_bytes(out / "images" / pid / f"{ordinal:02d}.webp", data)
        written += 1
    logger.info("relay dump -> %s: %s written, %s unchanged", out, written, skipped)
    return written


def _relay_main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="591 relay scraper (GitHub Actions side)")
    parser.add_argument("--output-dir", default="data/incoming", help="where to write JSON + WebP")
    parser.add_argument("--fixtures", action="store_true", help="replay captured fixtures (offline test)")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--placeholder", action="store_true",
                        help="synthesize placeholder WebPs instead of fetching the CDN (offline test)")
    parser.add_argument("--selftest", action="store_true",
                        help="warm-up + single search_rent page, print result, write nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.selftest:
        client = Client591()
        region_id, section_ids = _resolve_region_sections(
            os.environ.get("X591_REGION", "台北市"), os.environ.get("X591_SECTION", ""))
        kind = _resolve_kind(os.environ.get("X591_KIND", "整層住家"))
        try:
            result = client.search_rent(region_id=region_id, section_ids=section_ids, kind=kind,
                                        price_str=os.environ.get("X591_PRICE_STR", "15000_25000"))
            data = result.get("data", {})
            print(f"selftest OK: {len(data.get('items', []))} items, "
                  f"totalRows={data.get('totalRows')}, cookies={sorted(client._session.cookies.keys())}")
            return 0
        except Exception as e:
            print(f"selftest FAILED: {type(e).__name__}: {e}")
            return 1
    dump_relay_payloads(args.output_dir, fixtures=args.fixtures, limit=args.limit,
                        placeholder=args.placeholder)
    return 0


if __name__ == "__main__":
    sys.exit(_relay_main())
