"""Unified ingestion: mcp-591 API (primary) + 591scraper DOM fallback + WebP image pipeline.

API-first with full raw payload retention (zero data loss), see docs/data-maxification.md.
Supports an offline `--fixtures` mode replaying captured 591 responses for testing.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests
from PIL import Image

from .client591 import Client591
from .constants591 import REGIONS, RENT_KINDS, SECTIONS, SECTIONS_BY_REGION

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(os.environ.get("FIXTURES_DIR", ROOT / "external" / "mcp-591" / "tests" / "fixtures"))
IMAGES_DIR = Path(os.environ.get("IMAGES_DIR", ROOT / "data" / "images"))
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": _UA})


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
        if not isinstance(next_row, int) or next_row == first_row:
            break
        first_row = next_row

    out: list[dict] = []
    for item in items:
        pid = str(item.get("id"))
        meta = None
        try:
            resp = client.get_rent_detail(pid)
            d = resp.get("data")
            meta = d if isinstance(d, dict) and d else None
        except Exception:
            meta = None
        out.append({"raw_search": item, "raw_metadata": meta})
        if limit > 0 and len(out) >= limit:
            break
    return out


def _fetch_from_fixtures(limit: int) -> list[dict]:
    with open(FIXTURES_DIR / "search_rent.json") as f:
        items = json.load(f)["data"]["items"]
    with open(FIXTURES_DIR / "rent_detail.json") as f:
        detail = json.load(f)["data"]

    fixture_id = str(items[0]["id"]) if items else None
    out = []
    for i, item in enumerate(items):
        meta = detail if fixture_id and str(item["id"]) == fixture_id else None
        out.append({"raw_search": item, "raw_metadata": meta})
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


def _info_map(data: dict) -> dict:
    return {x.get("key"): x.get("value") for x in data.get("info", [])}


def normalize_listing(raw_search: dict, raw_metadata: dict | None) -> dict:
    item = raw_search or {}
    data = raw_metadata or {}
    info = _info_map(data)
    gtm = data.get("gtm_detail_data", {})
    addr = data.get("address", {}) or {}
    cost = {x.get("key"): x.get("value") for x in (data.get("cost") or {}).get("data", [])}

    pid = str(item.get("id") or gtm.get("item_id", "").lstrip("R"))
    title = item.get("title") or data.get("title") or ""
    price = _int_price(item.get("price") or data.get("price"))
    lat = None
    lng = None
    if addr.get("lat"):
        lat, lng = float(addr["lat"]), float(addr["lng"])
    elif data.get("lat") is not None:
        lat, lng = float(data["lat"]), float(data["lng"])

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
    tags = [t.get("value") if isinstance(t, dict) else t for t in (item.get("tags") or [])]
    if not tags and isinstance(data.get("tags"), list):
        tags = [t.get("value", "") for t in data["tags"] if isinstance(t, dict)]

    return {
        "listing_id": pid,
        "title": title,
        "price": price,
        "price_unit": item.get("price_unit") or data.get("priceUnit") or "元/月",
        "url": f"https://rent.591.com.tw/{pid}",
        "status": data.get("status") if data else None,
        "is_active": bool(data),
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
        "rent_per": _int_price(item.get("price_per")) or None,
        "rent_per_unit": item.get("price_per_unit") or "元/坪/月",
        "browse_count": item.get("browse_count"),
        "refresh_time": item.get("refresh_time") or (data.get("publish") or {}).get("postTime"),
        "tags": tags,
        "contain_cost": contain,
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
            if u and u not in urls:
                urls.append(u)
    return [_strip_suffix(u) for u in urls]


def _strip_suffix(url: str) -> str:
    if "!" in url:
        return url.split("!")[0]
    return url


def download_images(listing_id: str, urls: list[str], placeholder: bool = False) -> list[dict]:
    """Download originals, re-encode to WebP quality=85, return image row dicts.

    When placeholder=True (offline fixtures testing), generate solid-color WebP
    placeholders instead of fetching from the blocked 591 CDN.
    """
    if not urls:
        return []
    out_dir = IMAGES_DIR / listing_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for ordinal, url in enumerate(urls):
        ext = ".webp"
        path = out_dir / f"{ordinal:02d}{ext}"
        try:
            if not path.exists():
                if placeholder:
                    import hashlib
                    color = int(hashlib.md5(f"{listing_id}:{ordinal}".encode()).hexdigest()[:6], 16)
                    img = Image.new("RGB", (800, 600), (color >> 16 & 255, color >> 8 & 255, color & 255))
                else:
                    resp = _SESSION.get(url, timeout=30, verify=False)
                    resp.raise_for_status()
                    img = Image.open(__import__("io").BytesIO(resp.content)).convert("RGB")
                img.save(path, "WEBP", quality=85)
            rows.append({"ordinal": ordinal, "image_url": url, "image_path": str(path)})
        except Exception as e:
            print(f"[ingestion] image {ordinal} for {listing_id} failed: {e}")
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
        print(f"[ingestion] scraper fallback {listing_id} failed: {e}")
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
    return listing
