# 591scraper Analysis (`ceshine/591scraper`)

Vendored at `external/591scraper/` (MIT for repo code; **DrissionPage dependency is non-commercial licensed**).
Scripts: `collect_list.py` (list → IDs), `fetch_info.py` (ID → detail), `utils/post_processing.py`.

## Runtime
- DrissionPage via Chrome DevTools Protocol (no chromedriver), controls Chrome/Chromium directly; avoids `navigator.webdriver` detection.
- Requires system Chrome or Chromium.
- `create_browser(headless=False)`: `ChromiumPage(ChromiumOptions)` with:
  - `headless()` when requested
  - `--disable-blink-features=AutomationControlled`
  - `--no-first-run`, `--no-default-browser-check`
- Persistent user profile via `ChromiumOptions().set_paths(user_data_path="./browser_profile")` (persistent cookies/session; our ingestion uses this).

## `collect_list.py` (list pagination → IDs)
- Reads search URL from env `X591URL` (must contain `region` query param).
- Wait: `page.wait.eles_loaded("css:.item-info-title a", timeout=10)`; sleeps `random()*2+1`.
- Extract IDs: for each `a.item-info-title a`, regex `/(\d+)(?:\.html)?$` on href.
- Pagination: click element `text=下一頁`; stop on `#`/empty href or no match. `--max-pages N`.
- Output: `joblib.dump(list(listings), output)` → `cache/listings.jbl`.

## `fetch_info.py` (per-listing detail)
- Navigates `https://rent.591.com.tw/home/{listing_id}`; wait `css:div.title`; sleep `random()*3+1`.
- `get_listing_info` returns dict with these DOM-derived fields:

| Field | Selector / extraction |
|---|---|
| `id` | input arg |
| `title` | `css:.title h1` |
| `addr` | `css:div.address div` |
| `社區` (community) | `css:div.address p a` |
| `price` | `css:div.house-price` → `parse_price` (regex `^([\d,]+)`, strips commas; `0` if `--`/`無`/empty) |
| `desc` | `css:div.house-condition-content` (屋況介紹; full description text) |
| `poster` | `css:p.base-info-pc` (whitespace-collapsed; e.g. `屋主: 吳太太`, `仲介: … (收服務費)`) |
| `養寵物` | `css:section.service` → `"No"` if text contains `不可養寵物` else `"Yes"`; `None` if missing |
| `租金含` / `車位費` / `管理費` | element `text={label}` → parent → `css:div.text`; `""` if missing |
| `提供設備` (facilities) | `css:div.service-facility` → `css:dl:not(.del) dd` joined with `, ` |

- `get_page` retries (2 attempts) on `PageLoadError`; raises `NotExistException` when title contains `不存在` (delisted).
- Delisted entries skipped; per-listing sleep `random()*5`.

## Post-processing (`utils/post_processing.py`)
- `parse_price(s) -> int`
- `auto_marking_(df)`: sets `mark="x"` when title/desc contains `社宅`/`社會住宅` or facilities contain `機械車位`.
- `adjust_price_(df)`: `price_adjusted = price*(1 + 1/24 if '收取服務費' in poster else 1) + 管理費(parsed) + (2500 if 車位費 contains '費用另計')`.

## Output CSV columns
`mark, title, price, price_adjusted, link, addr, 社區, 車位費, 管理費, poster, 養寵物, 提供設備, id, fetched, desc`
(`格局/坪數/樓層/型態` are commented out — no longer scrapable on the current 591 DOM).

## Gaps & risks
- **No image collection** in either script — images come exclusively from the mcp-591 API `photoList`.
- `格局/坪數/樓層/型態` no longer scrapable (DOM changed); covered by API instead.
- `desc` (full text) may exceed what API `remark.content` provides → good fallback/complement.
- DrissionPage **non-commercial license** — personal use only.
- Slow (per-item browser navigation, sleeps) → fallback tier only.
