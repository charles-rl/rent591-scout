# 591 Scout Repository Filter & Configuration Specification

> Synced with codebase (`src/client591.py`, `src/constants591.py`, `src/ingestion.py`).
> Region/section IDs follow `src/constants591.py` (verified against live BFF payloads,
> e.g. 北投區 listings carry `sectionid:"9"`). Hard vs. soft constraint split follows
> the uncertainty policy: only deterministic metadata is dropped automatically;
> ambiguous qualities are flagged and left to the human reviewer.

## 1. Target Location Matrix & API Parameter Mappings

Below are the 10 target residential zones following the revised multi-node commute priority rankings.

| Final Rank | Strategic Residential Zone | 591 Region Name (EN / ZH) | 591 Section Name (EN / ZH) | Region Code (`regionid`) | Section Code (`sectionid`) |
| --- | --- | --- | --- | --- | --- |
| **1** | **Xizhi** | New Taipei City / 新北市 | Xizhi District / 汐止區 | `3` | `27` |
| **2** | **Sanchong** | New Taipei City / 新北市 | Sanchong District / 三重區 | `3` | `43` |
| **3** | **Nangang** | Taipei City / 臺北市 | Nangang District / 南港區 | `1` | `11` |
| **4** | **Neihu** | Taipei City / 臺北市 | Neihu District / 內湖區 | `1` | `10` |
| **5** | **Beitou** | Taipei City / 臺北市 | Beitou District / 北投區 | `1` | `9` |
| **6** | **Datong** | Taipei City / 臺北市 | Datong District / 大同區 | `1` | `2` |
| **7** | **Shilin** | Taipei City / 臺北市 | Shilin District / 士林區 | `1` | `8` |
| **8** | **Luzhou** | New Taipei City / 新北市 | Luzhou District / 蘆洲區 | `3` | `47` |
| **9** | **Tamsui** | New Taipei City / 新北市 | Tamsui District / 淡水區 | `3` | `50` |
| **10** | **Banqiao** | New Taipei City / 新北市 | Banqiao District / 板橋區 | `3` | `26` |

> Correction vs. earlier draft: Xizhi is `27` (not `46` = 林口區), Beitou is `9`
> (not `19` = 七堵區/基隆市), Luzhou is `47` (not `45` = 泰山區), Tamsui is `50`
> (not `58` = 關西鎮/新竹縣).

---

## 2. API Search Request Parameters

Endpoint: `GET https://bff-house.591.com.tw/v3/web/rent/list` (mobile BFF, see `src/client591.py`).

* **Region / Section (`regionid` / `sectionid`):** numeric IDs from `src/constants591.py`;
  `sectionid` is a comma-joined list (e.g. `27,43`).
* **Rent Price (`rentprice`):** `10000_17000` — **underscore**-separated TWD/month range
  (the old `10000$17000` syntax is invalid for this endpoint).
* **Rental Kind (`kind`):** single integer per request — `2` = 獨立套房 (independent studio),
  `3` = 分租套房 (sublet suite, private bathroom expected). The BFF API does **not** accept
  comma lists, so the scraper runs **two passes** (kind=2, then kind=3). `4` = 雅房
  (shared-bathroom room) is never queried and is hard-dropped if ingested.
* **Session handling:** no `urlJumpIp` cookie is required on the BFF endpoint.
  `Client591` sets a `T591_TOKEN` device-id cookie and primes WAF cookies via a warm-up
  GET against `https://m.591.com.tw/` (with 403 re-warm/backoff).
* Env knobs (`src/ingestion.py`): `X591_QUERIES` (multi-region spec
  `新北市:汐止區,三重區;台北市:南港區`), `X591_KINDS` (`獨立套房,分租套房`),
  `X591_PRICE_STR` (`10000_17000`), `ENFORCE_HARD_FILTERS`.

---

## 3. Stage 1: Deterministic Metadata Hard Filters

Applied at fetch/ingestion time in `ingestion.passes_hard_filters()` (both the relay
dumper and `main.py`). **Hard = certain, drop silently. Soft = uncertain, keep + flag
for human review.**

### Hard (certain — listing dropped before storage)

1. **Price Range:** Monthly rent must be $\ge 10{,}000\text{ NTD}$ and $\le 17{,}000\text{ NTD}$.
2. **Floor Space Area:** Usable area $\ge 6.0\text{ Ping}$ (坪) **when parsed**; missing/unknown area is not a drop (soft).
3. **Room Type:** Only `kind ∈ {2 獨立套房, 3 分租套房}`. Units with shared bathrooms
   (`kind == 4` 雅房) and any other kind are strictly dropped; unknown kind is soft-kept.
4. **Cooking Prohibition:** Explicit `嚴禁開伙` / `不可煮食` (and variants) in description/tags
   is a drop. Silence is *not* a drop (absence of prohibition passes).

### Soft (uncertain — kept, flagged to the user via warnings)

5. **Bathroom Verification (kind == 3):** text/vision should confirm 獨立衛浴 (in-unit private
   bathroom). If unverifiable, keep the listing with warning `衛浴獨立性未確認` — the user decides.
6. **Exterior Window / Natural Light:** validated via Stage 2 vision
   (`has_exterior_window`); window-less rooms get a warning, not an automatic drop.

---

## 4. Stage 3: NLP Heuristic Scoring & Warning Flag Engine

Implemented in `scoring.compute_heuristic_score()`; stored in `listings.heuristic_score`
and merged into `qwen_warnings`. Penalties fire only on **positive textual evidence** —
no evidence means no penalty and no false alarm (uncertainty stays with the user).

Scoring begins at a baseline of **100 points**.

| Feature / Condition | Trigger Criterion | Pipeline Warning Flag | Deduction Points |
| --- | --- | --- | --- |
| **Electricity Rate** | Utility rate $> 5.0\text{ NTD/kWh}$ parsed from text | `HIGH_ELEC_FEE` | `-15` |
| **Pet Allowance** | Positive ban wording (`不可寵`/`禁寵`/`嚴禁寵物` + the 2026-09 additions `不可養寵`/`不開放寵`) or a structured no-pet entry in `facilities` | `NO_PETS` | `-10` |
| **Walk-Up Floor** | No elevator (shape/facilities) AND floor $\ge 5$ | `HIGH_WALKUP` | `-25` |
| **Rooftop Structure** | Text indicates `"頂樓加蓋"` / `"頂加"` | `ILLEGAL_ROOFTOP` | `-10` *(Adjusted)* |
| **Garbage Management** | Text indicates `"追垃圾車"` | `MANUAL_TRASH` | `-10` |
| **Laundry Setup** | Shared/coin laundry mentioned (`投幣`/`共享洗衣`) | `SHARED_WASHER` | `-5` |
| **High-Cost Rent + Separate Elec.** | Price > 15,600 NTD/mo (env-overridable) with no evidence electricity is included | `ELEC_EXTRA_HIGH_COST` | `-10` |

$$\text{Final Composite Score} = 100 - \sum \text{Penalty Points}$$

---

## 5. Scraper Polling Tier Lists

Implemented as three cron stanzas in `.github/workflows/scrape_relay.yml`, each exporting
an `X591_QUERIES` spec consumed by `src/ingestion.py` (both kind passes per tier):

### Primary Polling Tier (Frequency: Every 15 Minutes — `*/15 * * * *`)
* **Xizhi District** (`region=3`, `section=27`)
* **Sanchong District** (`region=3`, `section=43`)
* **Nangang District** (`region=1`, `section=11`) *(Elevated to Tier 1 for direct Academic Shuttle access)*

### Secondary Polling Tier (Frequency: Every 1 Hour — `0 * * * *`)
* **Neihu District** (`region=1`, `section=10`)
* **Beitou District** (`region=1`, `section=9`)
* **Datong District** (`region=1`, `section=2`)
* **Shilin District** (`region=1`, `section=8`)
* **Luzhou District** (`region=3`, `section=47`)

### Tertiary Polling Tier (Frequency: Every 3 Hours — `0 */3 * * *`)
* **Tamsui District** (`region=3`, `section=50`)
* **Banqiao District** (`region=3`, `section=26`)
