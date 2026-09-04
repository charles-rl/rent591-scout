# Feedback Triage: Code vs. Dynamic Prompt

User/partner rating comments come in free text. Not every comment belongs in the
vision system prompt: rules that can be verified **deterministically** from listing
data should become code, so they fire 100% of the time, cost no tokens, and can be
unit-tested. Only subjective/visual taste should live as dynamic preference bullets.

Triage happens interactively in-chat (opencode reads the review, proposes routing,
the user confirms). `rate.py` itself stays a dumb recorder — it never edits code.

## Routing rules

| Comment character | Destination | Where |
|---|---|---|
| Verifiable from structured fields (`facilities`, `contain_cost`, `price`, `floor`, `shape`) or stable text patterns in description/tags | Deterministic penalty | `src/scoring.py` `heuristic_penalties()` + `PENALTY_POINTS`/`PENALTY_MESSAGES` |
| Disqualifying constraint the user states outright (e.g. "only females allowed ... as a constraint") | Deterministic hard filter (drop, never alerted) | `src/ingestion.py` `passes_hard_filters()` (enforced on relay + incoming sides) |
| Structured metadata the pipeline currently drops (e.g. 591 `houseInfo` pet entry) | Ingestion fix first, then penalty | `src/ingestion.py` `normalize_listing()` -> `facilities` |
| Needs judgment over images/unstructured nuance (bathroom quality, light, "vibe", clutter) | Dynamic prompt bullet | `dynamic_prompt.update_preferences()` -> Qwen system prompt |
| Numeric thresholds the user quotes (e.g. ">15600") | Module constant / env var next to the rule, never hardcoded in prose | `src/scoring.py` |

Decision aid: if you could write a unit test asserting the rule fires, it is code.
If two reasonable people could disagree about it, it is a prompt bullet.

## Worked example (reviews 21799503 / 21784695 / 21935557)

- "No pets allowed warning" (twice, partner + me) — code, twice over:
  1. `normalize_listing()` now maps the structured 591 pet entry
     (`houseInfo.data[key=pet]` / `service.notice[key=pet]`) into `facilities`;
     `heuristic_penalties()` includes `facilities` in its evidence blob.
  2. Pet wording patterns extended (`不可養寵`, `不開放寵`) — the original
     `不可寵/禁寵/嚴禁寵物` list did not match the common phrasing "不可養寵物".
- "Rent on the high end >15600 and electricity isn't included -> warn" — code:
  `ELEC_EXTRA_HIGH_COST` fires when `price > ELEC_EXTRA_COST_RENT` (15600, env-overridable)
  and there is no positive evidence electricity is included (`contain_cost` entry
  containing 電費, or 含電費/電費含/含水電/免電費 wording). Warning-only feedback was
  rejected in favour of a −10 penalty; the flag also enters the XGBoost fusion vector.
- Reversed electricity wording "電費1度6元" (listing 21784695) — code:
  `_ELEC_REVERSED_RE` catches the "N度M元" order that `HIGH_ELEC_FEE` previously missed.
- "5/5 perfect, no comments" (21935557) — nothing to route; pure training signal.
- `NO_PETS` and `ELEC_EXTRA_HIGH_COST` are now `FEATURE_NAMES` inputs (vector 14 -> 16).
  `_load_trained_model()` detects the width change and retrains the saved head
  automatically once enough ratings accumulate.

## Worked example (reviews 21927820 / 21918067 — female-only)

- "Only females allowed, should be included as a constraint" (twice, me + partner) —
  **hard filter**, not a penalty: `normalize_listing()` now extracts the structured
  591 gender entry (`houseInfo.data[key=sex]` / `service.notice[key=sex_2]`) into
  `listing["gender"]`, and `passes_hard_filters()` drops any value matching
  `GENDER_RESTRICTION_RE` (`限[男女]`). Both filter call sites (relay dump, incoming
  ingest) share these functions, so relay/incoming stay in sync automatically.
  No scoring flag and no `FEATURE_NAMES` change — excluded listings never reach scoring.
  Evidence is structured fields only: "鄰居都女性" titles and permissive values
  ("不拘", "男女皆可") never drop.
- 21938817 "Little detail about the shared apartment" — prompt bullet, added
  manually ("Flag listings that provide little detail about shared/common areas")
  because Qwen consolidation silently kept the list unchanged; the consolidation
  auto-added "female-only ... constraint" bullet was removed as a duplicate of the
  code rule above.
- 21864342 "Shared apartment" (3.0/2.5) — nothing routed: user chose to keep current
  filters, treated as pure training signal. Consolidation had inverted the sentiment
  ("Prefer shared apartments", "Prefer no-pets-only listings" from a rating that was
  fine *despite* NO_PETS) — both removed manually. Lesson: always audit the bullets
  after consolidating terse negative feedback.

## Rating intake + availability ranking (`scripts/rank_unrated.py`)

The standing post-rating workflow: record the ratings, then ask which listings
are still worth viewing. One command does the whole loop (the agent can just run it):

1. Record each rating with `rate.py` (bathroom score too — feeds the bath probe):
   `.venv/bin/python rate.py --id <591ID> --score N [--bathroom M] [--comment "..."]`.
   Past RATED_THRESHOLD=20, every rating retrains the XGBoost head automatically.
   Comments (if any) get triaged here per this doc before/while recording.
2. Run `.venv/bin/python scripts/rank_unrated.py`:
   - probes every **active unrated** listing URL through the PC devtunnel proxy
     and marks positive delisting evidence (404/410 or 物件已下架 markers) as
     `is_active=0` (fail-open on transport errors — an unreachable probe is not a
     delist, matching `health_check` semantics). Skipped automatically when the
     proxy probe fails; `--no-probe` forces skip.
   - re-scores all unrated listings with the current head (`score_all_unrated`)
     and prints **TOP / MIDDLE / BOTTOM N** (default 5, `--n K`) of what's available,
     each with location/price/ping/floor/kind + warning flags; `[DUP]` marks DINOv3
     duplicates. `--retrain` forces a head retrain first (only needed when the
     width check didn't already fire).

Reply to the user with the three tables as-is. MIDDLE block is the median ±N/2 —
useful for "what's average" when the top of the list is too good to be true and
the bottom is mostly penalty-driven rejects (頂層加蓋 / NO_PETS / HIGH_WALKUP).

## Adding a new deterministic rule (checklist)

1. `scoring.py`: flag name (UPPER_SNAKE), `PENALTY_POINTS`, `PENALTY_MESSAGES`
   (English message — ntfy headers are latin-1), evidence check in `heuristic_penalties()`.
2. If evidence lives in metadata the listing dict lacks: extend
   `ingestion.normalize_listing()` and, for retrained rows, `_row_listing()` +
   the SELECTs in `database.get_rated_samples()/get_scoring_rows()`.
3. Add the flag to `FEATURE_NAMES` + `tabular_vector()` if the user's ratings should
   train against it (width change auto-retrains the saved head via the width check).
4. Unit test in `tests/test_scoring.py` (positive + both negative branches);
   update the `test_fusion_vector_layout` width assertion.
5. Chinese legacy wording variants shown to users: add to `notifier._WARN_EN`.
