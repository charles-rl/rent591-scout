#!/usr/bin/env python3
"""Terminal audit of all AI system prompts, thresholds, and tunables vs docs/591research.md."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main
from src import ingestion, scoring, vision_llm

CONCERNS = {"window-less (無窗)": ("無窗", "採光", "exterior window"),
            "private bath (獨立衛浴)": ("獨立衛浴", "衛浴獨立"),
            "rooftop add (頂樓加蓋)": ("頂樓加蓋",),
            "cooking (開伙)": ("開伙",)}

MODEL_PATH = ROOT / "models" / "xgboost_head.json"


def aware(prompt: str, needles: tuple[str, ...]) -> bool:
    return any(n in prompt for n in needles)


def main_() -> int:
    w = 78
    print("=" * w)
    print("AI SYSTEM PROMPT & TUNABLE-FEATURE AUDIT — rent591-scout")
    print("=" * w)

    prompts = {
        "vision_llm.BASE_SYSTEM_PROMPT": vision_llm.BASE_SYSTEM_PROMPT,
        "vision_llm.DEFAULT_BULLETS": vision_llm.DEFAULT_BULLETS,
        "vision_llm._RETRY_NOTE": vision_llm._RETRY_NOTE,
        "vision_llm.consolidate_preferences (NLP summary)":
            "You summarize rental preferences concisely. | Task: Update the bulleted list "
            "of user preferences (max 7 items). Return ONLY the bulleted list.",
    }
    print("\n[1] SYSTEM PROMPT AWARENESS MATRIX")
    labels = ["無窗", "衛浴", "頂加", "開伙"]
    print(f"{'prompt':<52}" + "".join(f"{l:>6}" for l in labels))
    for name, text in prompts.items():
        row = f"{name:<52}" + "".join(f"{('YES' if aware(text, n) else 'no'):>6}" for n in CONCERNS.values())
        print(row)

    print("\n[2] CURRENT PROMPT STRINGS (verbatim)")
    for name in ("vision_llm.BASE_SYSTEM_PROMPT", "vision_llm.DEFAULT_BULLETS"):
        print(f"\n----- {name} -----\n{prompts[name].rstrip()}")

    try:
        conn = sqlite3.connect(f"file:{ROOT/'data'/'apartments.db'}?mode=ro", uri=True)
        row = conn.execute("SELECT prompt_bullet_list FROM dynamic_preferences ORDER BY id DESC LIMIT 1").fetchone()
        rated = conn.execute("SELECT COUNT(*) FROM listings WHERE user_rated=1").fetchone()[0]
        listings_left = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        conn.close()
        dyn = row[0] if row else "(empty -> DEFAULT_BULLETS fallback)"
    except sqlite3.Error as e:
        dyn, rated, listings_left = f"(db unreadable: {e})", "?", "?"
    print(f"\n[3] DYNAMIC PREFERENCES (live bullets) rated={rated} listings={listings_left}\n{dyn}")

    print("\n[4] THRESHOLDS / USER-TUNABLE PARAMETERS")
    rows = [
        ("SCORE_THRESHOLD", main.SCORE_THRESHOLD, "env", "notify cutoff (1-5 scale)"),
        ("DEDUP_THRESHOLD", main.DEDUP_THRESHOLD, "env", "DINOv3 group-cosine duplicate cutoff"),
        ("RATED_THRESHOLD", scoring.RATED_THRESHOLD, "env", "XGBoost takeover point"),
        ("VLM_ATTEMPTS", vision_llm.VLM_ATTEMPTS, "env", "JSON-retry attempts per VLM call"),
        ("VLM_MAX_IMAGES", vision_llm.VLM_MAX_IMAGES, "env", "photos per VLM call"),
        ("VLM_IMAGE_MAX_SIDE", vision_llm.VLM_IMAGE_MAX_SIDE, "env", "downscale cap (px)"),
        ("HARD_PRICE_MIN/MAX", f"{ingestion.HARD_PRICE_MIN}/{ingestion.HARD_PRICE_MAX}", "env", "hard rent window NTD"),
        ("HARD_MIN_AREA_PING", ingestion.HARD_MIN_AREA_PING, "env", "hard min usable area (坪)"),
        ("ENFORCE_HARD_FILTERS", "1 (on)", "env", "set 0 to disable Stage-1 drops"),
        ("PENALTY: HIGH_ELEC_FEE", scoring.PENALTY_POINTS["HIGH_ELEC_FEE"], "scoring.py", "rate > 5 NTD/kWh"),
        ("PENALTY: NO_PETS", scoring.PENALTY_POINTS["NO_PETS"], "scoring.py", "禁寵 keywords"),
        ("PENALTY: HIGH_WALKUP", scoring.PENALTY_POINTS["HIGH_WALKUP"], "scoring.py", "floor>=5, no elevator"),
        ("PENALTY: ILLEGAL_ROOFTOP", scoring.PENALTY_POINTS["ILLEGAL_ROOFTOP"], "scoring.py", "頂樓加蓋 (adjusted -10)"),
        ("PENALTY: MANUAL_TRASH", scoring.PENALTY_POINTS["MANUAL_TRASH"], "scoring.py", "追垃圾車"),
        ("PENALTY: SHARED_WASHER", scoring.PENALTY_POINTS["SHARED_WASHER"], "scoring.py", "投幣/共享洗衣"),
    ]
    for name, val, src, desc in rows:
        print(f"  {name:<26} {val!s:<14} ({src})  {desc}")

    print("\n[5] XGBOOST HEAD STATE")
    if MODEL_PATH.exists():
        head = json.loads(MODEL_PATH.read_text())
        trees = head.get("learner", {}).get("gradient_booster", {}).get("model", {}).get("trees", [])
        degenerate = all(len(t.get("children", [{}])) <= 1 for t in trees)
        nfeat = trees[0].get("num_feature", "?") if trees else "?"
        print(f"  model={MODEL_PATH.name} trees={len(trees)} features={nfeat} "
              f"degenerate={'YES (constant base_score; needs retrain after >20 ratings)' if degenerate else 'no'}")
        print("  note: feature vector is fixed at 768 DINO dims + 5 flags + 1 warning bit = 774.")
    else:
        print("  models/xgboost_head.json missing -> trains from rated rows on first use")

    print("\n[6] RECOMMENDATIONS")
    for line in [
        "1. BASE_SYSTEM_PROMPT now covers 無窗/獨立衛浴/頂樓加蓋/開伙 with an explicit",
        "   uncertainty policy (warn, don't guess). Keep new red-flag rules in",
        "   dynamic_preferences via `rate.py --comment` instead of editing code.",
        "2. If '衛浴獨立性未確認' noise is high on kind=3 results, add a bullet:",
        "   '- Only notify if bathroom is confirmed 獨立衛浴 from photos.'",
        "3. XGBoost head is degenerate until >RATED_THRESHOLD rated rows accumulate;",
        "   after first retrain (rate.py auto-triggers), consider raising",
        "   SCORE_THRESHOLD to 4.0 so rooftop/window warnings must survive scoring.",
        "4. DEDUP_THRESHOLD 0.95 is conservative; if re-photos of the same unit slip",
        "   through, lower to ~0.92. Vision confidence is boolean-only (no per-flag",
        "   confidence value exists in the schema) — a `bathroom_confidence` float",
        "   would be the highest-value schema addition.",
        "5. Penalty engine fires only on positive evidence (soft policy); tune",
        "   PENALTY_POINTS in src/scoring.py, or widen _PET/_ROOFTOP pattern tuples.",
    ]:
        print("  " + line)
    print("=" * w)
    return 0


if __name__ == "__main__":
    sys.exit(main_())
