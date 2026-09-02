#!/usr/bin/env python3
"""One-shot purge of data/apartments.db against docs/591research.md hard constraints.

Deletes listings (+ their listing_images / relay_state rows) violating ANY of:
  * price NULL or outside [10000, 17000] NTD
  * parsed area < 6.0 ping (NULL area kept: unknown != violating)
  * kind_name not in (獨立套房, 分租套房) — incl. 整層住家 / 雅房 / NULL
  * section NULL or outside the 10 target districts
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "apartments.db"

ACCEPTED_KINDS = ("獨立套房", "分租套房")
TARGET_SECTIONS = ("汐止區", "三重區", "南港區", "內湖區", "北投區",
                   "大同區", "士林區", "蘆洲區", "淡水區", "板橋區")


def main() -> int:
    if not DB.exists():
        print(f"database not found: {DB}")
        return 1
    conn = sqlite3.connect(str(DB))
    try:
        rows = conn.execute("SELECT listing_id, price, area, kind_name, section FROM listings").fetchall()
        doomed: dict[str, list[str]] = {}
        counts = {"price": 0, "area": 0, "kind": 0, "section": 0}
        for lid, price, area, kind_name, section in rows:
            reasons = []
            if not price or price < 10000 or price > 17000:
                reasons.append("price")
            if area is not None and area < 6.0:
                reasons.append("area")
            if kind_name not in ACCEPTED_KINDS:
                reasons.append("kind")
            if section not in TARGET_SECTIONS:
                reasons.append("section")
            if reasons:
                doomed[lid] = reasons
                for r in reasons:
                    counts[r] += 1
        ids = list(doomed)
        print(f"=== 591 hard-constraint purge ({DB}) ===")
        print(f"total listings before : {len(rows)}")
        for reason, n in counts.items():
            print(f"  violating {reason:<8}: {n}")
        if ids:
            q = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM listing_images WHERE listing_id IN ({q})", ids)
            conn.execute(f"DELETE FROM relay_state WHERE listing_id IN ({q})", ids)
            conn.execute(f"DELETE FROM listings WHERE listing_id IN ({q})", ids)
            conn.commit()
        remaining = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        print(f"purged                : {len(ids)}")
        print(f"compliant remaining   : {remaining}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
