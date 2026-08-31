#!/usr/bin/env python3
"""CLI feedback tool: rate a 591 listing and (optionally) consolidate dynamic preferences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import database, dynamic_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rate a 591 Apartment")
    parser.add_argument("--id", required=True, help="591 Listing ID")
    parser.add_argument("--score", type=float, required=True, help="Overall Score 1-5")
    parser.add_argument("--bathroom", type=float, default=3.0, help="Bathroom Score 1-5")
    parser.add_argument("--comment", type=str, default="", help="User + Partner feedback text")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite DB")
    args = parser.parse_args()

    conn = database.connect(args.db)
    ok = database.rate_listing(conn, args.id, args.score, args.bathroom, args.comment)
    if not ok:
        print(f"Listing {args.id} not found in DB.")
        conn.close()
        sys.exit(1)

    print(f"Rating saved for listing {args.id}.")

    if args.comment:
        bullets = dynamic_prompt.update_preferences(conn, args.comment)
        print(f"Dynamic Preference Prompt updated:\n{bullets}")
    conn.close()


if __name__ == "__main__":
    main()
