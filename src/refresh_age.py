"""591 refresh_time free-text age parsing (stdlib only).

Split out of src/scoring.py so the relay-side import graph (src.ingestion)
stays importable in the requests+Pillow-only GHA runner environment.
"""

from __future__ import annotations

import re
from datetime import date

_REL_AGE_RE = re.compile(r"(\d+)\s*(分鐘|分钟|小時|小时|天|日|週|周)\s*(?:前|內|内)")
_ABS_AGE_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_JUST_NOW_RE = re.compile(r"剛剛|刚刚")
# '昨日更新' / '昨天更新' — the search-list spelling carries no digit.
_YESTERDAY_RE = re.compile(r"昨[天日]")
_REL_AGE_DAYS = {"分鐘": 1 / 1440, "分钟": 1 / 1440, "小時": 1 / 24, "小时": 1 / 24,
                 "天": 1.0, "日": 1.0, "週": 7.0, "周": 7.0}
_MAX_AGE_DAYS = 370
# A date this many days ahead of today is clock/timezone skew, not last year's
# listing; anything further ahead is read as the previous year's occurrence.
_MAX_FUTURE_SKEW_DAYS = 2


def parse_days_on_market(refresh_time, today: date | None = None) -> float | None:
    """591 refresh_time free text -> listing age in days; None when unparseable.

    Handles the shapes actually observed in production and fixtures:
      '此房屋在1天前發佈' / '20分鐘內更新' / '4小時內更新'  -> relative
      '昨日更新'                                            -> 1.0
      '此房屋在剛剛發佈'                                    -> 0.0 (brand new, NOT unknown)
      '此房屋在9月7日發佈'                                 -> absolute, year inferred

    Absolute dates carry no year: a date later than today is last year's occurrence
    (e.g. '12月31日' seen on Jan 2 -> 2.0). A date only a day or two ahead of today
    cannot be either year's listing -- it is clock/timezone skew -- and returns
    None instead. Both rejections matter because the stored value feeds
    `tabular_vector`'s `max(0.0, dom)` clamp: a negative age would collapse onto
    log1p(0) == 0.0 and read as "brand new" (the exact conflation the -1.0 sentinel
    exists to prevent), while a skew date read as last year would report a
    ~365-day-old listing for a fresh one.

    The result is snapshotted into listings.days_on_market at ingest time --
    do NOT re-derive it at scoring time, or predicted_score stops being a
    function of stored data (the relative forms drift with the wall clock).
    """
    text = str(refresh_time or "")
    if _JUST_NOW_RE.search(text):
        return 0.0  # genuinely brand new, NOT unknown
    m = _REL_AGE_RE.search(text)
    if m:
        return float(int(m.group(1)) * _REL_AGE_DAYS[m.group(2)])
    if _YESTERDAY_RE.search(text):
        return 1.0
    m = _ABS_AGE_RE.search(text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        today = today or date.today()  # noqa: DTZ011 - local wall-clock date vs 591's local listing age
        for year in (today.year, today.year - 1):
            try:
                posted = date(year, month, day)
            except ValueError:
                continue
            age = (today - posted).days
            # Negative ages are never stored: tabular_vector's max(0.0, dom) clamp
            # would collapse them onto log1p(0) == 0.0, i.e. "brand new". A date a
            # day or two ahead is clock skew, not a listing from the future or from
            # last year -- return None so it becomes the -1.0 "unknown" sentinel.
            # Anything further ahead falls through to the previous year's occurrence
            # (a real positive age). (That fallback also makes _MAX_AGE_DAYS
            # effectively unreachable: the worst case is 365.)
            if 0 <= age <= _MAX_AGE_DAYS:
                return float(age)
            if year == today.year and -_MAX_FUTURE_SKEW_DAYS <= age < 0:
                return None
        return None
    return None
