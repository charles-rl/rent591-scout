"""Hard-filter edge cases not exercised by the hybrid pipeline payloads:
price band boundaries (inclusive) and the ENFORCE_HARD_FILTERS env bypass."""

from src import ingestion


def _listing(**over):
    base = {
        "price": 12000,
        "area": 8.0,
        "kind_name": "分租套房",
        "title": "ok listing",
        "description": "",
        "tags": [],
        "gender": "",
    }
    base.update(over)
    return base


def test_price_band_bounds_inclusive():
    assert ingestion.passes_hard_filters(_listing(price=ingestion.HARD_PRICE_MIN))[0]
    assert ingestion.passes_hard_filters(_listing(price=ingestion.HARD_PRICE_MAX))[0]


def test_price_above_max_drops():
    keep, reasons = ingestion.passes_hard_filters(_listing(price=ingestion.HARD_PRICE_MAX + 1))
    assert not keep
    assert any("outside" in r for r in reasons)


def test_price_none_or_zero_drops():
    assert not ingestion.passes_hard_filters(_listing(price=None))[0]
    assert not ingestion.passes_hard_filters(_listing(price=0))[0]


def test_enforce_hard_filters_env_bypass(monkeypatch):
    monkeypatch.setenv("ENFORCE_HARD_FILTERS", "0")
    keep, reasons = ingestion.passes_hard_filters(
        _listing(price=99999, kind_name="整層住家", gender="限女生"))
    assert keep and reasons == []


def test_filter_bypass_defaults_on(monkeypatch):
    monkeypatch.delenv("ENFORCE_HARD_FILTERS", raising=False)
    assert not ingestion.passes_hard_filters(_listing(price=99999))[0]
