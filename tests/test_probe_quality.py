"""Regression tests that the Ridge probes carry actual signal (not a collapsed constant).

Both Layer-1 probes fit 768-d unit-norm vectors, whose design matrix X^T X has
trace ~n. An alpha that dominates that (the old defaults: 10.0 for dino, 100.0 for
bathroom) collapses the solve onto the bias term, so the probe returns ~the training
mean for every input. A collapsed dino_visual_score/bath_model_score still produces
valid-looking predictions, so shape/plumbing tests pass while the ranking is noise.

These tests therefore assert DISCRIMINATION (better than a constant predictor) on
non-separable data, against the SHIPPED defaults -- no alpha monkeypatching. The
existing orthogonal-`_axis` fixtures cannot catch this: perfectly-separated centroids
are recoverable even by a near-zero-weight probe.
"""

import numpy as np
import pytest

from src import bathroom_probe, database
from src import visual_preference as vp


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _overlapping(rng, positive: bool, dim: int, noise: float = 0.4) -> np.ndarray:
    """Unit vectors around two heavily OVERLAPPING centroids.

    6 shared dims give both classes common mass, then 16 class-specific dims drawn
    from DISJOINT index ranges (positives 60-119, negatives 120-179) carry the
    label. Calibrated so the problem is learnable but NOT trivially separable: a
    correctly-regularized probe beats the 0.50 constant baseline on LOO while the
    old collapsed alphas (10 / 100) do not. That gap is what makes the test a real
    guard rather than a tautology (exact numbers depend on seed/noise; the tests
    below assert the properties, not fixed MAEs).
    """
    base = np.zeros(dim, dtype=np.float32)
    base[rng.integers(0, 60, size=6)] = 1.0            # shared by both classes
    signal = rng.integers(60, 120, size=16) if positive else rng.integers(120, 180, size=16)
    base[signal] = 1.0
    return _unit(base + rng.standard_normal(dim).astype(np.float32) * noise)


def _loo_mae(fit, predict, X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out MAE; the held-out sample never contributes to its own fit."""
    errs = []
    for i in range(X.shape[0]):
        keep = np.ones(X.shape[0], dtype=bool)
        keep[i] = False
        w, b = fit(X[keep], y[keep])
        errs.append(abs(predict(X[i], w, b) - float(y[i])))
    return float(np.mean(errs))


def _constant_mae(y: np.ndarray) -> float:
    return float(np.mean(np.abs(y - y.mean())))


# --------------------------------------------------------------------------- dino probe


def _dino_dataset(n: int = 26, seed: int = 7):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        good = i % 2 == 0
        X.append(_overlapping(rng, good, vp.EMBED_DIM))
        # user_score 1-5 -> (score-1)/4
        y.append(1.0 if good else 0.0)
    return np.stack(X), np.asarray(y, dtype=np.float32)


def test_dino_probe_alpha_is_in_the_usable_band():
    """768-d unit-norm inputs make X^T X tiny; alpha must stay small.

    Historical defaults: PROBE_ALPHA=10.0 (collapsed), ALPHA_HEAVY=100.0 (collapsed).
    """
    assert 0.001 <= vp.PROBE_ALPHA <= 0.1, (
        f"PROBE_ALPHA={vp.PROBE_ALPHA} collapses the probe; keep it in [0.001, 0.1]")


def test_dino_probe_beats_constant_baseline_loo():
    X, y = _dino_dataset()
    loo = _loo_mae(vp.train_probe, lambda v, w, b: float(np.clip(v @ w + b, 0.0, 1.0)), X, y)
    base = _constant_mae(y)
    assert loo < base, f"dino probe LOO-MAE {loo:.4f} >= constant baseline {base:.4f}"


def test_dino_probe_output_spread_is_not_collapsed():
    X, y = _dino_dataset()
    w, b = vp.train_probe(X, y)
    preds = np.clip(X @ w + b, 0.0, 1.0)
    assert float(preds.std()) > 0.05, (
        f"dino probe output std {preds.std():.4f} is effectively constant")


def test_dino_probe_loo_degrades_at_the_old_collapsed_alpha(monkeypatch):
    """Negative control for PROBE_ALPHA: pin the collapse instead of trusting a margin.

    Both sides use LITERAL alphas (0.01 vs 10.0), not config: a control that reads
    the "good" side from the shipped DINO_PROBE_ALPHA is not self-contained -- it
    only fails when the shipped value is wrong, not because alpha=10 is bad. The
    shipped value itself is pinned by test_dino_probe_alpha_is_in_the_usable_band.
    Measured on this fixture (n=26): alpha=0.01 -> std 0.496 / LOO-MAE 0.396;
    alpha=10 -> std 0.058 / LOO-MAE 0.506 (worse than the 0.500 constant baseline).
    The control asserts the LOO ordering and a RELATIVE std collapse: the absolute
    0.05 spread gate is a data-calibrated hairline (0.058 vs 0.05) and is pinned
    against the shipped alpha elsewhere.
    """
    X, y = _dino_dataset()
    base = _constant_mae(y)
    predict01 = lambda v, w, b: float(np.clip(v @ w + b, 0.0, 1.0))
    with monkeypatch.context() as m:
        m.setattr(vp, "PROBE_ALPHA", 0.01)
        good_loo = _loo_mae(vp.train_probe, predict01, X, y)
        w, b = vp.train_probe(X, y)
        good_std = float(np.clip(X @ w + b, 0.0, 1.0).std())

        m.setattr(vp, "PROBE_ALPHA", 10.0)
        bad_loo = _loo_mae(vp.train_probe, predict01, X, y)
        w, b = vp.train_probe(X, y)
        bad_std = float(np.clip(X @ w + b, 0.0, 1.0).std())

    assert good_std > 0.05, f"alpha=0.01 output must carry real spread, got {good_std:.4f}"
    assert bad_std < 0.25 * good_std, (
        f"alpha=10 (std {bad_std:.4f}) must collapse relative to alpha=0.01 (std {good_std:.4f})")
    assert good_loo < base, f"alpha=0.01 LOO-MAE {good_loo:.4f} must beat the constant {base:.4f}"
    assert bad_loo > base, (
        f"alpha=10 LOO-MAE {bad_loo:.4f} must fail even the constant baseline {base:.4f}")


# --------------------------------------------------------------------------- bathroom probe


def test_bathroom_probe_alphas_are_in_the_usable_band():
    assert 0.001 <= bathroom_probe.ALPHA_SOFT <= bathroom_probe.ALPHA_HEAVY <= 0.5, (
        f"ALPHA_SOFT={bathroom_probe.ALPHA_SOFT} / ALPHA_HEAVY={bathroom_probe.ALPHA_HEAVY} "
        "collapse the probe; keep both in [0.001, 0.5] with heavy >= soft")


def _bath_dataset(n: int = 36, seed: int = 7):
    # n=36 is deliberately above SOFT_N(30) so alpha_for() selects ALPHA_SOFT,
    # exercising the production "softened" path against the shipped default.
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        good = i % 2 == 0
        X.append(_overlapping(rng, good, bathroom_probe.EMBED_DIM))
        y.append(1.0 if good else 0.0)  # (bath_score-1)/4
    return np.stack(X), np.asarray(y, dtype=np.float32)


def _bath_predict_01(v, w, b):
    # compare in [0,1] space to match y, so _loo_mae subtracts like-for-like
    return float(np.clip(float(v @ w) + b, 0.0, 1.0))


def test_bathroom_probe_beats_constant_baseline_loo():
    X, y = _bath_dataset()
    alpha = bathroom_probe.alpha_for(X.shape[0])  # ALPHA_SOFT (n=36 > SOFT_N)
    loo = _loo_mae(lambda a, t: bathroom_probe._ridge_fit(a, t, alpha),
                   _bath_predict_01, X, y)
    base = _constant_mae(y)
    assert loo < base, f"bathroom probe LOO-MAE {loo:.4f} >= constant baseline {base:.4f}"


def test_bathroom_probe_output_spread_is_not_collapsed():
    X, y = _bath_dataset()
    w, b = bathroom_probe._ridge_fit(X, y, bathroom_probe.alpha_for(X.shape[0]))
    raw = X @ w + b
    assert float(raw.std()) > 0.05, (
        f"bathroom probe output std {raw.std():.4f} is effectively constant")


def test_bathroom_probe_output_collapses_at_the_old_heavy_alpha(monkeypatch):
    """Negative control for ALPHA_HEAVY, exercising the heavy arm (n <= SOFT_N), which
    nothing covered. Measured: ALPHA_HEAVY=0.05 -> output std 0.477, alpha=100 -> 0.005."""
    X, y = _bath_dataset(n=bathroom_probe.SOFT_N - 6)
    assert bathroom_probe.alpha_for(X.shape[0]) == bathroom_probe.ALPHA_HEAVY
    w_good, b_good = bathroom_probe._ridge_fit(X, y, bathroom_probe.ALPHA_HEAVY)
    w_bad, b_bad = bathroom_probe._ridge_fit(X, y, 100.0)
    assert float((X @ w_good + b_good).std()) > 0.05
    assert float((X @ w_bad + b_bad).std()) < 0.02, (
        "alpha=100 must collapse the bathroom probe output")


# --------------------------------------------------------------------------- gate behavior


def test_is_degenerate_true_when_probe_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(bathroom_probe, "PROBE_PATH", tmp_path / "absent.npz")
    assert bathroom_probe.is_degenerate(object()) is True


def test_is_degenerate_true_below_min_rated(tmp_path, monkeypatch):
    # Seed a VALID probe first: is_degenerate returns at the probe-absent branch
    # before it ever reaches the min-rated branch, so without this file the test
    # was a duplicate of test_is_degenerate_true_when_probe_absent.
    probe_path = tmp_path / "small.npz"
    np.savez(probe_path, weights=np.zeros(vp.EMBED_DIM, dtype=np.float32),
             bias=np.float32(0.5), n_trained=np.int32(50))
    monkeypatch.setattr(bathroom_probe, "PROBE_PATH", probe_path)
    conn = database.connect(tmp_path / "few.db")
    database.upsert_listing(conn, {"listing_id": "L0", "title": "L0", "price": 12000})
    database.rate_listing(conn, "L0", 4.0, 5.0)  # no bathroom-labelled images
    assert bathroom_probe.load_probe() is not None, "probe must load, or the branch is unreachable"
    assert bathroom_probe.is_degenerate(conn) is True
    conn.close()


def test_is_degenerate_false_on_a_healthy_probe(tmp_path, monkeypatch):
    """The gate's False return is what lets the hint through; nothing asserted it."""
    monkeypatch.setattr(bathroom_probe, "PROBE_PATH", tmp_path / "healthy.npz")
    conn = database.connect(tmp_path / "healthy.db")
    good = np.zeros(vp.EMBED_DIM, dtype=np.float32)
    good[[0, 1, 2]] = 1.0
    bad = np.zeros(vp.EMBED_DIM, dtype=np.float32)
    bad[[30, 31, 32]] = 1.0
    rng = np.random.default_rng(5)
    for i in range(24):
        lid = f"H{i}"
        is_bath = i % 2 == 0
        database.upsert_listing(conn, {"listing_id": lid, "title": lid, "price": 12000})
        database.rate_listing(conn, lid, 4.0, 5.0 if is_bath else 1.0)
        vec = _unit(np.asarray(good if is_bath else bad, dtype=np.float32)
                    + rng.standard_normal(vp.EMBED_DIM).astype(np.float32) * 0.05)
        # Every listing needs an is_bathroom=1 photo: get_bath_rated_samples counts
        # only those, so labelling just one class would leave a constant y and the
        # probe would collapse for a reason unrelated to the gate under test.
        database.replace_images(conn, lid, [{
            "ordinal": 0, "image_path": f"/p/{lid}_0", "is_bathroom": 1,
            "dino_embedding": vec.tobytes(),
        }])
    _w, _b, n = bathroom_probe.train_and_save(conn)
    assert n >= bathroom_probe.MIN_BATH_RATED
    assert bathroom_probe.is_degenerate(conn) is False
    conn.close()


def test_collapsed_probe_prompt_omits_bath_hint():
    """A degenerate probe must not inject a constant estimate into the vision prompt."""
    from src import vision_llm

    listing = {"title": "t", "bath_model_score": 3.8, "bath_hint_ok": False}
    text = vision_llm.build_messages(listing, [], None)[1]["content"]
    assert "Independent bathroom-quality estimate" not in text

    listing["bath_hint_ok"] = True
    text_ok = vision_llm.build_messages(listing, [], None)[1]["content"]
    assert "Independent bathroom-quality estimate" in text_ok


def test_zero_bath_score_omits_hint_even_when_probe_is_healthy():
    """(bath_model_score=0.0, bath_hint_ok=True) -- the dominant production case.

    0.0 is "no bathroom evidence", and vision_llm's `if bath_score and ...` treats
    it as falsy, so the hint must stay out of the prompt no matter how healthy the
    probe is. A naive `is not None` refactor would start leaking the estimate in.
    """
    from src import vision_llm

    listing = {"title": "t", "bath_model_score": 0.0, "bath_hint_ok": True}
    text = vision_llm.build_messages(listing, [], None)[1]["content"]
    assert "Independent bathroom-quality estimate" not in text


# --------------------------------------------------------------------------- cross-fitting


def _seed_memorisable(conn, n=27, seed=3):
    """Labeled rows whose embeddings carry a strong but not perfect class signal."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        good = i % 2 == 0
        vec = _unit(_overlapping(rng, good, vp.EMBED_DIM)
                    + rng.standard_normal(vp.EMBED_DIM).astype(np.float32) * 0.05)
        lid = f"L{i}"
        database.upsert_listing(conn, {
            "listing_id": lid, "title": lid,
            "dino_embedding": vec.astype(np.float32).tobytes()})
        database.rate_listing(conn, lid, 5.0 if good else 1.0)
    return database.get_rated_samples(conn)


def test_cross_fitted_values_are_less_leaky_than_in_sample(tmp_path):
    """Out-of-fold Layer-1 values must not reproduce the in-sample memorization.

    A probe fit on the labeled rows predicts them nearly perfectly, so a head
    trained on that column reaches 100% of its gain from a single split. The
    cross-fitted column has to describe unseen-listing behaviour instead.
    """
    conn = database.connect(tmp_path / "xfit.db")
    rows = _seed_memorisable(conn)
    cf = vp.cross_fitted_values(rows)
    assert cf is not None and len(cf) == len(rows) and np.isfinite(cf).all()

    y = np.asarray([float(r["user_score"]) for r in rows])
    X = vp._vectors(rows)
    w, b = vp.train_probe(X, (y - 1) / 4)
    corr_in = abs(np.corrcoef(np.clip(X @ w + b, 0, 1), y)[0, 1])
    corr_cf = abs(np.corrcoef(cf, y)[0, 1])
    assert corr_in > 0.99, f"expected in-sample memorization, got corr {corr_in:.3f}"
    assert corr_cf < corr_in, (
        f"cross-fitted corr {corr_cf:.3f} should be below in-sample {corr_in:.3f}")
    conn.close()


def test_cross_fitted_values_none_when_too_few_rows():
    fake = [{"dino_embedding": np.zeros(vp.EMBED_DIM, dtype=np.float32).tobytes(),
             "user_score": 3.0} for _ in range(2)]
    assert vp.cross_fitted_values(fake) is None


def test_cross_fitted_values_nan_for_rows_without_embeddings(tmp_path):
    conn = database.connect(tmp_path / "nan.db")
    rows = _seed_memorisable(conn, n=6)
    rows = list(rows)
    rows[0] = {"dino_embedding": None, "user_score": 3.0}   # no usable embedding
    cf = vp.cross_fitted_values(rows)
    assert cf is not None
    assert np.isnan(cf[0]), "row without an embedding must be NaN so callers fall back"
    assert np.isfinite(cf[1:]).all()
    conn.close()


def test_train_and_save_spreads_importance_beyond_dino(tmp_path, monkeypatch):
    """Cross-fitting must move gain OFF the Layer-1 column (differential check).

    The old assertion (`others.max() > 1e-4` given `dino < 0.99`) was a tautology,
    and its `dino < 0.99` rested on a tie-break between the leaked column and
    perfectly-collinear price/area features -- a coin flip across xgboost builds.
    This version (a) gives price/area only a NOISY second signal so they cannot
    perfectly separate the labels, and (b) measures the SAME seed twice, once with
    cross-fitting and once with the pre-fix in-sample column, asserting the
    differential rather than any absolute share.
    """
    import xgboost as xgb

    from src import scoring

    def _dino_gain():
        m = xgb.XGBRegressor()
        m.load_model(str(scoring.MODEL_PATH))
        imp = m.feature_importances_
        return float(imp[scoring.FEATURE_NAMES.index("dino_visual_score")])

    def _seed(db_name, n=30, seed=9):
        conn = database.connect(tmp_path / db_name)
        rows = _seed_memorisable(conn, n=n, seed=seed)
        rng = np.random.default_rng(seed + 1)
        for i, r in enumerate(rows):
            good = i % 2 == 0
            price = 13000.0 + (600.0 if good else -600.0) + float(rng.normal(0.0, 500.0))
            area = 12.0 + (1.5 if good else -1.5) + float(rng.normal(0.0, 2.0))
            conn.execute("UPDATE listings SET price=?, area=? WHERE listing_id=?",
                         (price, area, r["listing_id"]))
        conn.commit()
        return conn

    monkeypatch.setattr(scoring, "MODEL_PATH", tmp_path / "head.json")
    monkeypatch.setattr(vp, "PROBE_PATH", tmp_path / "probe.npz")
    monkeypatch.setattr(scoring, "RATED_THRESHOLD", 5)

    conn_cf = _seed("cf.db")
    scoring.train_and_save(conn_cf)
    dino_cf = _dino_gain()

    # Pre-fix regime: the head trains on the probe's IN-SAMPLE column.
    conn_in = _seed("in.db")
    monkeypatch.setattr(vp, "cross_fitted_values", lambda rows: None)
    scoring.train_and_save(conn_in)
    dino_in = _dino_gain()

    # The in-sample column memorizes the labels, so the leaky head must lean hard
    # on it -- this is the collapse the fix exists to prevent.
    assert dino_in > 0.5, (
        f"leaky control did not reproduce the collapse (dino gain {dino_in:.3f}); "
        "the differential below would be meaningless")
    # Cross-fitted: the head cannot memorize through dino_visual_score any more.
    assert dino_cf < 0.80, f"cross-fitted head still leans on dino_visual_score ({dino_cf:.3f})"
    assert dino_in - dino_cf > 0.15, (
        f"cross-fitting moved only {dino_in - dino_cf:.3f} of gain off the Layer-1 "
        f"column (in-sample {dino_in:.3f} vs cross-fit {dino_cf:.3f})")
    conn_cf.close()
    conn_in.close()


def test_cross_fitted_values_aligns_past_a_middle_hole(tmp_path):
    """Guards the `out[i]` vs `out[idx[i]]` class: the hole is in the MIDDLE, and the
    first VALID row (index 0 of X) must still get its own held-out prediction.

    The existing NaN test only ever removes row 0, where the index map is a constant
    shift -- a bug that assumed "row 0 is the hole" would pass it.
    """
    conn = database.connect(tmp_path / "midhole.db")
    rows = list(_seed_memorisable(conn, n=7))
    rows[3] = {"dino_embedding": None, "user_score": 3.0}
    cf = vp.cross_fitted_values(rows)
    assert cf is not None
    assert np.isnan(cf[3]), "the holed row must be NaN"
    assert np.isfinite(cf[[0, 1, 2, 4, 5, 6]]).all()

    X, y = vp.probe_dataset(rows)
    keep = np.ones(X.shape[0], dtype=bool)
    keep[0] = False  # rows[0] is the first VALID row -> X row 0
    w, b = vp.train_probe(X[keep], y[keep])
    assert cf[0] == pytest.approx(vp.probe_value(X[0], w, b), abs=1e-5)
    conn.close()


def test_fit_probe_trains_despite_a_row_without_an_embedding(tmp_path):
    """Regression: `_vectors` drops NULL/short blobs but `y` used to be built from ALL
    rated rows, so a single embedding-less row made `aug.T @ y` raise -- killing the
    probe AND silently disabling cross-fitting, with only a log line to show for it."""
    conn = database.connect(tmp_path / "hole.db")
    _seed_memorisable(conn, n=8)
    rows = list(database.get_rated_samples(conn))
    victim = rows[2]["listing_id"]
    conn.execute("UPDATE listings SET dino_embedding=NULL WHERE listing_id=?", (victim,))
    conn.commit()
    fresh = list(database.get_rated_samples(conn))
    assert sum(1 for r in fresh if not r["dino_embedding"]) == 1

    w, _b = vp.fit_probe(conn)         # must not raise
    assert w.shape == (vp.EMBED_DIM,)
    X, y = vp.probe_dataset(fresh)     # X and y must be the same length
    assert X.shape[0] == y.shape[0] == 7
    conn.close()


def test_train_and_save_warns_when_cross_fit_is_unavailable(tmp_path, monkeypatch, caplog):
    """A regression to the in-sample leak must never be silent."""
    from src import scoring

    monkeypatch.setattr(scoring, "MODEL_PATH", tmp_path / "head.json")
    monkeypatch.setattr(vp, "PROBE_PATH", tmp_path / "probe.npz")
    monkeypatch.setattr(scoring, "RATED_THRESHOLD", 5)
    monkeypatch.setattr(vp, "cross_fitted_values", lambda rows: None)
    conn = database.connect(tmp_path / "nofit.db")
    _seed_memorisable(conn, n=30, seed=9)
    with caplog.at_level("WARNING", logger="src.scoring"):
        scoring.train_and_save(conn)
    assert any("IN-SAMPLE" in r.message for r in caplog.records), (
        "expected an explicit leak-regression warning; the fallback is currently silent")

    import xgboost as xgb
    m = xgb.XGBRegressor()
    m.load_model(str(scoring.MODEL_PATH))
    assert m.get_booster().attributes().get("basis") == "in-sample"
    conn.close()


def test_train_and_save_stamps_the_head_and_probe_pair(tmp_path, monkeypatch):
    """The head trains on a cross-fitted column that is NOT derivable from the npz, so
    the pair can only be validated if both carry the same train_id."""
    import xgboost as xgb

    from src import scoring

    monkeypatch.setattr(scoring, "MODEL_PATH", tmp_path / "head.json")
    monkeypatch.setattr(vp, "PROBE_PATH", tmp_path / "probe.npz")
    monkeypatch.setattr(scoring, "RATED_THRESHOLD", 5)
    conn = database.connect(tmp_path / "stamped.db")
    _seed_memorisable(conn, n=30, seed=9)
    scoring.train_and_save(conn)

    meta = vp.load_probe_meta()
    assert meta is not None and meta["train_id"] and meta["alpha"] == vp.PROBE_ALPHA
    m = xgb.XGBRegressor()
    m.load_model(str(scoring.MODEL_PATH))
    attrs = m.get_booster().attributes()
    assert attrs["probe_train_id"] == meta["train_id"]
    assert attrs["feature_names_sha"] == scoring.FEATURE_NAMES_SHA
    assert attrs["basis"] == "out-of-fold"
    conn.close()


def test_head_is_stale_when_the_probe_is_replaced(tmp_path, monkeypatch):
    from src import scoring

    monkeypatch.setattr(scoring, "MODEL_PATH", tmp_path / "head.json")
    monkeypatch.setattr(vp, "PROBE_PATH", tmp_path / "probe.npz")
    monkeypatch.setattr(scoring, "RATED_THRESHOLD", 5)
    conn = database.connect(tmp_path / "stale.db")
    _seed_memorisable(conn, n=30, seed=9)
    scoring.train_and_save(conn)

    import xgboost as xgb
    m = xgb.XGBRegressor()
    m.load_model(str(scoring.MODEL_PATH))
    assert scoring._head_is_stale(m, conn) is False
    # a probe from a different generation must invalidate the head
    vp.save_probe(np.zeros(vp.EMBED_DIM, dtype=np.float32), 0.9, n_trained=30)
    assert scoring._head_is_stale(m, conn) is True
    conn.close()


def test_head_is_stale_on_an_unstamped_legacy_head(tmp_path, monkeypatch):
    """A head with no stamps cannot be proven against a STAMPED probe -> retrain.

    The old check short-circuited on the missing `probe_train_id`/`probe_alpha` and
    returned "fresh" whenever the rating count matched -- restoring an old head
    next to a new probe looked valid while serving mismatched geometry.
    """
    import xgboost as xgb

    from src import scoring

    monkeypatch.setattr(scoring, "MODEL_PATH", tmp_path / "head.json")
    monkeypatch.setattr(vp, "PROBE_PATH", tmp_path / "probe.npz")
    monkeypatch.setattr(scoring, "RATED_THRESHOLD", 5)
    conn = database.connect(tmp_path / "legacy.db")
    _seed_memorisable(conn, n=30, seed=9)
    scoring.train_and_save(conn)

    m = xgb.XGBRegressor()
    m.load_model(str(scoring.MODEL_PATH))
    assert scoring._head_is_stale(m, conn) is False

    # Overwrite with a genuine pre-stamp head: same width, zero booster attributes.
    legacy = xgb.XGBRegressor(n_estimators=1)
    legacy.fit(np.zeros((4, len(scoring.FEATURE_NAMES)), dtype=np.float32), np.zeros(4))
    legacy.save_model(str(scoring.MODEL_PATH))
    m = xgb.XGBRegressor()
    m.load_model(str(scoring.MODEL_PATH))
    assert (m.get_booster().attributes() or {}) == {}, "control must be a truly unstamped head"
    assert scoring._head_is_stale(m, conn) is True
    conn.close()


def test_check_score_health_never_rewrites_models(tmp_path, monkeypatch):
    """The --incoming health check is predict-only: a missing head is reported,
    not trained. `_load_trained_model(allow_retrain=False)` is the guard."""
    from src import scoring

    monkeypatch.setattr(scoring, "MODEL_PATH", tmp_path / "head.json")
    monkeypatch.setattr(vp, "PROBE_PATH", tmp_path / "probe.npz")
    monkeypatch.setattr(scoring, "RATED_THRESHOLD", 5)
    conn = database.connect(tmp_path / "health.db")
    _seed_memorisable(conn, n=30, seed=9)
    scoring.check_score_health(conn)           # no head exists -> must not train one
    assert not scoring.MODEL_PATH.exists(), "check_score_health wrote a model"
    assert not vp.PROBE_PATH.exists(), "check_score_health wrote a probe"
    conn.close()


def test_cross_fitted_values_none_when_embeddings_have_no_variance(tmp_path):
    """Identical labeled embeddings make LOO emit only the fold mean -- a y-leak.

    Must bail out so the caller keeps the full-probe value; otherwise the head
    splits on fold composition instead of on real features.
    """
    conn = database.connect(tmp_path / "degenerate.db")
    same = _unit(np.arange(vp.EMBED_DIM, dtype=np.float32) + 1.0)
    for i in range(10):
        lid = f"D{i}"
        database.upsert_listing(conn, {
            "listing_id": lid, "title": lid,
            "dino_embedding": same.astype(np.float32).tobytes()})
        database.rate_listing(conn, lid, 5.0 if i % 2 == 0 else 1.0)
    assert vp.cross_fitted_values(database.get_rated_samples(conn)) is None
    conn.close()
