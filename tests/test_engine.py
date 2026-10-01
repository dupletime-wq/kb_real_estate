import numpy as np
import pandas as pd

from kbforecast.evaluation import WFConfig, walk_forward
from kbforecast.features import build_features, make_targets
from kbforecast.intervals import conformal_quantiles
from kbforecast.kb_panel import KBPanel, build_hierarchy
from kbforecast.macro import align_weekly
from kbforecast import models as M
from tests.synthetic import make_panel


def test_hierarchy_keys_unique_and_seoul_names_plain():
    names = ["전국", "서울특별시", "강북14개구", "중구", "서구", "부산광역시", "중구", "서구", "대구광역시", "중구"]
    h = build_hierarchy(names)
    assert h["key"].is_unique
    # Seoul 중구 keeps its plain name; same-named districts elsewhere are prefixed with their city/province
    assert h.loc[(h["name"] == "중구") & (h["province"] == "서울특별시"), "key"].iloc[0] == "중구"
    assert "부산광역시 중구" in set(h["key"]) and "대구광역시 중구" in set(h["key"])


def test_features_are_causal():
    kb = make_panel()
    fs = build_features(kb)
    cut = kb.sale.index[300]
    kb2 = KBPanel(kb.sale.loc[:cut], kb.jeonse.loc[:cut], {k: v.loc[:cut] for k, v in kb.sentiment.items()}, kb.hierarchy, "x", ())
    fs2 = build_features(kb2)
    a = fs.X.xs(cut, level="date").sort_index(axis=1)
    b = fs2.X.xs(cut, level="date").sort_index(axis=1)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-5)


def test_walk_forward_training_labels_are_closed_before_origin():
    kb = make_panel()
    fs = build_features(kb)
    h = 13
    seen = []

    def spy(X_train, y_train, X_pred):
        seen.append((X_train.index.get_level_values("date").max(), X_pred.index.get_level_values("date").min()))
        return np.zeros(len(X_pred))

    cols = fs.groups["own"]
    cfg = WFConfig(horizon=h, first_origin="2013-01-07", eval_step=2, refit_every=26, min_train_rows=200)
    walk_forward(fs, cols, spy, cfg)
    assert seen
    for last_train_date, first_eval_date in seen:
        assert last_train_date + pd.Timedelta(weeks=h) <= first_eval_date


def test_macro_alignment_has_no_lookahead():
    monthly = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-02-01"]), "value": [1.0, 2.0]})
    weeks = pd.date_range("2020-01-06", periods=16, freq="W-MON")
    out = align_weekly(monthly, weeks, lag_days=60)
    assert out[weeks < pd.Timestamp("2020-03-01")].isna().all()  # Jan value not visible before Jan 1 + 60d
    assert out[weeks >= pd.Timestamp("2020-04-01")].iloc[0] == 2.0
    assert out[(weeks >= pd.Timestamp("2020-03-05")) & (weeks < pd.Timestamp("2020-03-31"))].eq(1.0).all()


def test_conformal_interval_ignores_unrealised_labels():
    kb = make_panel()
    fs = build_features(kb)
    h = 13
    y = make_targets(fs.log_price, h)
    pred = pd.DataFrame({"y": y, "pred": y.groupby(level="region").transform("mean")}).dropna(subset=["pred"])
    scale = pd.Series(0.05, index=pred.index)
    dates = fs.log_price.index
    base = conformal_quantiles(pred, scale, h, dates, min_calib=50)
    origin = dates[250]
    tampered = pred.copy()
    future = tampered.index.get_level_values("date") + pd.Timedelta(weeks=h) > origin
    tampered.loc[future, "y"] += 10.0  # labels that are not yet realised at `origin`
    alt = conformal_quantiles(tampered, scale, h, dates, min_calib=50)
    at = base.index.get_level_values("date") == origin
    np.testing.assert_allclose(base.loc[at, ["lo", "hi"]].to_numpy(), alt.loc[at, ["lo", "hi"]].to_numpy())


def test_pooled_ridge_recovers_a_simple_signal():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(4000, 3)), columns=list("abc"))
    y = 0.5 * X["a"].to_numpy() + rng.normal(scale=0.1, size=4000)
    pred = M.ridge_model(1.0)(X.iloc[:3000], y[:3000], X.iloc[3000:])
    assert np.corrcoef(pred, y[3000:])[0, 1] > 0.9


def test_hgb_tolerates_columns_that_are_all_nan_or_constant_in_training():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.normal(size=600), "late_survey": np.nan, "const": 1.0})
    X.loc[500:, "late_survey"] = rng.normal(size=100)  # only observed after the training window
    y = 0.4 * X["a"].to_numpy() + rng.normal(scale=0.1, size=600)
    pred = M.hgb_model(max_iter=50, min_samples_leaf=20)(X.iloc[:400], y[:400], X.iloc[400:])
    assert np.isfinite(pred).all() and np.corrcoef(pred, y[400:])[0, 1] > 0.7


def test_pruned_features_are_left_out_of_the_models_but_kept_for_intervals():
    from kbforecast.engine import PRUNED_FEATURES, model_columns

    fs = build_features(make_panel())
    cols = model_columns(fs)
    assert not set(cols) & PRUNED_FEATURES
    assert {"vol13", "vol52"} <= set(fs.X.columns)  # still computed: they scale the prediction intervals
    assert {"r1", "r4", "r8", "r39", "rel13", "rel26"} <= set(cols)  # the families whose removal hurt stay


def test_long_horizons_use_the_ridge_alone():
    from kbforecast.engine import ANCHORS, LONG_HORIZON, RIDGE_ALPHA, blend_model

    assert 78 in ANCHORS and 104 in ANCHORS and max(ANCHORS) == 104
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = X["a"].to_numpy() * 0.01 + rng.normal(0, 0.01, 300)
    Xp = pd.DataFrame(rng.normal(size=(20, 4)), columns=list("abcd"))
    long_pred = blend_model(LONG_HORIZON)(X, y, Xp)
    ridge_pred = M.ridge_model(RIDGE_ALPHA[LONG_HORIZON])(X, y, Xp)
    np.testing.assert_allclose(long_pred, ridge_pred)
    assert not np.allclose(blend_model(26)(X, y, Xp), M.ridge_model(RIDGE_ALPHA[26])(X, y, Xp))  # shorter horizons still blend in the trees


def test_pooled_mean_baseline_is_causal_and_uses_closed_labels_only():
    from kbforecast.evaluation import pooled_mean_baseline

    kb = make_panel()
    fs = build_features(kb)
    h = 13
    base = pooled_mean_baseline(fs, h)
    cut = kb.sale.index[300]
    kb2 = KBPanel(kb.sale.loc[:cut], kb.jeonse.loc[:cut], {k: v.loc[:cut] for k, v in kb.sentiment.items()}, kb.hierarchy, "x", ())
    base2 = pooled_mean_baseline(build_features(kb2), h)
    a, b = base.xs(cut, level="date"), base2.xs(cut, level="date")
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index(), check_exact=False, atol=1e-6)  # unchanged by cutting the future
    y = make_targets(fs.log_price, h)
    usable = fs.X[["r52", "vol52"]].notna().all(axis=1)
    pos = list(fs.log_price.index).index(cut)
    closed = fs.log_price.index[: pos - h + 1]
    sel = y.loc[(closed, slice(None))][usable.loc[(closed, slice(None))]].dropna()
    assert np.isclose(float(a.iloc[0]), float(sel.mean()), atol=1e-6)  # exactly the mean of labels that had closed by `cut`


def test_interval_score_rewards_covering_and_penalises_misses():
    from kbforecast.intervals import interval_score

    y, lo, hi = np.array([0.0, 0.0, 0.5]), np.array([-0.1, 0.1, -0.1]), np.array([0.1, 0.2, 0.1])
    s = interval_score(y, lo, hi, alpha=0.10)
    assert np.isclose(s[0], 0.2)  # covered: just the width
    assert np.isclose(s[1], 0.1 + 20 * 0.1) and np.isclose(s[2], 0.2 + 20 * 0.4)  # missed: width + (2/alpha) * distance



def test_engine_overlay_stops_at_the_cutoff_and_labels_use_observed_prices_only():
    from kbforecast.engine import fit_engine
    from kbforecast.overlay import RateSeries

    kb = make_panel(extra_cities=60)
    obs = {"sale": kb.sale.notna(), "jeonse": kb.jeonse.notna()}
    obs["sale"].iloc[100:103, :] = False  # three weeks of filled prices in every series
    kb = KBPanel(kb.sale, kb.jeonse, kb.sentiment, kb.hierarchy, kb.fingerprint, kb.warnings, obs)
    frame = pd.DataFrame({"date": pd.to_datetime(["2008-01-01", "2012-01-02", "2014-06-02", "2020-01-01"]), "value": [3.0, 2.5, 3.0, 3.5]})
    rate = RateSeries(frame, pd.Timestamp("2020-01-01"), "test")
    fit = fit_engine(kb, anchors=(13, 78), first_origin="2013-01-07", refit_every=52, eval_step=4, rate=rate, overlay_max_horizon=52)
    assert 13 in fit.overlay and 78 not in fit.overlay  # no overlay (and no overlay slope) beyond the cutoff
    assert "overlay" in fit.predictions[13].columns and "overlay" not in fit.predictions[78].columns
    assert fit.settings["overlay_max_horizon"] == 52 and fit.settings["observed_only_labels"] is True
    filled_week = kb.sale.index[101]
    realised = fit.predictions[13].dropna(subset=["y"])
    origin_dates = realised.index.get_level_values("date")
    assert not ((origin_dates == filled_week) | (origin_dates == filled_week - pd.Timedelta(weeks=13))).any()  # no label touches a filled price
