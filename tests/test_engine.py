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
