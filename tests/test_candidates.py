import dataclasses

import numpy as np
import pandas as pd
import pytest

from kbforecast import candidates as C
from kbforecast import models as M
from kbforecast.kb_panel import KBPanel
from tests.synthetic import make_panel


def _kb_with_observed(seed=0, weeks=420):
    kb = make_panel(weeks=weeks, seed=seed)
    obs = kb.sale.notna() | True
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(rng.random(kb.sale.shape) > 0.04, index=kb.sale.index, columns=kb.sale.columns)  # ~4% filled cells
    return dataclasses.replace(kb, observed={"sale": obs, "jeonse": obs})


def _truncate(kb: KBPanel, n: int) -> KBPanel:
    idx = kb.sale.index[:n]
    obs = None if kb.observed is None else {k: v.loc[idx] for k, v in kb.observed.items()}
    return dataclasses.replace(kb, sale=kb.sale.loc[idx], jeonse=kb.jeonse.loc[idx], sentiment={k: v.loc[idx] for k, v in kb.sentiment.items()}, observed=obs)


def test_run_length_counts_consecutive_same_sign_weeks_and_caps():
    sign = np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0], [0.0], [1.0], [np.nan]])
    out = C._run_length(sign, cap=2)[:, 0]
    assert out[:6].tolist() == [1.0, 2.0, 2.0, -1.0, -2.0, 0.0] and out[6] == 1.0 and np.isnan(out[7])


def test_sentiment_and_quality_features_do_not_depend_on_the_future():
    kb = _kb_with_observed()
    full = {**C.sentiment_price_features(kb), **C.quality_features(kb)}
    cut = 300
    part = {**C.sentiment_price_features(_truncate(kb, cut)), **C.quality_features(_truncate(kb, cut))}
    assert set(full) == set(C.FAMILIES["B"]) | set(C.FAMILIES["C"])
    for name in full:
        pd.testing.assert_frame_equal(full[name].iloc[:cut], part[name], check_exact=False, atol=1e-9, obj=name)


def test_observation_age_and_fill_ratio_follow_the_mask():
    kb = make_panel(weeks=60)
    obs = pd.DataFrame(True, index=kb.sale.index, columns=kb.sale.columns)
    obs.iloc[30:33, :] = False  # three filled weeks
    out = C.quality_features(dataclasses.replace(kb, observed={"sale": obs, "jeonse": obs}))
    age, ratio = out["obs_age"]["서울특별시"], out["fill_ratio26"]["서울특별시"]
    assert age.iloc[29] == 0 and age.iloc[30] == 1 and age.iloc[32] == 3 and age.iloc[33] == 0
    assert ratio.iloc[32] == pytest.approx(3 / 26) and ratio.iloc[20] == 0


def test_without_an_observed_mask_nothing_is_invented():
    kb = make_panel(weeks=60)
    assert kb.observed is None and C.quality_features(kb) == {}
    with pytest.raises(ValueError):
        C.build_candidate_features(kb, ("obs_age",))
    with pytest.raises(KeyError):
        C.build_candidate_features(kb, ("not_a_feature",))
    with pytest.raises(ValueError):
        C.build_candidate_features(kb, ("vol_rel36",))  # needs the trade history


def _history(months: int, base: float = 100.0) -> pd.DataFrame:
    rows = []
    for k, m in enumerate(pd.period_range("2008-01", periods=months, freq="M")):
        day = m.to_timestamp() + pd.Timedelta(days=14)
        for name, code in __import__("kbforecast.trades", fromlist=["x"]).SEOUL_GU_CODES.items():
            rows.append({"sgg_cd": code, "deal_date": day.strftime("%Y-%m-%d"), "n_all": base + 10 * (k % 5), "n_cancelled": 0})
    return pd.DataFrame(rows)


def test_volume_features_are_point_in_time_and_not_interpolated():
    kb = make_panel(weeks=700)
    kb = dataclasses.replace(kb, hierarchy=kb.hierarchy)
    hist = _history(180)
    base = C.volume_features(kb, hist)
    assert set(base) == set(C.FAMILIES["V"]) and base["vol_rel36"]["강남구"].notna().any()
    # Seoul series only: outside Seoul the columns exist but stay empty
    assert base["vol_rel36"]["경기도"].isna().all() and base["vol_chg3"]["수원시"].isna().all()
    t = kb.sale.index.get_loc(pd.Timestamp("2016-06-06"))
    ref = base["vol_rel36"]["서울특별시"].iloc[t]
    # contracts newer than the assumed lag (12 weeks before 2020) are invisible: rewrite everything after the cutoff month
    cutoff = kb.sale.index[t] - pd.Timedelta(weeks=12)
    changed = hist.copy()
    newer = pd.to_datetime(changed["deal_date"]) > cutoff
    changed.loc[newer, "n_all"] *= 7
    assert C.volume_features(kb, changed)["vol_rel36"]["서울특별시"].iloc[t] == ref
    # a month that ended before the cutoff is visible
    older = pd.to_datetime(hist["deal_date"]).between(cutoff - pd.Timedelta(days=60), cutoff - pd.Timedelta(days=35))
    changed2 = hist.copy()
    changed2.loc[older, "n_all"] *= 7
    assert C.volume_features(kb, changed2)["vol_rel36"]["서울특별시"].iloc[t] != ref
    # the weekly value is the latest known month held constant (no drift toward the next month inside a month)
    s = base["vol_rel36"]["서울특별시"]
    w = s.loc["2016-06-06":"2016-06-27"].dropna()
    assert w.nunique() <= 2


def test_volume_features_use_the_shorter_lag_only_from_2020_05_25():
    kb = make_panel(weeks=800)
    hist = _history(210)
    f = C.volume_features(kb, hist)["vol_rel36"]["서울특별시"]
    before, after = pd.Timestamp("2020-05-18"), pd.Timestamp("2020-06-01")
    changed = hist.copy()
    d = pd.to_datetime(changed["deal_date"])
    # a month ending 9-11 weeks before 2020-06-01 is known on that date (8-week lag) but was not known 2 weeks earlier (12-week lag)
    month = d.between("2020-03-01", "2020-03-31")
    changed.loc[month, "n_all"] *= 5
    g = C.volume_features(kb, changed)["vol_rel36"]["서울특별시"]
    assert g.loc[after] != f.loc[after] and g.loc[before] == f.loc[before]


def test_time_split_purges_label_windows_and_timeval_hgb_picks_a_count():
    dates = pd.date_range("2010-01-04", periods=300, freq="W-MON")
    fit, val = M.time_split_masks(np.repeat(dates, 5), purge_weeks=26, val_share=0.2)
    d = np.repeat(dates, 5)
    assert not (fit & val).any() and d[fit].max() < d[val].min() - pd.Timedelta(weeks=26) + pd.Timedelta(days=1)
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([dates, [f"r{i}" for i in range(220)]], names=["date", "region"])  # 66,000 rows
    X = pd.DataFrame(rng.normal(size=(len(idx), 4)), index=idx, columns=list("abcd"))
    y = 0.3 * X["a"].to_numpy() + rng.normal(0, 1, len(idx))
    rec = []
    pred = M.hgb_model_timeval(13, max_iter=120, min_samples_leaf=50, record=rec)(X.iloc[:30000], y[:30000], X.iloc[30000:30100])
    assert len(pred) == 100 and rec[0]["selected_on"].startswith("purged") and 20 <= rec[0]["n_iter"] <= 120 and rec[0]["early_stopping_active"] is False
    rec2 = []
    M.hgb_model(max_iter=60, min_samples_leaf=50, record=rec2)(X, y, X.iloc[:10])  # row_stride=2 -> 33,000 rows fitted
    assert rec2[0]["early_stopping_active"] is True  # scikit-learn 'auto' with > 10,000 fitted rows: a random 10% hold-out decides when to stop
    rec_small = []
    M.hgb_model(max_iter=60, min_samples_leaf=50, record=rec_small)(X.iloc[:15000], y[:15000], X.iloc[:10])  # 7,500 fitted rows after the stride
    assert rec_small[0]["early_stopping_active"] is False  # production's row_stride halves the count BEFORE the 10,000-row rule applies
    rec3 = []
    M.hgb_model(max_iter=60, min_samples_leaf=50, early_stopping=False, record=rec3)(X, y, X.iloc[:10])
    assert rec3[0]["early_stopping_active"] is False and rec3[0]["n_iter"] == 60


def test_feature_experiment_runner_end_to_end_on_a_synthetic_panel(tmp_path):
    """Smoke test of the whole candidate pipeline (labels, overlay, candidates, comparison, verdicts, log) on synthetic data; the numbers mean nothing."""
    from kbforecast import evalsuite as E
    from kbforecast.experiments import RunConfig, run_feature_experiments
    from kbforecast.variants import NAMED_VARIANTS

    kb = make_panel(weeks=420, extra_cities=60)
    rng = np.random.default_rng(1)
    obs = pd.DataFrame(rng.random(kb.sale.shape) > 0.03, index=kb.sale.index, columns=kb.sale.columns)
    kb = dataclasses.replace(kb, observed={"sale": obs, "jeonse": obs})
    history = _history(190)
    cfg = RunConfig(first_origin="2013-06-03", refit_every=52, eval_step=4, min_train_rows=3000, n_boot=100, primary=(13, 26), extra=(78,))
    variants = [NAMED_VARIANTS[n] for n in ("B_px_sent", "C_obs_age", "D_hgb_fixed", "V_chg3")]
    log = E.ExperimentLog(tmp_path / "log.jsonl")
    out = run_feature_experiments(kb, cfg, variants, tmp_path / "run", log, history, combine=False)
    assert set(out["verdicts"]) >= {v.name for v in variants} | {f"{v.name} [Ridge only]" for v in variants if v.extra_features}
    assert all(v["verdict"] in ("채택", "보류", "기각") for v in out["verdicts"].values())
    assert (tmp_path / "run" / "candidate_summary.csv").exists() and log.n_candidates() == 4
    summary = pd.read_csv(tmp_path / "run" / "candidate_summary.csv")
    assert {"MAE_base", "MAE_cand", "MAE_simple_base", "RMSE_cand", "bias_cand", "p_abs", "boot90_rel_lo"} <= set(summary.columns)
    assert set(summary["horizon"]) == {13, 26, 78}  # primary horizons decide, extra horizons are only reported


def test_data_diagnostics_reports_stale_filled_and_estimated_inputs():
    from kbforecast.hub import HubReport
    from kbforecast.overlay import RateSeries
    from kbforecast.report import data_diagnostics

    kb = make_panel(weeks=200)
    obs = pd.DataFrame(True, index=kb.sale.index, columns=kb.sale.columns)
    obs.iloc[-1, :] = False  # the newest week was filled everywhere
    obs.iloc[-30:-20, kb.sale.columns.get_loc("노원구")] = False
    kb = dataclasses.replace(kb, observed={"sale": obs, "jeonse": obs})
    report = HubReport(True, "x", tuple(kb.sale.index[-2:]), 10, 11, ("수원시",), ("강남11개구",))
    rate = RateSeries(pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "value": [3.0]}), kb.sale.index[-1] - pd.Timedelta(days=30), "t")
    diag = data_diagnostics(kb, report, None, rate, None, today=kb.sale.index[-1] + pd.Timedelta(weeks=6))
    head = diag["headline"].set_index("항목")
    assert head.loc["오늘 기준 경과 주수", "상태"] == "오래됨" and "강남11개구" in head.loc["추정(상위 범위 변화폭)으로 채운 심리 범위", "값"]
    assert "수원시" in head.loc["허브 시계열이 끝난 지역", "값"]
    reg = diag["regions"].set_index("region")
    assert (reg["status"] == "최근 관측 공백").all() and reg.loc["노원구", "filled_share_last_26w"] > reg.loc["강남구", "filled_share_last_26w"] - 1e-9
    sent = diag["sentiment"]
    assert sent.loc[sent["scope"] == "강남11개구", "newest_weeks"].str.startswith("추정").all() and not sent.loc[sent["scope"] == "서울특별시", "newest_weeks"].str.startswith("추정").any()


def test_engine_variant_changes_columns_settings_and_correction_but_default_is_untouched():
    from kbforecast.calibration import BiasCfg
    from kbforecast.engine import fit_engine
    from kbforecast.variants import EngineVariant

    kb = make_panel(weeks=420, extra_cities=60)
    kw = dict(anchors=(13,), first_origin="2013-06-03", refit_every=52, eval_step=4)
    default = fit_engine(kb, **kw)
    assert "variant" not in default.settings and "px_sent" not in default.columns  # production behaviour is the same as before
    v = EngineVariant("t", extra_features=("px_sent",), hgb_mode="fixed", bias=BiasCfg("seoul_x0.5", "seoul", 0.5))
    fit = fit_engine(kb, variant=v, **kw)
    assert fit.columns[-1] == "px_sent" and fit.settings["variant"]["hgb_mode"] == "fixed" and fit.settings["variant"]["bias"]["name"] == "seoul_x0.5"
    assert "bias_corr" in fit.predictions[13].columns and "bias_corr" not in default.predictions[13].columns
    with pytest.raises(ValueError):
        fit_engine(kb, variant=EngineVariant("v", extra_features=("vol_rel36",)), **kw)  # volume features without history are refused, not invented
