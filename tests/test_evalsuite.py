import numpy as np
import pandas as pd

from kbforecast import evalsuite as E


def _frame(n_dates=120, regions=("서울특별시", "강남구", "부산광역시"), seed=0, bias=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-05", periods=n_dates, freq="2W-MON")
    idx = pd.MultiIndex.from_product([dates, regions], names=["date", "region"])
    y = rng.normal(0.02, 0.03, len(idx))
    base = y + rng.normal(0, 0.02, len(idx)) + bias
    cand = y + rng.normal(0, 0.02, len(idx))
    return pd.DataFrame({"y": y, "base": base, "cand": cand}, index=idx)


def test_units_and_sign_conventions():
    idx = pd.MultiIndex.from_product([pd.date_range("2020-01-06", periods=2, freq="W-MON"), ["서울특별시"]], names=["date", "region"])
    f = pd.DataFrame({"y": [0.10, 0.00], "pred": [0.00, 0.10]}, index=idx)
    s = E.error_stats(f)
    assert np.isclose(s["MAE_log_pp"], 10.0) and np.isclose(s["bias_pred_minus_actual_pp"], 0.0)  # +-0.10 cancel in bias, not in MAE
    assert np.isclose(s["median_residual_actual_minus_pred_pp"], 0.0)
    assert np.isclose(s["MAE_simple_pp"], (abs(np.expm1(0.0) - np.expm1(0.10)) * 100 + abs(np.expm1(0.10) - np.expm1(0.0)) * 100) / 2)  # a different unit
    assert s["MAE_simple_pp"] != s["MAE_log_pp"]
    under = pd.DataFrame({"y": [0.1], "pred": [0.0]}, index=idx[:1])
    t = E.error_stats(under)
    assert t["bias_pred_minus_actual_pp"] < 0 and t["median_residual_actual_minus_pred_pp"] > 0  # under-prediction: bias negative, residual positive


def test_region_sets_split_seoul_city_districts_and_outside():
    sets = E.region_sets(["서울특별시", "강북14개구", "강남11개구", "강남구", "노원구", "부산광역시", "경기도 수원시"])
    assert sets["서울시 지수"] == {"서울특별시"} and sets["서울 25개 구"] == {"강남구", "노원구"}
    assert sets["서울 28"] == {"서울특별시", "강북14개구", "강남11개구", "강남구", "노원구"} and sets["서울 외"] == {"부산광역시", "경기도 수원시"}
    assert sets["전체"] == sets["서울 28"] | sets["서울 외"]


def test_origin_losses_average_regions_of_an_origin_together():
    f = _frame()
    losses = E.origin_losses(f, "base", "cand", {"서울특별시", "강남구"})
    assert len(losses) == f.index.get_level_values("date").nunique()  # one row per origin, regions already averaged
    d0 = f.xs(f.index.get_level_values("date")[0], level="date").loc[["서울특별시", "강남구"]]
    assert np.isclose(losses["base"].iloc[0], ((d0["base"] - d0["y"]).abs() * 100).mean())


def test_hac_and_bootstrap_detect_a_real_improvement_and_not_a_null():
    f = _frame(n_dates=300, seed=1)
    all_regions = set(f.index.get_level_values("region"))
    better = E.compare(f, "base", "cand", 26, {"전체": all_regions}, n_boot=500)["groups"]["전체"]
    assert better["diff_MAE_log_pp"] < 0 and better["hac_abs"]["p_improvement"] < 0.1 and better["boot_abs"]["rel_hi_pct"] < 0
    same = f.assign(cand=f["base"] + np.random.default_rng(5).normal(0, 1e-4, len(f)))
    null = E.compare(same, "base", "cand", 26, {"전체": all_regions}, n_boot=500)["groups"]["전체"]
    assert abs(null["rel_MAE_pct"]) < 0.5 and null["boot_abs"]["rel_lo_pct"] < 0 < null["boot_abs"]["rel_hi_pct"]


def test_block_bootstrap_resamples_whole_blocks_of_origins():
    base = np.arange(100, dtype=float) + 1.0
    cand = base * 0.9
    out = E.block_bootstrap(base, cand, horizon=26, n_boot=300, seed=3)
    assert out["block"] == 13  # ceil(26 weeks / 2-week origin step): the origins whose 26-week windows overlap
    assert out["rel_hi_pct"] < -9.9 and out["rel_lo_pct"] > -10.1  # a constant -10% ratio survives any resampling


def test_origin_state_is_causal_and_uses_only_past_thresholds():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2012-01-02", periods=200, freq="W-MON")
    idx = pd.MultiIndex.from_product([dates, [f"r{i}" for i in range(40)]], names=["date", "region"])
    s = pd.Series(rng.normal(size=len(idx)) + np.repeat(np.linspace(0, 3, 200), 40), index=idx)
    full = E.origin_state(s, min_obs=400)
    cut = 120
    part = E.origin_state(s[s.index.get_level_values("date") <= dates[cut]], min_obs=400)
    pd.testing.assert_series_equal(full[full.index.get_level_values("date") <= dates[cut]], part)  # cutting the future changes nothing
    assert full.dropna().isin([0, 1, 2]).all() and full.isna().iloc[:300].all()  # not defined before enough history


def _per_h(rel, outside=0.0, period=-2.0, hi=-1.0):
    g = {"rel_MAE_pct": rel, "periods": {p: {"rel_pct": period} for p in E.PERIODS}, "boot_abs": {"rel_hi_pct": hi}}
    return {"groups": {"서울 28": g, "서울 외": {"rel_MAE_pct": outside}}}


def test_decision_rule_is_fixed_and_conservative():
    adopt = {h: _per_h(-3.0) for h in E.PRIMARY_HORIZONS}
    assert E.decide(adopt)["verdict"] == "채택"
    assert E.decide({h: _per_h(+0.5) for h in E.PRIMARY_HORIZONS})["verdict"] == "기각"
    assert E.decide({h: _per_h(-0.3) for h in E.PRIMARY_HORIZONS})["verdict"] == "보류"  # better, but not by enough
    worse_outside = {**adopt, 26: _per_h(-3.0, outside=1.2)}
    assert E.decide(worse_outside)["verdict"] == "보류"
    ns = {h: _per_h(-3.0, hi=+0.2) for h in E.PRIMARY_HORIZONS}
    assert E.decide(ns)["verdict"] == "보류"  # interval does not exclude zero
    assert E.decide({13: _per_h(-3.0)})["verdict"] != "채택"  # missing horizons never adopt


def test_experiment_log_counts_candidates_and_tests_and_annotates_p_values(tmp_path):
    log = E.ExperimentLog(tmp_path / "log.jsonl")
    for i in range(4):
        log.record(f"c{i}", "A", {"k": i}, [13], {"score": -0.1 * i})
    log.record("c1", "A", {"k": 1}, [26], {})  # same candidate on another horizon: not a new candidate, but a new test
    log.record("c1", "A", {"k": 1}, [104], {})  # a non-primary horizon never counts as a test
    log.record("overlay", "reference", {}, [52], {})  # reference comparisons are not tries
    assert log.n_candidates() == 4 and log.n_tests() == 5 and log.families() == {"A": 4}
    assert np.isclose(log.adjusted_p(0.02), 0.10) and log.adjusted_p(0.02, include_prior=True) == 1.0
    table = pd.DataFrame({"candidate": list("abc"), "p_abs": [0.01, 0.20, np.nan]})
    out = log.annotate(table)
    assert np.isclose(out.loc[0, "p_bonferroni_logged"], 0.05) and out.loc[1, "p_bonferroni_logged"] == 1.0 and out["p_raw"].tolist()[:2] == [0.01, 0.20]
    assert np.isclose(out.loc[0, "p_holm_this_table"], 0.02) and np.isclose(out.loc[1, "p_holm_this_table"], 0.20) and np.isnan(out.loc[2, "p_holm_this_table"])
    assert (out["n_candidates_logged"] == 4).all() and (out["evidence"] == "exploratory").all() and "earlier README candidates" in log.summary_text()


def test_verdict_marks_evidence_and_names_the_horizons_behind_the_score():
    only52 = {52: _per_h(+3.28)}
    d = E.decide(only52)
    assert d["score_horizons"] == [52] and d["verdict"] == "기각" and any("missing" in r and "not a 13/26/52-week average" in r for r in d["reasons"])
    assert "탐색적" in d["evidence"] and E.decide({h: _per_h(-3.0) for h in E.PRIMARY_HORIZONS})["score_horizons"] == [13, 26, 52]
