import json
import dataclasses

import numpy as np
import pandas as pd
import pytest

from kbforecast import candidates as C
from kbforecast import models as M
from kbforecast.kb_panel import KBPanel, seoul_region_keys
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
    out = run_feature_experiments(kb, cfg, variants, tmp_path / "run", log, history, windows=(("2014-01-06", "2014-12-31"), ("2015-01-05", "2016-12-31")), min_internal_rows=50)
    assert set(out["verdicts"]) >= {v.name for v in variants} | {f"{v.name} [Ridge only]" for v in variants if v.extra_features}
    assert all(v["verdict"] in ("채택", "보류", "기각") and "탐색적" in v["evidence"] for v in out["verdicts"].values())  # every verdict says it is exploratory
    run = tmp_path / "run"
    assert (run / "candidate_scorecard_exploratory.csv").exists() and (run / "selection_history.csv").exists() and (run / "selection_external_summary.csv").exists()
    summary = pd.read_csv(run / "candidate_scorecard_exploratory.csv")
    assert {"MAE_base", "MAE_cand", "MAE_simple_base", "RMSE_cand", "bias_cand", "p_raw", "p_bonferroni_logged", "p_bonferroni_with_prior_assumed", "p_holm_this_table", "n_candidates_logged", "n_tests_logged", "evidence"} <= set(summary.columns)
    assert set(summary["horizon"]) == {13, 26, 78}  # primary horizons decide, extra horizons are only reported
    has_p = summary["p_raw"].notna()
    assert (summary["n_candidates_logged"] >= 4).all() and (summary.loc[has_p, "p_bonferroni_logged"] >= summary.loc[has_p, "p_raw"]).all() and has_p.any()
    hist = pd.read_csv(run / "selection_history.csv")
    assert len(hist) == 2 and set(hist["chosen"]) <= {"baseline", *(v.name for v in variants)} or hist["chosen"].str.startswith("combo_").any()
    assert "selection_procedure" in {e["candidate"] for e in log.entries()} and "탐색적" in json.loads((run / "selection_verdict.json").read_text(encoding="utf-8"))["evidence"]


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


def test_train_start_masks_labels_of_earlier_origins_and_the_runner_accepts_it(tmp_path):
    from kbforecast import evalsuite as E
    from kbforecast.experiments import RunConfig, mask_targets, run_feature_experiments
    from kbforecast.features import make_targets
    from kbforecast.variants import NAMED_VARIANTS

    kb = make_panel(weeks=420, extra_cities=60)
    y = make_targets(np.log(kb.sale), 13)
    start = pd.Timestamp("2011-06-06")
    masked = mask_targets(y, start)
    dates = masked.index.get_level_values("date")
    assert masked[dates < start].isna().all() and masked[dates >= start].equals(y[dates >= start]) and mask_targets(y, None) is y
    cfg = RunConfig(first_origin="2013-06-03", refit_every=104, eval_step=4, min_train_rows=3000, n_boot=50, primary=(13,), extra=())
    out = run_feature_experiments(kb, cfg, [NAMED_VARIANTS["D_hgb_fixed"]], tmp_path / "r", E.ExperimentLog(tmp_path / "log.jsonl"), None, with_ridge_check=False, train_start=start, windows=(("2014-01-06", "2016-12-31"),), min_internal_rows=50)
    assert "D_hgb_fixed" in out["verdicts"]


def _repro_frames(noise=0.0, shift_label=0.0, drop_rows=0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-06", periods=60, freq="2W-MON")
    regions = ["서울특별시", "강남구", "부산광역시"]
    idx = pd.MultiIndex.from_product([dates, regions], names=["date", "region"])
    y = rng.normal(0.02, 0.03, len(idx))
    raw = y + rng.normal(0, 0.02, len(idx))
    overlay = np.where(idx.get_level_values("region").isin(["서울특별시", "강남구"]), -0.002, 0.0)
    frozen = pd.DataFrame({"y": y, "raw": raw, "pred": raw + overlay}, index=idx)
    fresh = pd.DataFrame({"y": y + shift_label, "pred_raw": raw + rng.normal(0, noise, len(idx)) if noise else raw, "pred": 0.0}, index=idx)
    fresh["pred"] = fresh["pred_raw"] + overlay
    if drop_rows:
        fresh = fresh.iloc[drop_rows:]
    return {52: frozen}, {52: fresh}


def _repro(frozen, fresh, **over):
    from kbforecast.experiments import reproduction_report

    kw = dict(fingerprint="aa", frozen_fingerprint="aa", fresh_columns=["r1", "r4"], frozen_columns=["r1", "r4"], refit_every=26, frozen_refit_every=26, labels_observed=True, frozen_labels="observed")
    kw.update(over)
    return reproduction_report(fresh, frozen, **kw)


def test_reproduction_report_passes_on_identical_runs_and_reports_actual_differences():
    frozen, fresh = _repro_frames()
    ok = _repro(frozen, fresh)
    assert ok["verdict"] == "pass" and ok["table"].loc[0, "final_max_abs_diff"] == 0 and ok["table"].loc[0, "y_rows_differing"] == 0
    frozen, fresh = _repro_frames(noise=0.004)
    warn = _repro(frozen, fresh)
    t = warn["table"].iloc[0]
    assert warn["verdict"] == "warn" and t["final_mean_abs_diff"] > 5e-4 and 0.9 < t["final_corr"] < 1 and t["raw_mean_abs_diff"] > 0  # same setup, forecasts drift: not a pass
    assert warn["problems"] == [] and warn["warnings"]


def test_reproduction_report_fails_on_a_different_dataset_labels_rows_or_schedule():
    frozen, fresh = _repro_frames()
    assert _repro(frozen, fresh, fingerprint="bb")["verdict"] == "fail"
    assert _repro(frozen, fresh, refit_every=39)["verdict"] == "fail"
    assert _repro(frozen, fresh, labels_observed=False)["verdict"] == "fail"
    assert _repro(frozen, fresh, fresh_columns=["r1"])["verdict"] == "fail"
    f2, n2 = _repro_frames(shift_label=0.01)
    r = _repro(f2, n2)
    assert r["verdict"] == "fail" and "labels differ" in r["problems"] and r["table"].loc[0, "y_rows_differing"] > 0
    f3, n3 = _repro_frames(drop_rows=5)
    r3 = _repro(f3, n3)
    assert r3["verdict"] == "fail" and r3["table"].loc[0, "frozen_rows_missing_in_fresh"] == 5
    f4, n4 = _repro_frames()
    n4[52]["pred"] = n4[52]["pred"] + np.where(n4[52].index.get_level_values("region") == "서울특별시", 0.01, 0.0)  # overlay applied differently
    assert "overlay term differs" in _repro(f4, n4)["problems"]


def test_candidate_script_runs_end_to_end_with_a_stand_in_workbook(tmp_path, monkeypatch, capsys):
    """Runs scripts/experiment_candidates.py itself (reproduction report, HGB record, state decomposition, candidate A, family C scorecard and
    selection validation) on a synthetic panel with the parser and the frozen artifact replaced; guards the script-level wiring, not any number."""
    import importlib.util
    import sys
    from pathlib import Path

    from kbforecast import selection as S

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("experiment_candidates_script", root / "scripts" / "experiment_candidates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kb = make_panel(weeks=420, extra_cities=60)
    rng = np.random.default_rng(3)
    obs = pd.DataFrame(rng.random(kb.sale.shape) > 0.03, index=kb.sale.index, columns=kb.sale.columns)
    kb = dataclasses.replace(kb, observed={"sale": obs, "jeonse": obs}, fingerprint="standin-workbook")
    frozen_idx = pd.MultiIndex.from_product([pd.date_range("2014-01-06", periods=20, freq="2W-MON"), ["서울특별시", "강남구"]], names=["date", "region"])
    frozen = pd.DataFrame({"y": 0.0, "raw": 0.0, "pred": 0.0}, index=frozen_idx)
    meta = {"long_config": {"data_fingerprint": "some-other-workbook", "refit_every": 26, "labels": "observed"}, "long_feature_columns": ["r1"]}
    monkeypatch.setattr(mod, "parse_kb_panel", lambda _bytes: kb)
    monkeypatch.setattr(mod, "load_frozen_baselines", lambda: ({52: frozen}, kb.sale.index, meta))
    monkeypatch.setattr(S, "EXTERNAL_WINDOWS", (("2015-01-05", "2015-12-31"), ("2016-01-04", "2016-12-31")))
    monkeypatch.setattr(S, "MIN_INTERNAL_SEOUL_ROWS", 20)
    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["experiment_candidates.py", str(wb), "--families", "C", "--n-boot", "40"])
    mod.main()
    text = capsys.readouterr().out
    out = next((tmp_path / "experiments").glob("new_workbook_*"))  # a different workbook is stored separately
    assert "mode: new_workbook" in text and "EXPLORATORY scorecard" in text and "EXTERNAL validation of the selection procedure" in text and "No production setting was changed" in text
    for name in ("baseline_reproduction.json", "hgb_baseline_behaviour.json", "candidate_A_summary.csv", "candidate_scorecard_exploratory.csv", "selection_history.csv", "run_metadata.json"):
        assert (out / name).exists(), name
    run_meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert run_meta["mode"] == "new_workbook" and "versions" in run_meta and "탐색적" in run_meta["evidence"]
    assert json.loads((out / "baseline_reproduction.json").read_text(encoding="utf-8"))["verdict"] == "fail"  # a different dataset never counts as a reproduction
    assert (tmp_path / "experiments" / "log.jsonl").exists()


def test_truncated_panel_is_cut_everywhere_and_its_fingerprint_says_so():
    from kbforecast.kb_panel import truncate_panel

    kb = _kb_with_observed(weeks=300)
    cut = kb.sale.index[200]
    t = truncate_panel(kb, cut)
    assert t.sale.index.max() == cut == t.jeonse.index.max() == t.observed["sale"].index.max() and all(v.index.max() <= cut for v in t.sentiment.values())
    assert t.fingerprint != kb.fingerprint and "truncated" in t.fingerprint and len(kb.sale) == 300  # the original is untouched


def test_reproduction_by_truncation_cannot_verify_the_fingerprint_but_still_checks_labels():
    frozen, fresh = _repro_frames()
    ok = _repro(frozen, fresh, fingerprint="new-file-cut", require_same_fingerprint=False)
    assert ok["verdict"] == "pass" and ok["checks"]["data_fingerprint_equal"] is None
    f2, n2 = _repro_frames(shift_label=0.01)  # a revised history shows up as different labels
    assert _repro(f2, n2, fingerprint="new-file-cut", require_same_fingerprint=False)["verdict"] == "fail"


def test_long_memory_and_valuation_features_are_causal_and_defined_as_documented():
    kb = _kb_with_observed(weeks=420)
    full = {**C.long_memory_features(kb), **C.valuation_features(kb)}
    cut = 330
    part_kb = _truncate(kb, cut)
    part = {**C.long_memory_features(part_kb), **C.valuation_features(part_kb)}
    assert set(full) == set(C.FAMILIES["L"]) | set(C.FAMILIES["J"])
    for name in full:
        pd.testing.assert_frame_equal(full[name].iloc[:cut], part[name], check_exact=False, atol=1e-9, obj=name)
    L = np.log(kb.sale)
    assert np.isclose(full["r104"]["서울특별시"].iloc[200], L["서울특별시"].iloc[200] - L["서울특별시"].iloc[96])
    assert np.isclose(full["sj_level"]["강남구"].iloc[50], np.log(kb.sale["강남구"].iloc[50] / kb.jeonse["강남구"].iloc[50]))
    assert full["pdev260"]["서울특별시"].iloc[:155].isna().all() and full["pdev260"]["서울특별시"].iloc[160:].notna().all()  # needs 156 weeks first
    got = C.build_candidate_features(kb, ("r104", "sj_z156"))
    assert list(got) == ["r104", "sj_z156"]
    seoul = [c for c in kb.sale.columns if c in seoul_region_keys(kb.hierarchy)]
    other = [c for c in kb.sale.columns if c not in seoul]
    assert seoul and other
    pd.testing.assert_frame_equal(full["sj_level_seoul"][seoul], full["sj_level"][seoul])  # same values for Seoul
    assert full["sj_level_seoul"][other].isna().all().all() and full["sj_level"][other].notna().any().any()  # nothing elsewhere


def test_time_validated_ridge_picks_alpha_per_group_and_never_validates_on_its_own_fit_rows():
    from kbforecast import models as M

    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-04", periods=260, freq="W-MON")
    regions = [f"서울{i}" for i in range(15)] + [f"지방{i}" for i in range(45)]
    idx = pd.MultiIndex.from_product([dates, regions], names=["date", "region"])
    X = pd.DataFrame(rng.normal(size=(len(idx), 5)), index=idx, columns=list("abcde"))
    is_seoul = idx.get_level_values("region").str.startswith("서울")
    # Seoul has a real signal, the rest is pure noise: a group-specific alpha should shrink the rest much harder
    y = np.where(is_seoul, 0.5 * X["a"].to_numpy(), 0.0) + rng.normal(0, 0.5, len(idx))
    rec = []
    fn = M.ridge_model_timeval(13, group_fn=lambda r: "seoul" if r.startswith("서울") else "other", record=rec)
    pred = fn(X.iloc[:12000], y[:12000], X.iloc[12000:12200])
    alphas = rec[0]["alpha_by_group"]
    assert len(pred) == 200 and np.isfinite(pred).all() and alphas["seoul"] < alphas["other"]
    rec2 = []
    M.ridge_model_timeval(13, record=rec2)(X.iloc[:12000], y[:12000], X.iloc[12000:12010])
    assert list(rec2[0]["alpha_by_group"]) == ["0"] and rec2[0]["alpha_by_group"]["0"] in M.ALPHA_GRID


def test_long_horizon_script_runs_end_to_end_on_a_stand_in_workbook(tmp_path, monkeypatch, capsys):
    """scripts/experiment_long_horizon.py with the parser replaced and the reproduction cut skipped; guards the wiring of 52/104/208-week primary horizons."""
    import importlib.util
    import sys
    from pathlib import Path

    from kbforecast import selection as S

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("experiment_long_horizon_script", root / "scripts" / "experiment_long_horizon.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kb = make_panel(weeks=560, extra_cities=60)
    rng = np.random.default_rng(5)
    obs = pd.DataFrame(rng.random(kb.sale.shape) > 0.03, index=kb.sale.index, columns=kb.sale.columns)
    kb = dataclasses.replace(kb, observed={"sale": obs, "jeonse": obs}, fingerprint="standin-long")
    monkeypatch.setattr(mod, "parse_kb_panel", lambda _bytes: kb)
    monkeypatch.setattr(mod, "CANDIDATES", ("R_alpha_cv", "L_valuation"))
    monkeypatch.setattr(S, "EXTERNAL_WINDOWS", (("2016-01-04", "2016-12-31"), ("2017-01-02", "2018-12-31")))
    monkeypatch.setattr(S, "MIN_INTERNAL_SEOUL_ROWS", 20)
    wb = tmp_path / "wb.xlsx"
    wb.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["experiment_long_horizon.py", str(wb), "--n-boot", "30", "--reproduction-cut", "", "--min-train-rows", "1500"])
    mod.main()
    text = capsys.readouterr().out
    out = next((tmp_path / "experiments").glob("long_horizon_*"))
    assert "EXPLORATORY scorecard (52/104/208 decide" in text and "EXTERNAL validation of the selection procedure" in text and "nothing was added to the app" in text.replace("No production setting was changed and ", "")
    for name in ("context_model_vs_naive.csv", "candidate_scorecard_exploratory.csv", "selection_history.csv", "run_metadata.json"):
        assert (out / name).exists(), name
    sc = pd.read_csv(out / "candidate_scorecard_exploratory.csv")
    assert set(sc["horizon"]) == {52, 78, 104, 208}
    v = json.loads((out / "verdicts_exploratory.json").read_text(encoding="utf-8"))
    assert all(x["score_horizons"] in ([52, 104, 208], [52, 104], [52], [104], [104, 208], [208], [52, 208], []) for x in v.values())


def test_fx_features_use_only_rates_up_to_the_day_before_and_are_causal(tmp_path):
    kb = _kb_with_observed(weeks=420)
    days = pd.date_range(kb.sale.index[0] - pd.Timedelta(days=400), kb.sale.index[-1] + pd.Timedelta(days=30), freq="D")
    rate = pd.Series(1000.0 + np.cumsum(np.random.default_rng(1).normal(0, 3, len(days))), index=days)
    path = tmp_path / "fx.csv"
    pd.DataFrame({"date": days, "value": rate.to_numpy()}).to_csv(path, index=False)
    full = C.fx_features(kb, path)
    assert set(full) == set(C.FAMILIES["X"])
    t = 200
    expected = np.log(rate.loc[kb.sale.index[t] - pd.Timedelta(days=1)]) - np.log(rate.loc[kb.sale.index[t - 26] - pd.Timedelta(days=1)])
    assert np.isclose(full["fx_r26"]["서울특별시"].iloc[t], expected)
    # changing the rate ON or after the week's date cannot move that week's feature
    altered = rate.copy()
    altered.loc[kb.sale.index[t]:] *= 1.5
    path2 = tmp_path / "fx2.csv"
    pd.DataFrame({"date": days, "value": altered.to_numpy()}).to_csv(path2, index=False)
    changed = C.fx_features(kb, path2)
    for name in full:
        pd.testing.assert_frame_equal(full[name].iloc[:t + 1], changed[name].iloc[:t + 1], obj=name)
    # Seoul-only versions: same values for Seoul, empty elsewhere; the common versions fill every region
    seoul = [c for c in kb.sale.columns if c in seoul_region_keys(kb.hierarchy)]
    other = [c for c in kb.sale.columns if c not in seoul]
    assert seoul and other
    pd.testing.assert_frame_equal(full["fx_dev156_seoul"][seoul], full["fx_dev156"][seoul])
    assert full["fx_dev156_seoul"][other].isna().all().all() and full["fx_dev156"][other].iloc[300:].notna().all().all()


def test_population_names_are_normalised_and_features_use_only_known_months(tmp_path):
    from kbforecast import regional as R

    assert R.normalise_name("전남광주통합특별시 동구") == ("광주", "동구")
    assert R.normalise_name("전남광주통합특별시 목포시") == ("전남", "목포시")
    assert R.normalise_name("강원특별자치도 춘천시") == ("강원", "춘천시")
    assert R.normalise_name("인천광역시 남구") == ("인천", "미추홀구")
    kb = _kb_with_observed(weeks=420)
    months = pd.period_range("2008-01", periods=250, freq="M")
    rows = []
    for key in ("서울특별시", "강남구"):
        for i, m in enumerate(months):
            rows.append({"ym": str(m), "code": "1100000000" if key == "서울특별시" else "1168000000", "name": "서울특별시" if key == "서울특별시" else "서울특별시 강남구",
                         "pop": 1_000_000 * (1.002 ** i), "households": 400_000 * (1.001 ** i)})
    pop = pd.DataFrame(rows)
    full = R.population_features(kb, pop)
    assert set(full) == set(C.FAMILIES["P"])
    t = 300
    d = kb.sale.index[t]
    last_usable = (d - pd.Timedelta(days=R.POP_LAG_DAYS)).to_period("M")
    last_usable = last_usable if last_usable.end_time.normalize() <= d - pd.Timedelta(days=R.POP_LAG_DAYS) else last_usable - 1
    i = (last_usable - months[0]).n
    expected = np.log(1.002 ** 12)
    assert np.isclose(full["pop_g12"]["서울특별시"].iloc[t], expected) and i >= 12
    # changing a month that is not yet usable at week t leaves every earlier week unchanged
    altered = pop.copy()
    cutoff = (last_usable + 1).strftime("%Y-%m")
    altered.loc[altered["ym"] >= cutoff, "pop"] *= 3
    changed = R.population_features(kb, altered)
    for name in full:
        pd.testing.assert_frame_equal(full[name].iloc[:t + 1], changed[name].iloc[:t + 1], obj=name)
