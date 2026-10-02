import numpy as np
import pandas as pd

from kbforecast.calibration import BIAS_CANDIDATES, BiasCfg, median_bias_correction, walk_forward_select


def _panel(n=260, bias=0.03, h=13, seed=0, regions=("서울특별시", "강남구", "부산광역시", "대구광역시")):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-02", periods=n + 60, freq="W-MON")
    sel = dates[:n:2]
    idx = pd.MultiIndex.from_product([sel, regions], names=["date", "region"])
    y = rng.normal(0.02, 0.03, len(idx))
    pred = y - bias + rng.normal(0, 0.02, len(idx))  # the forecast under-predicts by `bias`
    return pd.DataFrame({"y": y, "pred": pred}, index=idx), dates


SEOUL = {"서울특별시", "강남구"}
SMALL = {"global": 40, "seoul": 20}  # the synthetic panel has 4 regions, the real thresholds assume ~200


def test_correction_uses_only_residuals_whose_target_week_has_passed():
    frame, dates = _panel()
    cfg = BiasCfg("global_x1.0", "global", 1.0)
    full, per_origin = median_bias_correction(frame, dates, 13, cfg, SEOUL, SMALL)
    # change every outcome that is NOT yet closed at the probe origin: the correction at that origin must not move
    probe = dates[150]
    changed = frame.copy()
    not_closed = (changed.index.get_level_values("date") > probe - pd.Timedelta(weeks=13))
    changed.loc[not_closed, "y"] += 1.0
    again, per_origin2 = median_bias_correction(changed, dates, 13, cfg, SEOUL, SMALL)
    assert per_origin[probe] == per_origin2[probe]
    # ... and it does react to a closed outcome
    closed = changed.index.get_level_values("date") < probe - pd.Timedelta(weeks=14)
    changed2 = frame.copy()
    changed2.loc[closed, "y"] += 0.5
    assert median_bias_correction(changed2, dates, 13, cfg, SEOUL, SMALL)[1][probe] != per_origin[probe]


def test_correction_recovers_a_stable_bias_and_shrink_scales_it():
    frame, dates = _panel(bias=0.03)
    late = frame.index.get_level_values("date") > dates[200]
    full, _ = median_bias_correction(frame, dates, 13, BiasCfg("g1", "global", 1.0), SEOUL, SMALL)
    half, _ = median_bias_correction(frame, dates, 13, BiasCfg("g05", "global", 0.5), SEOUL, SMALL)
    assert abs(full.loc[late, "bias_corr"].median() - 0.03) < 0.006
    assert np.allclose(half["bias_corr"], 0.5 * full["bias_corr"])
    mae = lambda f: (f["pred"] - f["y"]).abs()[late].mean()  # noqa: E731
    assert mae(full) < mae(frame.assign(pred=frame["pred"]))  # removes the systematic part of the error
    assert (full["pred_precorr"] == frame["pred"]).all()  # the uncorrected forecast stays available


def test_seoul_scope_touches_only_seoul_rows_and_none_changes_nothing():
    frame, dates = _panel()
    out, _ = median_bias_correction(frame, dates, 13, BiasCfg("s1", "seoul", 1.0), SEOUL, SMALL)
    outside = ~out.index.get_level_values("region").isin(SEOUL)
    assert (out.loc[outside, "bias_corr"] == 0).all() and (out.loc[~outside, "bias_corr"] != 0).any()
    none, per = median_bias_correction(frame, dates, 13, BIAS_CANDIDATES[0], SEOUL, SMALL)
    assert (none["pred"] == frame["pred"]).all() and per.empty


def test_unlabelled_rows_never_enter_the_residual_pool_and_no_bias_means_no_correction_choice():
    frame, dates = _panel(bias=0.0)
    frame.loc[frame.index.get_level_values("date") < dates[100], "y"] = np.nan  # e.g. targets built from filled prices were dropped
    out, per = median_bias_correction(frame, dates, 13, BiasCfg("g1", "global", 1.0), SEOUL, SMALL)
    early = out.index.get_level_values("date") < dates[100]
    assert (out.loc[early, "bias_corr"] == 0).all()  # nothing closed + labelled yet -> no correction
    sel, table = walk_forward_select(*_panel(bias=0.0, n=520), 13, SEOUL, windows=(("2018-01-01", "2019-12-31"),), min_rows=SMALL)
    assert table["chosen"].iloc[0] == "none"  # without bias nothing beats `none` by the required margin


def test_selection_uses_only_closed_origins_and_picks_a_correction_when_bias_is_real():
    frame, dates = _panel(bias=0.04, n=560)
    sel, table = walk_forward_select(frame, dates, 13, SEOUL, windows=(("2018-01-01", "2019-12-31"),), min_rows=SMALL)
    row = table.iloc[0]
    assert row["chosen"] != "none" and row["selection_rows"] > 0
    start = int(dates.searchsorted(pd.Timestamp("2018-01-01")))
    last_used = (frame.index.get_level_values("date").map(lambda d: dates.get_loc(d)) + 13 <= start).sum()
    assert row["selection_rows"] <= last_used  # only origins whose target week closed before the window started
    in_win = (sel.index.get_level_values("date") >= "2018-01-01") & (sel.index.get_level_values("date") <= "2019-12-31")
    assert (sel.loc[~in_win, "bias_corr"] == 0).all() and (sel.loc[in_win, "bias_corr"] != 0).any()
