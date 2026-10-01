import numpy as np
import pandas as pd

from kbforecast.kb_panel import _fill_short_gaps


def test_gap_fill_carries_forward_and_never_uses_later_values():
    idx = pd.date_range("2020-01-06", periods=8, freq="W-MON")
    wide = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan, np.nan], "b": [np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0]}, index=idx)
    out = _fill_short_gaps(wide, limit=2)
    assert out["a"].iloc[1] == 1.0 and out["a"].iloc[2] == 1.0  # not interpolated towards the later 4.0
    assert out["a"].iloc[5] == 5.0 and out["a"].iloc[6] == 5.0 and np.isnan(out["a"].iloc[7])  # at most `limit` weeks, also after a series ends
    assert np.isnan(out["b"].iloc[0])  # nothing before the first observation
    assert out["b"].iloc[2] == 2.0 and out["b"].iloc[3] == 2.0 and np.isnan(out["b"].iloc[4])


def test_cutting_off_the_future_never_changes_earlier_rows():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-06", periods=40, freq="W-MON")
    wide = pd.DataFrame(rng.normal(size=(40, 3)).cumsum(axis=0), index=idx, columns=list("abc"))
    wide[rng.random(wide.shape) < 0.25] = np.nan  # scattered gaps, runs of any length, some at the very end of a prefix
    full = _fill_short_gaps(wide)
    for k in range(1, 41):
        pd.testing.assert_frame_equal(_fill_short_gaps(wide.iloc[:k]), full.iloc[:k])  # the truncated raw data gives the same prefix


def test_targets_built_only_from_observed_prices():
    from kbforecast.features import make_targets

    idx = pd.date_range("2020-01-06", periods=8, freq="W-MON")
    price = pd.DataFrame({"a": np.log([100, 101, 102, 103, 104, 105, 106, 107.0])}, index=idx)
    observed = pd.DataFrame({"a": [True, True, False, True, True, True, True, True]}, index=idx)  # week 2 is a filled value
    plain = make_targets(price, 2)
    masked = make_targets(price, 2, observed)
    assert plain.notna().sum() == 6 and masked.notna().sum() == 4
    assert np.isnan(masked.loc[(idx[0], "a")]) and np.isnan(masked.loc[(idx[2], "a")])  # target on, or origin at, the filled week
    assert masked.loc[(idx[1], "a")] == plain.loc[(idx[1], "a")] and masked.loc[(idx[3], "a")] == plain.loc[(idx[3], "a")]


def test_hub_extension_extends_the_observed_mask():
    from kbforecast.hub import extend_kb_panel
    from tests.test_hub import NEW_WEEKS, _hub_for
    from tests.synthetic import make_panel
    from dataclasses import replace

    kb = make_panel()
    obs = {"sale": kb.sale.notna(), "jeonse": kb.jeonse.notna()}
    ext, report = extend_kb_panel(replace(kb, observed=obs), _hub_for(kb))
    assert report.applied
    for name, frame in (("sale", ext.sale), ("jeonse", ext.jeonse)):
        assert ext.observed[name].shape == frame.shape and ext.observed[name].index.equals(frame.index)
        assert ext.observed[name].iloc[-NEW_WEEKS:].all().all()  # hub weeks are real observations
        assert (ext.observed[name] == frame.notna()).all().all()
    assert ext.swap_target().observed["sale"] is ext.observed["jeonse"]
