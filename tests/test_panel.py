import numpy as np
import pandas as pd

from kbforecast.kb_panel import _fill_short_gaps


def test_gap_fill_carries_forward_and_never_uses_later_values():
    idx = pd.date_range("2020-01-06", periods=8, freq="W-MON")
    wide = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan, np.nan], "b": [np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0]}, index=idx)
    out = _fill_short_gaps(wide, limit=2)
    assert out["a"].iloc[1] == 1.0 and out["a"].iloc[2] == 1.0  # not interpolated towards the later 4.0
    assert np.isnan(out["a"].iloc[5:]).all()  # the series ended after 5.0: no extrapolation past its last observation
    assert np.isnan(out["b"].iloc[0])  # nor before the first one
    assert out["b"].iloc[2] == 2.0 and out["b"].iloc[3] == 2.0 and np.isnan(out["b"].iloc[4])  # fills at most `limit` weeks


def test_filled_values_depend_only_on_the_past():
    idx = pd.date_range("2020-01-06", periods=10, freq="W-MON")
    rng = np.random.default_rng(0)
    full = pd.DataFrame({"a": rng.normal(size=10).cumsum()}, index=idx)
    full.iloc[[3, 4], 0] = np.nan
    out = _fill_short_gaps(full)
    # every filled cell equals the last observation before it, whatever the later values are
    assert out["a"].iloc[3] == full["a"].iloc[2] and out["a"].iloc[4] == full["a"].iloc[2]
    altered = full.copy()
    altered.iloc[5:, 0] += 100.0  # change the future: the values at the gap must not move
    pd.testing.assert_series_equal(_fill_short_gaps(altered)["a"].iloc[:5], out["a"].iloc[:5])
