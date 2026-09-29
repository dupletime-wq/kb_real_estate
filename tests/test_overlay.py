import numpy as np
import pandas as pd

from kbforecast.kb_panel import seoul_region_keys
from kbforecast.overlay import RateSeries, apply_seoul_rate_overlay, load_base_rate, rate_change_weekly
from tests.synthetic import make_panel

DATES = pd.date_range("2015-01-05", periods=200, freq="W-MON")
SEOUL = {"서울특별시", "강남구"}


def _rate(changes: dict[str, float], through: str = "2019-01-01") -> RateSeries:
    frame = pd.DataFrame({"date": pd.to_datetime(list(changes)), "value": list(changes.values())})
    return RateSeries(frame, pd.Timestamp(through), "test")


def _pred(z: pd.Series, coef: float, horizon: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([DATES, ["서울특별시", "강남구", "부산광역시", "대구광역시"]], names=["date", "region"])
    zz = z.reindex(idx.get_level_values("date")).to_numpy()
    y = coef * np.nan_to_num(zz) + rng.normal(0, 0.01, len(idx))
    return pd.DataFrame({"y": y, "pred": 0.0}, index=idx)


def test_rate_change_has_publication_lag_and_no_lookahead():
    rate = _rate({"2014-01-01": 2.0, "2016-03-07": 2.5})  # hike announced on a Monday
    z = rate_change_weekly(rate, DATES, weeks=4)
    day = pd.Timestamp("2016-03-07")
    assert z.loc[day] == 0.0  # not yet known on the announcement date itself (1-day lag)
    assert z.loc[day + pd.Timedelta(weeks=1)] == 0.5
    assert z.loc[day + pd.Timedelta(weeks=4)] == 0.5
    assert z.loc[day + pd.Timedelta(weeks=5)] == 0.0  # the 4-week window has rolled past the hike
    # dropping later announcements never changes earlier values
    truncated = rate_change_weekly(_rate({"2014-01-01": 2.0}), DATES, weeks=4)
    pd.testing.assert_series_equal(z.loc[:day], truncated.loc[:day])


def test_overlay_slope_is_never_positive_and_only_seoul_moves():
    z = pd.Series(np.sin(np.arange(len(DATES)) / 7.0), index=DATES)
    pred = _pred(z, coef=+0.05, horizon=13)  # the data "want" a positive slope; the constraint must prevent it
    out, slopes = apply_seoul_rate_overlay(pred, z, 13, DATES, SEOUL, min_rows=20)
    assert (slopes <= 0).all()
    assert (out["overlay"] == 0).all()  # zero slope everywhere -> no adjustment
    pred = _pred(z, coef=-0.05, horizon=13)
    out, slopes = apply_seoul_rate_overlay(pred, z, 13, DATES, SEOUL, min_rows=20)
    assert slopes.iloc[-1] < -0.02
    is_seoul = out.index.get_level_values("region").isin(SEOUL)
    assert (out.loc[~is_seoul, "overlay"] == 0).all()
    assert (out.loc[~is_seoul, "pred"] == out.loc[~is_seoul, "pred_raw"]).all()
    assert (out.loc[is_seoul, "overlay"] != 0).any()


def test_overlay_ignores_labels_that_are_not_yet_realised():
    h = 13
    z = pd.Series(np.cos(np.arange(len(DATES)) / 5.0), index=DATES)
    pred = _pred(z, coef=-0.05, horizon=h)
    base, _ = apply_seoul_rate_overlay(pred, z, h, DATES, SEOUL, min_rows=20)
    origin = DATES[120]
    tampered = pred.copy()
    dates = tampered.index.get_level_values("date")
    tampered.loc[dates > origin - pd.Timedelta(weeks=h), "y"] = 5.0  # labels that close after `origin`
    alt, _ = apply_seoul_rate_overlay(tampered, z, h, DATES, SEOUL, min_rows=20)
    a = base.xs(origin, level="date")["overlay"]
    b = alt.xs(origin, level="date")["overlay"]
    pd.testing.assert_series_equal(a, b)


def test_bundled_rate_snapshot_is_sane():
    rate = load_base_rate(None)
    assert rate.frame["date"].is_monotonic_increasing
    assert rate.known_through >= pd.Timestamp("2026-01-01")
    assert rate.frame["value"].between(0, 10).all()
    z = rate_change_weekly(rate, pd.date_range("2021-06-07", "2023-06-05", freq="W-MON"))
    assert z.max() >= 1.5  # the 2021-22 tightening cycle is visible


def test_seoul_region_keys_cover_city_groups_and_districts():
    kb = make_panel()
    assert seoul_region_keys(kb.hierarchy) == {"서울특별시", "강북14개구", "강남11개구", "강북구", "노원구", "강남구", "서초구"}
