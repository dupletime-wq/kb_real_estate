import numpy as np
import pandas as pd
import pytest

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


def test_scenario_path_and_adjustment_fade():
    from kbforecast.overlay import scenario_adjustments, scenario_path

    rate = _rate({"2014-01-01": 3.0}, through="2018-02-01")
    path = scenario_path(rate, 3.6, gap_weeks=4)
    assert list(path["value"]) == [3.25, 3.5, 3.6]  # last move shortened to land on the terminal rate
    assert path["date"].is_monotonic_increasing and path["date"].iloc[0] > pd.Timestamp("2018-02-01")
    run = scenario_adjustments(rate, 3.5, 4, {26: -0.02}, {26: 0.05})
    tl = run["timeline"]
    from kbforecast.overlay import SIGMA_BASE

    assert run["peak_signal"] == pytest.approx(0.5 / SIGMA_BASE)  # a 0.5%p rise over 26 weeks, in standardised units
    assert tl["adj_26"].min() == pytest.approx(-0.02 * 0.5 / SIGMA_BASE * 100)
    assert tl["signal"].iloc[-1] == 0.0  # drag fades once the last move is 26+ weeks old
    assert run["summary"][26]["return_at_peak_pct"] < run["summary"][26]["raw_return_pct"]
    # no move needed -> nothing beyond what is already known
    flat = scenario_adjustments(rate, 3.0, 4, {26: -0.02}, {26: 0.05})
    assert flat["path"].empty


def test_rate_signal_averages_base_rate_and_cd_changes():
    from kbforecast.overlay import SIGMA_BASE, SIGMA_CD, rate_signal_weekly

    base = _rate({"2014-01-01": 2.0, "2016-01-04": 2.5}, through="2018-01-01")
    cd_frame = pd.DataFrame({"date": pd.date_range("2014-01-01", "2018-01-01", freq="D")})
    cd_frame["value"] = np.where(cd_frame["date"] >= "2016-01-04", 2.9, 2.1)  # the CD rate moved by 0.8 when the base rate moved by 0.5
    cd = RateSeries(cd_frame, pd.Timestamp("2018-01-01"), "test")
    day = pd.Timestamp("2016-01-11") + pd.Timedelta(weeks=1)
    both = rate_signal_weekly(base, cd, DATES, weeks=26)
    only = rate_signal_weekly(base, None, DATES, weeks=26)
    assert only.loc[day] == pytest.approx(0.5 / SIGMA_BASE)
    assert both.loc[day] == pytest.approx(0.5 * (0.5 / SIGMA_BASE + 0.8 / SIGMA_CD))


def test_scenario_extends_cd_with_base_rate_moves():
    from kbforecast.overlay import SIGMA_BASE, SIGMA_CD, scenario_adjustments

    base = _rate({"2014-01-01": 3.0}, through="2018-02-01")
    cd_frame = pd.DataFrame({"date": pd.date_range("2014-01-01", "2018-02-01", freq="D"), "value": 3.3})
    cd = RateSeries(cd_frame, pd.Timestamp("2018-02-01"), "test")
    run = scenario_adjustments(base, 3.5, 4, {26: -0.02}, {26: 0.05}, cd=cd)
    assert run["peak_signal"] == pytest.approx(0.5 * (0.5 / SIGMA_BASE + 0.5 / SIGMA_CD))  # CD follows the +0.5 base-rate path
