import numpy as np
import pandas as pd

from kbforecast import outside as O


def _frame():
    idx = pd.MultiIndex.from_product([pd.date_range("2015-01-05", periods=3, freq="W-MON"), ["서울특별시", "강남구", "대구광역시", "청주시"]], names=["date", "region"])
    return pd.DataFrame({"pred": np.arange(12, dtype=float), "mean": 100.0 + np.arange(12)}, index=idx)


def test_seoul_rows_are_unchanged_and_others_are_blended_with_the_fixed_weight():
    f = _frame()
    seoul = {"서울특별시", "강남구"}
    out = O.shrink_outside_seoul(f, "pred", "mean", seoul)
    in_seoul = f.index.get_level_values("region").isin(seoul)
    assert (out[in_seoul] == f.loc[in_seoul, "pred"]).all()
    assert np.allclose(out[~in_seoul], 0.5 * f.loc[~in_seoul, "pred"] + 0.5 * f.loc[~in_seoul, "mean"])
    assert O.WEIGHT == 0.5


def _res(rel_out, hi, rel_all, periods):
    return {"groups": {"서울 외": {"rel_MAE_pct": rel_out, "boot_abs": {"rel_hi_pct": hi}, "periods": {k: {"rel_pct": v} for k, v in periods.items()}}, "전체": {"rel_MAE_pct": rel_all}}}


def test_decision_rule_is_the_one_written_in_the_module():
    ok = {104: _res(-2.0, -0.5, -1.0, {"2014-2019": -1.0}), 208: _res(-8.0, -3.0, -4.0, {"2014-2019": -5.0})}
    assert O.decide_outside(ok)["verdict"] == "채택"
    wide = {104: _res(-2.0, 0.4, -1.0, {"2014-2019": -1.0}), 208: _res(-8.0, -3.0, -4.0, {"2014-2019": -5.0})}
    assert O.decide_outside(wide)["verdict"] == "보류"  # interval includes zero at 104w
    worse = {104: _res(2.0, 3.0, 1.0, {}), 208: _res(-8.0, -3.0, -4.0, {})}
    assert O.decide_outside(worse)["verdict"] == "보류" and O.decide_outside(worse)["score"] < 0
    none = {104: _res(1.0, 2.0, 0.5, {}), 208: _res(0.0, 1.0, 0.1, {})}
    assert O.decide_outside(none)["verdict"] == "기각"
    period = {104: _res(-2.0, -0.5, -1.0, {"2020-2021": 3.0}), 208: _res(-8.0, -3.0, -4.0, {"2020-2021": 2.5})}
    assert O.decide_outside(period)["verdict"] == "보류"
    assert O.decide_outside({104: ok[104]})["verdict"] == "기각"  # a missing primary horizon can never adopt
