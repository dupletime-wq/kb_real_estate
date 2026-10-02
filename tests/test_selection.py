import numpy as np
import pandas as pd

from kbforecast import evalsuite as E
from kbforecast import selection as S

WINDOWS = (("2018-01-01", "2018-12-31"), ("2019-01-07", "2019-12-31"))


def _wides(good_until="2017-12-31", cand_good_pre=True, seed=0, regions=None):
    """Synthetic wide frames: candidate `good` is much better than baseline before `good_until` and much worse afterwards;
    candidate `steady` is mildly better everywhere; `noise` is just worse."""
    rng = np.random.default_rng(seed)
    regions = regions or ["서울특별시", "강남구", "노원구", "부산광역시", "대구광역시", "경기도 수원시"]
    wides = {}
    for h in (13, 26, 52):
        dates = pd.date_range("2012-01-02", "2019-12-30", freq="2W-MON")
        idx = pd.MultiIndex.from_product([dates, regions], names=["date", "region"])
        y = rng.normal(0.03, 0.04, len(idx))
        base = y + rng.normal(0, 0.03, len(idx))
        early = idx.get_level_values("date") <= pd.Timestamp(good_until)
        good = np.where(early, y + rng.normal(0, 0.012, len(idx)), y + rng.normal(0, 0.09, len(idx)))
        steady = y + rng.normal(0, 0.027, len(idx))
        noise = y + rng.normal(0, 0.05, len(idx))
        wides[h] = pd.DataFrame({"y": y, "baseline": base, "good": good, "steady": steady, "noise": noise}, index=idx)
    return wides


def test_selection_uses_only_origins_that_closed_before_each_window_and_never_the_window_itself():
    wides = _wides()
    hist, ext = S.run_selection(wides, "baseline", ["good", "steady", "noise"], WINDOWS, None, min_rows=20)
    # rewrite everything from the start of the first window onwards: the choice for window 1 must not change
    changed = {h: w.copy() for h, w in wides.items()}
    for h, w in changed.items():
        late = w.index.get_level_values("date") >= pd.Timestamp("2018-01-01") - pd.Timedelta(weeks=h) + pd.Timedelta(days=1)
        w.loc[late, "good"] = w.loc[late, "y"] + 5.0  # catastrophic after the labels stopped being closed
        w.loc[late, "noise"] = w.loc[late, "y"]  # perfect
    hist2, _ = S.run_selection(changed, "baseline", ["good", "steady", "noise"], WINDOWS[:1], None, min_rows=20)
    assert hist2["chosen"].iloc[0] == hist["chosen"].iloc[0] == "good"
    assert "good" in hist["passing_internal"].iloc[0]


def test_a_candidate_that_was_great_inside_the_selection_period_is_graded_on_the_next_window():
    wides = _wides()
    hist, ext = S.run_selection(wides, "baseline", ["good", "steady", "noise"], WINDOWS, None, min_rows=20)
    assert hist["chosen"].iloc[0] == "good"  # chosen on 2012..2017 where it was far better
    rep = S.external_report(ext, n_boot=200)
    g = rep["per_horizon"][26]["groups"]["서울 28"]
    assert g["rel_MAE_pct"] > 0  # frozen choice made things worse in the window: the external grade shows it
    assert rep["decision"]["verdict"] == "기각" and "탐색적" in rep["decision"]["evidence"]
    assert set(ext[26]["chosen"]) <= {"good", "baseline", "steady"} and len(ext[26]) > 0


def test_nothing_passing_means_baseline_and_the_internal_rule_is_applied_to_outside_seoul_too():
    wides = _wides()
    for w in wides.values():
        w["steady"] = w["baseline"] + 0.0  # no gain at all
        w["good"] = w["baseline"] + 0.0
        w["noise"] = w["baseline"] + 0.0
    hist, ext = S.run_selection(wides, "baseline", ["good", "steady", "noise"], WINDOWS, None, min_rows=20)
    assert (hist["chosen"] == "baseline").all()
    rep = S.external_report(ext, n_boot=100)
    assert all(abs(r["groups"]["서울 28"]["rel_MAE_pct"]) < 1e-9 for r in rep["per_horizon"].values())
    # better inside Seoul but clearly worse outside: not selected
    w2 = _wides()
    for w in w2.values():
        outside = ~w.index.get_level_values("region").isin(["서울특별시", "강남구", "노원구"])
        w["good"] = np.where(outside, w["y"] + 0.2, w["y"] + 0.001)
        w["steady"] = w["baseline"]
        w["noise"] = w["baseline"]
    h2, _ = S.run_selection(w2, "baseline", ["good", "steady", "noise"], WINDOWS, None, min_rows=20)
    assert (h2["chosen"] == "baseline").all()


def test_combinations_are_built_only_from_candidates_that_passed_in_the_selection_period():
    wides = _wides()
    seen = []

    def combo_fn(passing):
        seen.append(list(passing))
        series = {h: (w[["good", "steady"]].mean(axis=1)) for h, w in wides.items()}
        return "combo_x", series

    hist, ext = S.run_selection(wides, "baseline", ["good", "steady", "noise"], WINDOWS[:1], combo_fn, min_rows=20)
    assert seen and "noise" not in seen[0] and set(seen[0]) == {"good", "steady"} and seen[0][0] == "good"  # ordered best internal score first
    assert hist["chosen"].iloc[0] in {"good", "combo_x"}
