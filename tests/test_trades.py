import json

import pandas as pd

from kbforecast.trades import SEOUL_GU_CODES, as_known_at, collect, daily_counts, fetch_month, months_back, parse_page, write_snapshot


def _item(day: int, cancelled: bool = False, rgst: str = "", direct: bool = False, ym=(2025, 8)) -> str:
    return (f"<item><cdealDay>{'25.09.01' if cancelled else ' '}</cdealDay><cdealType>{'O' if cancelled else ' '}</cdealType><dealAmount>63,000</dealAmount>"
            f"<dealDay>{day}</dealDay><dealMonth>{ym[1]}</dealMonth><dealYear>{ym[0]}</dealYear><dealingGbn>{'직거래' if direct else '중개거래'}</dealingGbn>"
            f"<rgstDate>{rgst or ' '}</rgstDate><sggCd>11680</sggCd></item>")


def _page(items: list[str], total: int) -> str:
    return f"<response><header><resultCode>000</resultCode></header><body><items>{''.join(items)}</items><totalCount>{total}</totalCount></body></response>"


def test_parse_and_daily_counts_flag_cancelled_registered_and_direct():
    xml = _page([_item(5), _item(5, cancelled=True), _item(5, rgst="26.04.27"), _item(9, direct=True)], 4)
    rows, total = parse_page(xml)
    assert total == 4 and len(rows) == 4 and rows[1]["cdealType"] == "O"
    out = daily_counts(pd.DataFrame(rows), "11680").set_index("deal_date")
    assert out.loc["2025-08-05"].to_dict() == {"sgg_cd": "11680", "n_all": 3, "n_cancelled": 1, "n_registered": 1, "n_direct": 0}
    assert out.loc["2025-08-09", "n_direct"] == 1


def test_fetch_month_follows_pages_and_collect_covers_every_district():
    calls = []

    def get(sgg, ym, page=1):
        calls.append((sgg, ym, page))
        return _page([_item(1), _item(2)] if page == 1 else [_item(3)], 3)

    df = fetch_month(get, "11680", "202508")
    assert len(df) == 3 and calls == [("11680", "202508", 1), ("11680", "202508", 2)]
    calls.clear()
    counts, failed = collect(get, ["202508", "202507"])
    assert counts["sgg_cd"].nunique() == len(SEOUL_GU_CODES) == 25 and len(calls) == 25 * 2 * 2 and failed == []


def test_months_back_and_snapshot_files(tmp_path):
    from datetime import date, datetime, timezone

    assert months_back(date(2026, 2, 10), 4) == ["202602", "202601", "202512", "202511"]
    counts = pd.DataFrame({"sgg_cd": ["11680"], "deal_date": ["2025-08-05"], "n_all": [3], "n_cancelled": [1], "n_registered": [1], "n_direct": [0]})
    path = write_snapshot(counts, tmp_path, ["202508", "202507"], "abc", datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert path.parent.name == "2026-10-01" and pd.read_csv(path).shape == (1, 6)
    meta = json.loads((path.parent / "meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "snapshot" and meta["contract_months"] == ["202507", "202508"] and "key" not in json.dumps(meta).lower()


def test_assumed_lag_hides_recent_contract_days_and_uses_a_longer_lag_before_2020():
    days = pd.date_range("2019-09-01", "2019-12-31", freq="D").append(pd.date_range("2021-09-01", "2021-12-31", freq="D"))
    hist = pd.DataFrame({"sgg_cd": "11680", "deal_date": days.strftime("%Y-%m-%d"), "n_all": 1})
    old = as_known_at(hist, pd.Timestamp("2019-12-30"))
    new = as_known_at(hist, pd.Timestamp("2021-12-27"))
    assert pd.to_datetime(old["deal_date"]).max() == pd.Timestamp("2019-12-30") - pd.Timedelta(weeks=12)
    assert pd.to_datetime(new["deal_date"]).max() == pd.Timestamp("2021-12-27") - pd.Timedelta(weeks=8)


def test_failed_district_months_are_retried_once_then_reported_not_dropped_silently():
    from kbforecast.trades import TradeApiError

    attempts: dict[tuple[str, str], int] = {}

    def get(sgg, ym, page=1):
        attempts[(sgg, ym)] = attempts.get((sgg, ym), 0) + 1
        if sgg == "11110" and (ym == "202508" or attempts[(sgg, ym)] < 2):  # one flaky, one permanently failing
            raise TradeApiError("boom")
        return _page([_item(1)], 1)

    counts, failed = collect(get, ["202508", "202507"], retry_pause=0.0)
    assert failed == ["11110_202508"]  # reported
    assert attempts[("11110", "202507")] == 2 and "11110" in set(counts["sgg_cd"])  # the flaky one succeeded on the second pass
    assert counts["sgg_cd"].nunique() == 25


def test_weekly_counts_net_vs_gross_and_week_boundaries():
    from kbforecast.trades import weekly_net_and_gross

    hist = pd.DataFrame({"sgg_cd": ["11680"] * 3, "deal_date": ["2024-01-01", "2024-01-03", "2024-01-08"], "n_all": [5, 4, 2], "n_cancelled": [1, 0, 2]})
    ends = pd.DatetimeIndex(["2024-01-07", "2024-01-14"])
    net, gross = weekly_net_and_gross(hist, ends)
    assert gross.loc["2024-01-07", "11680"] == 9 and net.loc["2024-01-07", "11680"] == 8  # days 01-01..01-07
    assert gross.loc["2024-01-14", "11680"] == 2 and net.loc["2024-01-14", "11680"] == 0  # 01-08 only, all cancelled


def test_volume_features_use_only_weeks_old_enough_to_be_reported():
    import numpy as np

    from kbforecast.trades import ASSUMED_LAG_WEEKS, volume_features

    idx = pd.date_range("2015-01-05", "2022-12-26", freq="W-MON")
    rng = np.random.default_rng(1)
    weekly = pd.DataFrame(rng.poisson(30, size=(len(idx), 2)).astype(float), index=idx, columns=["a", "b"])
    base = volume_features(weekly)
    # cutting off (or rewriting) the newest weeks leaves every earlier as-of feature unchanged ...
    cut = len(idx) - 40
    pd.testing.assert_frame_equal(volume_features(weekly.iloc[:cut])["tv_ratio"], base["tv_ratio"].iloc[:cut])
    # ... and an as-of date does not react to weeks newer than the assumed lag (12 weeks before 2020-05-25, 8 after)
    t = idx.get_loc(pd.Timestamp("2018-06-04"))
    changed = weekly.copy()
    changed.iloc[t - ASSUMED_LAG_WEEKS["before_2020"] + 1 : t + 1] *= 5.0  # weeks still inside the reporting window
    after = volume_features(changed)
    assert after["tv_ratio"].iloc[t].equals(base["tv_ratio"].iloc[t])
    changed2 = weekly.copy()
    changed2.iloc[t - ASSUMED_LAG_WEEKS["before_2020"]] *= 5.0  # the newest week that counts as known
    assert not volume_features(changed2)["tv_ratio"].iloc[t].equals(base["tv_ratio"].iloc[t])


def test_verify_history_flags_mapping_and_coverage_problems():
    from kbforecast.trades import verify_history

    rows = []
    for m in pd.period_range("2020-01", "2020-04", freq="M"):
        for name, code in SEOUL_GU_CODES.items():
            if code == "11110" and str(m) == "2020-03":
                continue  # one district-month without any deal while the others traded
            rows.append({"sgg_cd": code, "deal_date": f"{m}-15", "n_all": 3, "n_cancelled": 0})
    ok = verify_history(pd.DataFrame(rows), "2020-01")
    assert ok["codes_missing"] == [] and ok["codes_unexpected"] == [] and ok["duplicate_code_day_rows"] == 0
    assert ok["district_months_without_deals"] == [("11110", "2020-03")]
    bad = pd.DataFrame(rows + [{"sgg_cd": "99999", "deal_date": "2020-01-15", "n_all": 1, "n_cancelled": 2}, rows[0]])
    r = verify_history(bad, "2020-01")
    assert r["codes_unexpected"] == ["99999"] and r["duplicate_code_day_rows"] == 1 and r["negative_or_cancelled_gt_all"] == 1
