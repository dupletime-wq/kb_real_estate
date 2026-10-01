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
