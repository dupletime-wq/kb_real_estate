import numpy as np
import pandas as pd

from kbforecast.hub import HubData, extend_kb_panel, fetch_prices
from tests.synthetic import make_panel

NEW_WEEKS = 3


def _hub_for(kb, seed: int = 1, drop_new_for: tuple[str, ...] = (), skip_scopes: tuple[str, ...] = ("강북14개구", "강남11개구")) -> HubData:
    """Pretend hub: the workbook's own series (same values) plus a few new weeks, under made-up node codes."""
    rng = np.random.default_rng(seed)
    last = kb.sale.index.max()
    new = pd.date_range(last + pd.Timedelta(weeks=1), periods=NEW_WEEKS, freq="W-MON")
    idx = kb.sale.index[-100:].append(new)

    def grow(frame: pd.DataFrame, noise: float) -> dict[str, pd.Series]:
        out = {}
        for i, column in enumerate(frame.columns):
            hist = frame[column].iloc[-100:]
            ahead = hist.iloc[-1] * np.exp(np.cumsum(rng.normal(0.001, noise, NEW_WEEKS)))
            s = pd.Series(np.concatenate([hist.to_numpy(), ahead]), index=idx)
            if column in drop_new_for:
                s.loc[new] = np.nan
            out[f"N{i:03d}"] = s
        return out

    prices = {"sale": grow(kb.sale, 0.003), "jeonse": grow(kb.jeonse, 0.002)}
    sentiment = {}
    for key, frame in kb.sentiment.items():
        sentiment[key] = {}
        for i, scope in enumerate(frame.columns):
            if scope in skip_scopes:
                continue
            hist = frame[scope].iloc[-100:]
            s = pd.Series(np.concatenate([hist.to_numpy(), hist.iloc[-1] + np.cumsum(rng.normal(0, 1.0, NEW_WEEKS))]), index=idx)
            sentiment[key][f"S{i:03d}"] = s.round(1)  # the hub rounds to 0.1
    return HubData(prices, sentiment, "20990101")


def test_extension_appends_only_new_weeks_and_keeps_history():
    kb = make_panel()
    hub = _hub_for(kb)
    ext, report = extend_kb_panel(kb, hub)
    assert report.applied and len(report.new_dates) == NEW_WEEKS
    assert len(ext.sale) == len(kb.sale) + NEW_WEEKS and ext.sale.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(ext.sale.iloc[: len(kb.sale)], kb.sale, check_freq=False)  # workbook history untouched
    pd.testing.assert_frame_equal(ext.jeonse.iloc[: len(kb.jeonse)], kb.jeonse, check_freq=False)
    assert ext.fingerprint != kb.fingerprint and ext.fingerprint.startswith(kb.fingerprint)
    assert not ext.sale.iloc[-NEW_WEEKS:].isna().any().any()


def test_seoul_halves_missing_at_hub_are_estimated_from_seoul_change():
    kb = make_panel()
    ext, report = extend_kb_panel(kb, _hub_for(kb))
    last = kb.sale.index.max()
    seoul = ext.sentiment["buyer"]["서울특별시"]
    est = ext.sentiment["buyer"]["강남11개구"].iloc[-NEW_WEEKS:]
    expected = kb.sentiment["buyer"]["강남11개구"].loc[last] + (seoul.iloc[-NEW_WEEKS:] - seoul.loc[last])
    pd.testing.assert_series_equal(est, expected, check_names=False)
    assert "강남11개구" in report.message


def test_workbook_returned_unchanged_when_values_do_not_match():
    kb = make_panel()
    hub = _hub_for(kb)
    for code, s in hub.prices["sale"].items():  # corrupt every node's overlap: nothing can be matched
        hub.prices["sale"][code] = s * 1.01
    ext, report = extend_kb_panel(kb, hub)
    assert not report.applied and ext is kb and "일치" in report.message


def test_nothing_new_published_means_no_change():
    kb = make_panel()
    hub = _hub_for(kb)
    cut = kb.sale.index.max()
    hub = HubData(
        {k: {c: s.loc[:cut] for c, s in v.items()} for k, v in hub.prices.items()},
        {k: {c: s.loc[:cut] for c, s in v.items()} for k, v in hub.sentiment.items()}, hub.updated,
    )
    ext, report = extend_kb_panel(kb, hub)
    assert not report.applied and ext is kb


def test_ambiguous_nodes_are_never_used():
    kb = make_panel()
    hub = _hub_for(kb)
    codes = list(hub.prices["sale"])
    hub.prices["sale"]["DUP"] = hub.prices["sale"][codes[0]].copy()  # two nodes reproduce region 0 equally well -> region 0 unmatched
    ext, report = extend_kb_panel(kb, hub)
    assert not report.applied and "10/11" in report.message  # 10 of 11 active regions is below the 95% coverage bar
    assert ext is kb


def test_price_crawl_aligns_prices_with_dates_and_descends_into_named_cities():
    dates = ["20260907", "20260914", "20260921"]

    def response(codes_names):
        return {"업데이트일자": "20260921", "날짜리스트": dates,
                "데이터리스트": [{"지역코드": c, "지역명": n, "dataList": [100.0 + i, 101.0 + i, 102.0 + i, 0.15]} for i, (c, n) in enumerate(codes_names)]}  # trailing weekly change

    tree = {None: [("0000000000", "전국"), ("4100000000", "경기")], "4100000000": [("4111000000", "수원시")], "4111000000": [("4111100000", "장안구")]}

    def get(endpoint, params):
        return response(tree[params.get("지역코드")])

    nodes, updated, top = fetch_prices(get, "01", {"수원시"})
    assert updated == "20260921" and top == ["0000000000", "4100000000"]
    assert set(nodes) == {"0000000000", "4100000000", "4111000000", "4111100000"}
    assert list(nodes["4111100000"]) == [100.0, 101.0, 102.0]  # the trailing 0.15 is not a price
    assert nodes["0000000000"].index[-1] == pd.Timestamp("2026-09-21")
