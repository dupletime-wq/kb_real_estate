"""Extend an uploaded KB weekly workbook with the weeks published after it, from KB 데이터허브's public JSON API.

The workbook is a snapshot; the data hub (data.kbland.kr, the same endpoints its web pages call) usually has a few more
weeks of the same series. Using them moves the forecast origin forward without touching the validated model: the model
is the same, it just starts from a later date (and sees the newer sentiment and the rate decisions made in between).

Safety rules, because this is an unofficial endpoint and a silent mismatch would corrupt forecasts:
  * Regions are matched to hub nodes purely by *values*: a node is used only if it reproduces the workbook's
    overlapping weeks exactly (price indices, to 1e-6) and no other node does. Names and codes are never trusted.
  * Sentiment scopes are matched the same way (the hub rounds to 0.1, so within 0.06).
  * If too few regions match, a sentiment scope is missing or nothing new is published, the workbook is returned unchanged
    together with the reason. Network and parse errors are reported the same way (callers should catch `HubError`).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import time
from typing import Any, Callable
from urllib.parse import quote

import numpy as np
import pandas as pd

from .kb_panel import KBPanel

BASE = "https://data-api.kbland.kr/bfmstat/"
HEADERS = {"osType": "HUB", "Origin": "https://data.kbland.kr", "Referer": "https://data.kbland.kr/", "User-Agent": "Mozilla/5.0"}
PRICE_TOL = 1e-6
SENTIMENT_TOL = 0.06
MIN_PRICE_COVERAGE = 0.95  # share of currently active workbook regions that must be matched and extended
MIN_OVERLAP = 3  # weeks a node must reproduce exactly to be trusted (short-history regions)
STEP = pd.Timedelta(weeks=1)

Getter = Callable[[str, "dict[str, str]"], Any]  # (endpoint, params) -> the response's dataBody.data

SENTIMENT_ENDPOINTS = {  # workbook sentiment key -> (endpoint, 매매전세코드, line field)
    "buyer": ("hrtIndx/trmSppsIndx", "01", "매수우위지수"),
    "sale_txn": ("hrtIndx/trmTranIndx", "01", "거래활발지수"),
    "jeonse_supply": ("hrtIndx/trmSppsIndx", "02", "전세수급지수"),
    "jeonse_txn": ("hrtIndx/trmTranIndx", "02", "거래활발지수"),
}
PRICE_KINDS = {"sale": "01", "jeonse": "02"}
# The hub publishes sentiment for the nation and the provinces but not for the two Seoul halves (nor the pre-merger 광주/전남 or
# the aggregates) the workbook has. Their new weeks are estimated as (last workbook value) + (change of the Seoul series since then). Backtest on the workbook history: mean abs error
# of that estimate 2.5 index points four weeks out (buyer index), versus 9.9 for carrying the last value forward.
PROXY_BASE = {"강북14개구": "서울특별시", "강남11개구": "서울특별시"}  # every other scope the hub lacks falls back to 전국


class HubError(RuntimeError):
    pass


def http_getter(retries: int = 4, timeout: float = 60.0) -> Getter:
    import requests

    def get(endpoint: str, params: dict[str, str]) -> Any:
        query = "&".join(f"{quote(k)}={v}" for k, v in params.items())  # keep commas literal; keys/values are plain
        last: Exception | None = None
        for attempt in range(retries):
            try:
                payload = requests.get(f"{BASE}{endpoint}?{query}", headers=HEADERS, timeout=timeout).json()
                header = payload.get("dataHeader", {})
                if header.get("resultCode") != "10000":
                    raise HubError(f"{endpoint}: {header.get('message')}")
                return payload["dataBody"]["data"]
            except HubError:
                raise
            except Exception as exc:  # noqa: BLE001 - network hiccups are retried, then reported
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise HubError(f"{endpoint}: {last}")

    return get


@dataclass(frozen=True)
class HubData:
    """Raw material fetched from the hub."""

    prices: dict[str, dict[str, pd.Series]]  # 'sale' / 'jeonse' -> node code -> weekly index
    sentiment: dict[str, dict[str, pd.Series]]  # workbook sentiment key -> scope code -> weekly index
    updated: str  # the hub's own 업데이트일자 (YYYYMMDD)


@dataclass(frozen=True)
class HubReport:
    applied: bool
    message: str
    new_dates: tuple[pd.Timestamp, ...] = ()
    matched_regions: int = 0
    total_regions: int = 0
    ended_regions: tuple[str, ...] = field(default_factory=tuple)  # active in the workbook, but their hub series ended


# ----------------------------------------------------------------------------- fetching
def _price_call(get: Getter, kind: str, code: str | None) -> dict:
    params = {"월간주간구분코드": "02", "매매전세코드": kind}
    if code:
        params["지역코드"] = code
    data = get("weekMnthlyHuseTrnd/priceIndex", params)
    if not isinstance(data, dict) or "데이터리스트" not in data or "날짜리스트" not in data:
        raise HubError("priceIndex: unexpected response")
    return data


def _series(values: list, dates: pd.DatetimeIndex) -> pd.Series:
    # the hub appends one extra trailing element (latest weekly change) to some lists; prices align with the date list
    return pd.Series(np.asarray(values[: len(dates)], dtype=float), index=dates)


def fetch_prices(get: Getter, kind: str, descend_names: set[str], workers: int = 6) -> tuple[dict[str, pd.Series], str, list[str]]:
    """Every hub node for one kind (aggregates, provinces, districts). Cities that have districts of their own are
    descended into; their names come from the workbook hierarchy. Returns (nodes, hub update date, top-level codes)."""
    top = _price_call(get, kind, None)
    dates = pd.DatetimeIndex(pd.to_datetime(top["날짜리스트"], format="%Y%m%d"))
    nodes = {item["지역코드"]: _series(item["dataList"], dates) for item in top["데이터리스트"]}
    top_codes = list(nodes)
    level = [c for c in top_codes if c.endswith("00000000") and c != "0000000000"]
    seen = set(level)

    def children(code: str) -> dict:
        return _price_call(get, kind, code)

    while level:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(children, level))
        nxt: list[str] = []
        for code, data in zip(level, results):
            child_dates = pd.DatetimeIndex(pd.to_datetime(data["날짜리스트"], format="%Y%m%d"))
            for item in data["데이터리스트"]:
                child = item["지역코드"]
                if child == code or child in nodes:
                    continue
                nodes[child] = _series(item["dataList"], child_dates)
                if item["지역명"] in descend_names and child not in seen:
                    seen.add(child)
                    nxt.append(child)
        level = nxt
    return nodes, str(top.get("업데이트일자", "")), top_codes


def fetch_sentiment_scope(get: Getter, key: str, code: str) -> pd.Series:
    endpoint, kind, field_name = SENTIMENT_ENDPOINTS[key]
    try:
        data = get(endpoint, {"월간주간구분코드": "02", "매매전세코드": kind, "법정동코드": code})
    except HubError:
        return pd.Series(dtype=float)  # not every aggregate code has a survey; a *needed* scope that is missing is caught when matching
    rows = data.get("line", []) if isinstance(data, dict) else []
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series([r[field_name] for r in rows], index=pd.to_datetime([r["기준날짜"] for r in rows]), dtype=float)


def fetch_hub(hierarchy: pd.DataFrame, get: Getter | None = None, workers: int = 6) -> HubData:
    """Fetch prices (sale + jeonse) and weekly sentiment from the hub. Raises HubError on any failure."""
    get = get or http_getter()
    descend = {p for p in hierarchy.loc[hierarchy["level"] == "gu", "parent"].dropna() if isinstance(p, str)}
    prices: dict[str, dict[str, pd.Series]] = {}
    updated, top_codes = "", []
    for name, kind in PRICE_KINDS.items():
        prices[name], updated, top_codes = fetch_prices(get, kind, descend, workers)
    jobs = [(key, code) for key in SENTIMENT_ENDPOINTS for code in top_codes]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = list(pool.map(lambda job: fetch_sentiment_scope(get, *job), jobs))
    sentiment: dict[str, dict[str, pd.Series]] = {key: {} for key in SENTIMENT_ENDPOINTS}
    for (key, code), series in zip(jobs, fetched):
        if len(series):
            sentiment[key][code] = series
    return HubData(prices, sentiment, updated)


# ----------------------------------------------------------------------------- matching / extension
def _match_by_values(target: pd.DataFrame, nodes: dict[str, pd.Series], tol: float, min_overlap: int, min_overlap_long: int = 20) -> dict[str, str]:
    """workbook column -> node code, only where exactly one node reproduces the column's overlapping values."""
    api = pd.DataFrame(nodes)
    overlap = api.index.intersection(target.index)
    if len(overlap) == 0:
        return {}
    a = api.reindex(overlap).to_numpy(dtype=float)
    codes = list(api.columns)
    matched: dict[str, str] = {}
    for column in target.columns:
        s = target[column].reindex(overlap).to_numpy(dtype=float)
        valid = ~np.isnan(s)
        if valid.sum() < min_overlap:
            continue
        diff = np.abs(a[valid] - s[valid][:, None])
        diff = np.where(np.isnan(diff), np.inf, diff)
        worst = diff.max(axis=0)
        ok = np.flatnonzero(worst < tol)
        if len(ok) == 1:
            matched[column] = codes[int(ok[0])]
    return matched


def _unchanged(kb: KBPanel, why: str) -> tuple[KBPanel, HubReport]:
    return kb, HubReport(False, why)


def extend_kb_panel(kb: KBPanel, hub: HubData) -> tuple[KBPanel, HubReport]:
    """Append the weeks published after the workbook. Returns the panel unchanged (with the reason) if anything looks off."""
    last = kb.sale.index.max()
    active = [c for c in kb.sale.columns if kb.sale[c].iloc[-8:].notna().all()]
    matches: dict[str, dict[str, str]] = {}
    for name, panel in (("sale", kb.sale), ("jeonse", kb.jeonse)):
        matches[name] = _match_by_values(panel, hub.prices[name], PRICE_TOL, MIN_OVERLAP)
    covered = [c for c in active if c in matches["sale"] and c in matches["jeonse"]]
    if not active or len(covered) < MIN_PRICE_COVERAGE * len(active):
        return _unchanged(kb, f"허브 자료와 값이 일치하는 지역이 부족합니다 ({len(covered)}/{len(active)}개)")

    sent_matches: dict[str, dict[str, str]] = {}
    proxied: dict[str, str] = {}
    for key, frame in kb.sentiment.items():
        sent_matches[key] = _match_by_values(frame, hub.sentiment.get(key, {}), SENTIMENT_TOL, MIN_OVERLAP, 20)
        for column in frame.columns:
            if column in sent_matches[key]:
                continue
            base = PROXY_BASE.get(column, "전국")
            if base not in sent_matches[key] and base not in frame.columns:
                return _unchanged(kb, f"심리지표 {key}의 '{column}' 범위를 추정할 기준이 없습니다")
            proxied[column] = base  # estimated below from the base scope's change since the workbook ended

    # candidate new weeks: after the workbook, present for every sentiment scope and (almost) every active region
    def new_slice(series: pd.Series) -> pd.Series:
        return series[series.index > last]

    have = None
    for name, panel in (("sale", kb.sale), ("jeonse", kb.jeonse)):
        counts = pd.Series(0, index=pd.DatetimeIndex([]), dtype=int)
        for region in covered:
            s = new_slice(hub.prices[name][matches[name][region]]).dropna()
            counts = counts.add(pd.Series(1, index=s.index), fill_value=0)
        ok = counts[counts >= MIN_PRICE_COVERAGE * len(covered)].index
        have = ok if have is None else have.intersection(ok)
    for key, frame in kb.sentiment.items():
        for column in frame.columns:
            if column in sent_matches[key]:
                s = new_slice(hub.sentiment[key][sent_matches[key][column]]).dropna()
                have = have.intersection(s.index)
    new_dates: list[pd.Timestamp] = []
    expected = last + STEP
    for d in sorted(have):
        if d == expected:
            new_dates.append(d)
            expected += STEP
        elif d > expected:
            break
    if not new_dates:
        return _unchanged(kb, "워크북 이후에 허브에 새로 공개된 주간 자료가 없습니다")

    def extend(frame: pd.DataFrame, nodes: dict[str, pd.Series], mapping: dict[str, str]) -> pd.DataFrame:
        add = pd.DataFrame(index=pd.DatetimeIndex(new_dates), columns=frame.columns, dtype=float)
        for column, code in mapping.items():
            add[column] = nodes[code].reindex(add.index).to_numpy()
        return pd.concat([frame, add.astype(float)])

    sale = extend(kb.sale, hub.prices["sale"], matches["sale"])
    jeonse = extend(kb.jeonse, hub.prices["jeonse"], matches["jeonse"])
    sentiment = {key: extend(frame, hub.sentiment[key], sent_matches[key]) for key, frame in kb.sentiment.items()}
    for key, frame in sentiment.items():
        for column, base_name in proxied.items():
            if column in frame.columns and column not in sent_matches[key]:
                base = frame[base_name]
                frame.loc[new_dates, column] = frame[column].loc[last] + (base.loc[new_dates] - base.loc[last])
    ended = tuple(c for c in active if sale[c].iloc[-len(new_dates):].isna().any())
    text = f"KB 데이터허브에서 {new_dates[0].date()} ~ {new_dates[-1].date()} {len(new_dates)}주 자료를 추가했습니다 (워크북 마지막 주 {last.date()})"
    if proxied:
        text += "; 허브에 없는 심리지표 범위(" + ", ".join(sorted(proxied)) + ")는 서울/전국 변화폭으로 추정"
    extended = replace(
        kb, sale=sale, jeonse=jeonse, sentiment=sentiment,
        fingerprint=f"{kb.fingerprint}+hub{new_dates[-1]:%Y%m%d}",
    )
    return extended, HubReport(True, text, tuple(new_dates), len(covered), len(active), ended)
