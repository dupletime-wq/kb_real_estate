"""MOLIT apartment sale transactions (실거래가) for the 25 Seoul districts: collection and point-in-time bookkeeping.

The public API (`RTMSDataSvcAptTrade`, data.go.kr) returns, per district and CONTRACT month, every reported sale. There is no report-date
field. What it does carry that changes after the fact: `cdealType == "O"` with `cdealDay` for cancelled (해제) deals, and `rgstDate`
(registration date, filled in weeks to months later; also adds the building number). Months long past contain no cancelled rows at all.
So the data of a contract month keep changing: reports arrive up to ~30 days after the contract (60 before the 2020 change), cancellations
and registration fields arrive later still. A history downloaded today is therefore the *final* picture, not what was known then.

Two collections, kept apart on purpose:
  * snapshots (`trade_snapshots/<fetch date>/`): the last few contract months, fetched on a schedule and stored with their fetch date.
    Real point-in-time information, accumulating from the first run on.
  * history (`trade_history/`): all contract months since 2006, fetched once. Final/revised data. Usable for back-testing only through an
    explicit assumed-lag rule (`as_known_at`); never as if it were what was published at the time.
Only daily counts are stored (per district, contract day: all / cancelled / registered / direct), enough for volume features and small
enough to commit (the container is ephemeral).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from typing import Callable

import numpy as np
import pandas as pd

ENDPOINT = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"
PAGE_ROWS = 1000
SEOUL_GROUPS = ("강북14개구", "강남11개구")  # same names as kb_panel.SEOUL_GROUPS
SEOUL_GU_CODES = {
    "종로구": "11110", "중구": "11140", "용산구": "11170", "성동구": "11200", "광진구": "11215", "동대문구": "11230", "중랑구": "11260",
    "성북구": "11290", "강북구": "11305", "도봉구": "11320", "노원구": "11350", "은평구": "11380", "서대문구": "11410", "마포구": "11440",
    "양천구": "11470", "강서구": "11500", "구로구": "11530", "금천구": "11545", "영등포구": "11560", "동작구": "11590", "관악구": "11620",
    "서초구": "11650", "강남구": "11680", "송파구": "11710", "강동구": "11740",
}
COUNT_COLUMNS = ["sgg_cd", "deal_date", "n_all", "n_cancelled", "n_registered", "n_direct"]
# Reporting deadline after the contract date: 60 days before the 2020 amendment, 30 days after (the exact effective date is not verified here).
# Contracts made before the amendment could still be reported for up to 60 days after it, so the shorter lag is only used from 2020-05-25 on.
ASSUMED_LAG_WEEKS = {"before_2020": 12, "from_2020": 8}
LAG_SWITCH = pd.Timestamp("2020-05-25")  # as-of date from which the shorter lag applies

Getter = Callable[[str, str], str]  # (sgg_cd, deal_ym) page -> raw XML; see `http_getter`


class TradeApiError(RuntimeError):
    pass


def _mask(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


def http_getter(key: str, retries: int = 6, timeout: float = 60.0, pause: float = 0.25) -> Callable[[str, str, int], str]:
    """Returns get(sgg_cd, deal_ym, page) -> XML text, retrying network hiccups with backoff. The key is never printed."""
    import requests

    session = requests.Session()  # keep-alive: far fewer TLS handshakes through the proxy, which is where the resets happen

    def get(sgg_cd: str, deal_ym: str, page: int = 1) -> str:
        nonlocal session
        service_key = key if "%" in key else requests.utils.quote(key, safe="")
        url = f"{ENDPOINT}?serviceKey={service_key}&LAWD_CD={sgg_cd}&DEAL_YMD={deal_ym}&numOfRows={PAGE_ROWS}&pageNo={page}"
        last = ""
        for attempt in range(retries):
            try:
                text = session.get(url, timeout=timeout).text
                if "<resultCode>000</resultCode>" in text or "<resultCode>00</resultCode>" in text:
                    time.sleep(pause)
                    return text
                last = _mask(text[:200], key)
            except Exception as exc:  # noqa: BLE001 - connection resets through the proxy are retried on a fresh session
                last = _mask(type(exc).__name__ + ": " + str(exc)[:120], key)
                session = requests.Session()
            time.sleep(1.5 * (attempt + 1))
        raise TradeApiError(f"{sgg_cd} {deal_ym}: {last}")

    return get


def parse_page(xml_text: str) -> tuple[list[dict[str, str]], int]:
    root = ET.fromstring(xml_text)
    total = int((root.findtext(".//totalCount") or "0").strip() or 0)
    rows = [{child.tag: (child.text or "").strip() for child in item} for item in root.iter("item")]
    return rows, total


def fetch_month(get: Callable[[str, str, int], str], sgg_cd: str, deal_ym: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    page = 1
    while True:
        batch, total = parse_page(get(sgg_cd, deal_ym, page))
        rows.extend(batch)
        if not batch or len(rows) >= total:
            break
        page += 1
    return pd.DataFrame(rows)


def weekly_net_and_gross(history: pd.DataFrame, week_ends: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly contract counts per district code on the given week-ending dates (a week = the 7 days ending on that date).

    net = reported deals minus those marked cancelled, gross = all reported deals. Cancelled rows only exist from 2020 on (older
    months were cleaned of them), so only `net` has the same meaning in every year; `gross` is the sensitivity variant.
    """
    days = pd.to_datetime(history["deal_date"])
    out = []
    for column in (history["n_all"] - history["n_cancelled"], history["n_all"]):
        daily = column.groupby([history["sgg_cd"], days]).sum().unstack(0).fillna(0.0)
        daily = daily.reindex(pd.date_range(min(daily.index.min(), week_ends.min() - pd.Timedelta(days=6)), max(daily.index.max(), week_ends.max()), freq="D"), fill_value=0.0)
        weekly = daily.rolling(7, min_periods=7).sum().reindex(week_ends)
        out.append(weekly)
    return out[0], out[1]


def seoul_weekly(weekly: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Weekly counts keyed like the KB panel: 25 districts, the two halves (sum of members) and the city (sum of all)."""
    by_name = weekly.rename(columns={code: name for name, code in SEOUL_GU_CODES.items()})
    out = by_name.copy()
    gu = hierarchy[(hierarchy["province"] == "서울특별시") & (hierarchy["level"] == "gu")]
    for group in SEOUL_GROUPS:
        members = [k for k in gu.index[gu["parent"] == group] if k in by_name.columns]
        out[group] = by_name[members].sum(axis=1, min_count=len(members)) if members else np.nan
    out["서울특별시"] = by_name.sum(axis=1, min_count=by_name.shape[1])
    return out


def volume_features(weekly: pd.DataFrame, extra_lag_weeks: int = 0) -> dict[str, pd.DataFrame]:
    """Point-in-time features at every KB week from the weekly contract counts (see module docstring)."""
    dates = weekly.index
    vol4 = weekly.rolling(4, min_periods=4).sum()
    base = vol4.rolling(156, min_periods=104).mean()
    lag = np.where(dates >= LAG_SWITCH, ASSUMED_LAG_WEEKS["from_2020"], ASSUMED_LAG_WEEKS["before_2020"]) + extra_lag_weeks
    j = np.arange(len(dates)) - lag  # position of the newest week that counts as known at each as-of date
    ok = j >= 13
    jj = np.where(ok, j, 0)
    jj13 = np.where(ok, j - 13, 0)
    cur = vol4.to_numpy()[jj]
    ratio = np.log((cur + 1.0) / (base.to_numpy()[jj] + 1.0))
    chg = np.log((cur + 1.0) / (vol4.to_numpy()[jj13] + 1.0))
    ratio[~ok], chg[~ok] = np.nan, np.nan
    return {"tv_ratio": pd.DataFrame(ratio, index=dates, columns=weekly.columns), "tv_chg13": pd.DataFrame(chg, index=dates, columns=weekly.columns)}


def daily_counts(raw: pd.DataFrame, sgg_cd: str) -> pd.DataFrame:
    """Per contract day: all reported deals, cancelled ones (해제), ones with a registration date, direct (직거래) deals."""
    if raw.empty:
        return pd.DataFrame(columns=COUNT_COLUMNS)
    day = pd.to_datetime(dict(year=raw["dealYear"].astype(int), month=raw["dealMonth"].astype(int), day=raw["dealDay"].astype(int)))
    frame = pd.DataFrame(
        {
            "deal_date": day.dt.strftime("%Y-%m-%d"),
            "n_all": 1,
            "n_cancelled": (raw.get("cdealType", pd.Series("", index=raw.index)).str.upper() == "O").astype(int),
            "n_registered": raw.get("rgstDate", pd.Series("", index=raw.index)).str.len().gt(0).astype(int),
            "n_direct": (raw.get("dealingGbn", pd.Series("", index=raw.index)) == "직거래").astype(int),
        }
    )
    out = frame.groupby("deal_date", as_index=False).sum()
    out.insert(0, "sgg_cd", sgg_cd)
    return out[COUNT_COLUMNS]


def months_back(today: date, n: int) -> list[str]:
    """The current contract month and the n-1 before it, as YYYYMM."""
    out, y, m = [], today.year, today.month
    for _ in range(n):
        out.append(f"{y}{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out


def collect(
    get: Callable[[str, str, int], str], months: list[str], codes: dict[str, str] | None = None,
    progress: Callable[[str], None] | None = None, retry_pause: float = 30.0,
) -> tuple[pd.DataFrame, list[str]]:
    """(counts, failed 'sgg_ym' keys). A district-month that still fails after the getter's own retries is retried once more in a
    second pass; whatever remains is reported, never silently dropped, and does not abort the run."""
    codes = codes or SEOUL_GU_CODES
    parts: list[pd.DataFrame] = []
    todo = [(name, code, ym) for ym in months for name, code in codes.items()]
    for attempt in range(2):
        failed = []
        for name, code, ym in todo:
            try:
                parts.append(daily_counts(fetch_month(get, code, ym), code))
            except TradeApiError:
                failed.append((name, code, ym))
        if progress:
            progress(f"pass {attempt + 1}: {len(todo) - len(failed)}/{len(todo)} ok")
        todo = failed
        if not todo:
            break
        time.sleep(retry_pause)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COUNT_COLUMNS)
    return out, [f"{code}_{ym}" for _, code, ym in todo]


def write_snapshot(counts: pd.DataFrame, root: Path, months: list[str], commit: str, fetched_at: datetime | None = None, kind: str = "snapshot", failed: list[str] | None = None) -> Path:
    """Store counts under <root>/<fetch date>/ with metadata. Re-running on the same date overwrites that day's file (same-day data are equal in practice)."""
    fetched_at = fetched_at or datetime.now(timezone.utc)
    folder = Path(root) / fetched_at.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "seoul_daily_counts.csv.gz"
    counts.to_csv(path, index=False, compression="gzip")
    meta = {
        "kind": kind, "fetched_at_utc": fetched_at.isoformat(timespec="seconds"), "contract_months": [min(months), max(months)], "n_months": len(months),
        "rows": int(len(counts)), "districts": int(counts["sgg_cd"].nunique()) if len(counts) else 0, "deals": int(counts["n_all"].sum()) if len(counts) else 0,
        "endpoint": "RTMSDataSvcAptTrade", "git_commit": commit, "failed_district_months": failed or [],
        "note": "snapshot = point-in-time (what the API showed on this date)" if kind == "snapshot" else "history = final/revised data fetched on this date; NOT what was known at the time",
    }
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def as_known_at(history: pd.DataFrame, as_of: pd.Timestamp, lag_weeks: dict[str, int] | None = None) -> pd.DataFrame:
    """Assumed-lag reconstruction of what a volume series could have shown at `as_of` from FINAL history: only contract days up to
    as_of minus the assumed reporting+publication lag (12 weeks before 2020-05-25, 8 weeks after; assumptions, not measured). Cancelled
    deals are kept (cancellations were not yet known), see the `n_all` column."""
    lag_weeks = lag_weeks or ASSUMED_LAG_WEEKS
    weeks = lag_weeks["from_2020"] if as_of >= LAG_SWITCH else lag_weeks["before_2020"]
    cutoff = as_of - pd.Timedelta(weeks=weeks)
    days = pd.to_datetime(history["deal_date"])
    return history.loc[days <= cutoff]


def verify_history(history: pd.DataFrame, first_month: str = "2006-01") -> dict:
    """Mapping and coverage checks on the daily history: the 25 Seoul district codes, no other codes, no duplicate (code, day) rows,
    and for every code and contract month some deals (a district-month with none while the rest of Seoul traded is reported)."""
    issues: dict = {}
    codes = set(SEOUL_GU_CODES.values())
    seen = set(history["sgg_cd"].unique())
    issues["codes_missing"] = sorted(codes - seen)
    issues["codes_unexpected"] = sorted(seen - codes)
    issues["duplicate_code_day_rows"] = int(history.duplicated(["sgg_cd", "deal_date"]).sum())
    months = pd.PeriodIndex(pd.to_datetime(history["deal_date"]), freq="M")
    grid = history.groupby([history["sgg_cd"], months])["n_all"].sum().unstack(0).reindex(pd.period_range(first_month, months.max(), freq="M"))
    issues["months_covered"] = f"{grid.index.min()}..{grid.index.max()}"
    empty = grid.isna() | (grid == 0)
    seoul_total = grid.fillna(0).sum(axis=1)
    issues["district_months_without_deals"] = [(c, str(m)) for m in grid.index for c in grid.columns if empty.at[m, c] and seoul_total[m] > 0]
    issues["negative_or_cancelled_gt_all"] = int(((history["n_cancelled"] > history["n_all"]) | (history["n_all"] < 0)).sum())
    return issues


def verify_mapping_against_kb(kb) -> dict:
    """The 25 district names used for the MOLIT codes must be Seoul district series of the KB panel (and nothing else of the same name)."""
    h = kb.hierarchy
    seoul_gu = set(h.index[(h["province"] == "서울특별시") & (h["level"] == "gu")])
    names = set(SEOUL_GU_CODES)
    return {"names_not_in_kb_seoul_gu": sorted(names - seoul_gu), "kb_seoul_gu_without_molit_code": sorted(seoul_gu - names)}
