"""BOK ECOS macro series: fetch, cache, and leakage-safe alignment onto the weekly grid."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import pandas as pd
import requests

ECOS_URL = "https://ecos.bok.or.kr/api/StatisticSearch"


@dataclass(frozen=True)
class MacroSpec:
    name: str
    stat_code: str
    cycle: str  # 'D' daily or 'M' monthly
    items: tuple[str, ...]
    lag_days: int  # days after the reference date until the value is publicly available
    label: str


# lag_days: daily market series are available next day; monthly stats are published with the delays observed
# on the live API (M2/loans/mortgage/unsold lag ~2 months; consumer survey is out within the month; KOSPI ~1 month).
MACRO_SPECS: dict[str, MacroSpec] = {
    s.name: s
    for s in (
        MacroSpec("base_rate", "722Y001", "D", ("0101000",), 1, "한국은행 기준금리"),
        MacroSpec("cd91", "817Y002", "D", ("010502000",), 1, "CD(91일)"),
        MacroSpec("ktb3y", "817Y002", "D", ("010200000",), 1, "국고채(3년)"),
        MacroSpec("ktb10y", "817Y002", "D", ("010210000",), 1, "국고채(10년)"),
        MacroSpec("usdkrw", "731Y003", "D", ("0000003",), 1, "원/달러 환율"),
        MacroSpec("mortgage_rate", "121Y006", "M", ("BECBLA0302",), 65, "주택담보대출 금리(신규취급액)"),
        MacroSpec("housing_loan", "151Y005", "M", ("11100A0",), 65, "주택관련대출(예금취급기관)"),
        MacroSpec("m2", "161Y009", "M", ("BBHS00",), 65, "M2(평잔, 계절조정)"),
        MacroSpec("csi_house_all", "511Y002", "M", ("FMFB", "99988"), 28, "주택가격전망CSI(전국)"),
        MacroSpec("csi_house_seoul", "511Y002", "M", ("FMFB", "F0001"), 28, "주택가격전망CSI(서울)"),
        MacroSpec("csi_rate_all", "511Y002", "M", ("FMBG", "99988"), 28, "금리수준전망CSI"),
        MacroSpec("csi_sentiment", "511Y002", "M", ("FME", "99988"), 28, "소비자심리지수"),
        MacroSpec("unsold_seoul", "901Y074", "M", ("I410B",), 65, "미분양주택(서울)"),
        MacroSpec("unsold_nation", "901Y074", "M", ("I410A",), 65, "미분양주택(전국)"),
        MacroSpec("kospi_m", "902Y002", "M", ("KOR",), 35, "주가지수(한국, 월)"),
    )
}


def ecos_api_key() -> str:
    return os.environ.get("ECOS_API_KEY", "").strip()


def fetch_series(spec: MacroSpec, api_key: str, end: pd.Timestamp | None = None, timeout: int = 60) -> pd.DataFrame:
    """Raw ECOS series: columns [date, value] with *reference-period* dates."""
    end = end or pd.Timestamp.today()
    daily = spec.cycle == "D"
    start_s, end_s = ("20030101", end.strftime("%Y%m%d")) if daily else ("200301", end.strftime("%Y%m"))
    url = "/".join([ECOS_URL, api_key, "json", "kr", "1", "100000", spec.stat_code, spec.cycle, start_s, end_s, *spec.items])
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    block = payload.get("StatisticSearch")
    if block is None:
        raise RuntimeError(payload.get("RESULT", {}).get("MESSAGE", "ECOS 응답을 해석하지 못했습니다."))
    frame = pd.DataFrame(block.get("row", []))
    if frame.empty:
        return pd.DataFrame(columns=["date", "value"])
    frame["date"] = pd.to_datetime(frame["TIME"], format="%Y%m%d" if daily else "%Y%m")
    frame["value"] = pd.to_numeric(frame["DATA_VALUE"], errors="coerce")
    return frame.dropna(subset=["value"]).sort_values("date")[["date", "value"]].reset_index(drop=True)


def load_macro(
    api_key: str | None = None,
    cache_dir: str | Path | None = None,
    names: tuple[str, ...] | None = None,
    refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]:
    """Load all (or selected) macro series. Uses CSV cache when present unless `refresh`; never raises for one bad series."""
    series: dict[str, pd.DataFrame] = {}
    problems: list[str] = []
    cache = Path(cache_dir) if cache_dir else None
    for name, spec in MACRO_SPECS.items():
        if names is not None and name not in names:
            continue
        path = cache / f"{name}.csv" if cache else None
        if path is not None and path.exists() and not refresh:
            frame = pd.read_csv(path, parse_dates=["date"])
            series[name] = frame
            continue
        if not api_key:
            problems.append(f"{spec.label}: 인증키가 없어 건너뜁니다.")
            continue
        try:
            frame = fetch_series(spec, api_key)
        except Exception as exc:
            problems.append(f"{spec.label}: {exc}")
            continue
        if frame.empty:
            problems.append(f"{spec.label}: 데이터가 비어 있습니다.")
            continue
        series[name] = frame
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
    return series, tuple(problems)


def align_weekly(frame: pd.DataFrame, index: pd.DatetimeIndex, lag_days: int) -> pd.Series:
    """As-of join without look-ahead: a value is visible only `lag_days` after its reference date."""
    if frame.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    source = frame.assign(avail=frame["date"] + pd.Timedelta(days=lag_days)).sort_values("avail")[["avail", "value"]]
    target = pd.DataFrame({"avail": pd.DatetimeIndex(index)}).sort_values("avail")
    merged = pd.merge_asof(target, source, on="avail", direction="backward")
    return pd.Series(merged["value"].to_numpy(dtype=float), index=merged["avail"]).reindex(index)


def macro_weekly(macro: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Raw macro levels aligned to the weekly grid (one column per series)."""
    cols = {name: align_weekly(frame, index, MACRO_SPECS[name].lag_days) for name, frame in macro.items() if name in MACRO_SPECS}
    return pd.DataFrame(cols, index=index)
