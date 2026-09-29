"""Parse the KB weekly time-series workbook into a clean region panel.

Outputs (see `KBPanel`):
  sale / jeonse : wide weekly (Monday) index frames, one column per region key
  sentiment     : {indicator: wide frame indexed by weekly date, one column per sentiment scope}
  hierarchy     : region metadata (level, province, parent, sentiment scope)
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .xlsx_reader import read_workbook_rows

EXCEL_EPOCH = pd.Timestamp("1899-12-30")
WEEKLY = "W-MON"

AGGREGATES = ("전국", "6개광역시", "5개광역시", "수도권", "기타지방")
PROVINCES = (
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "전남광주통합특별시", "(구)광주광역시",
    "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원특별자치도", "충청북도", "충청남도",
    "전북특별자치도", "(구)전라남도", "경상북도", "경상남도", "제주도",
)
SEOUL_GROUPS = ("강북14개구", "강남11개구")
INDICATOR_SHEETS = {
    "5.매수우위": "buyer",
    "6.매매거래활발": "sale_txn",
    "7.전세수급": "jeonse_supply",
    "8.전세거래활발": "jeonse_txn",
}
SENTIMENT_ALIAS = {"제주도": "제주특별자치도"}


@dataclass(frozen=True)
class KBPanel:
    sale: pd.DataFrame
    jeonse: pd.DataFrame
    sentiment: dict[str, pd.DataFrame]
    hierarchy: pd.DataFrame
    fingerprint: str
    warnings: tuple[str, ...]

    @property
    def last_date(self) -> pd.Timestamp:
        return self.sale.index.max()

    def swap_target(self) -> "KBPanel":
        """View of the panel with sale and jeonse swapped, so the same engine can forecast the jeonse index."""
        return KBPanel(self.jeonse, self.sale, self.sentiment, self.hierarchy, self.fingerprint, self.warnings)


def _to_date(text: str) -> pd.Timestamp | None:
    if not text:
        return None
    number = pd.to_numeric(text.replace(",", ""), errors="coerce")
    if pd.notna(number) and float(number) > 1000:
        return (EXCEL_EPOCH + pd.to_timedelta(float(number), unit="D")).normalize()
    parsed = pd.to_datetime(text.replace("/", "-").replace(".", "-"), errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


def _to_float(text: str | None) -> float:
    if text is None or text == "":
        return np.nan
    value = pd.to_numeric(str(text).replace(",", ""), errors="coerce")
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else np.nan


def _monday(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts - pd.Timedelta(days=int(ts.weekday()))).normalize()


def _clean_scope(text: str) -> str:
    """'부산광역시  Busan' -> '부산광역시', '6개광역시 6 Large Cities' -> '6개광역시'."""
    match = re.match(r"^[()가-힣0-9]+", text.strip())
    return match.group(0) if match else text.strip()


def build_hierarchy(names: list[str]) -> pd.DataFrame:
    """Assign every header name a unique key plus level/province/parent, using the workbook's ordering."""
    records: list[dict[str, Any]] = []
    province = None
    group = None
    city = None
    for name in names:
        if name in AGGREGATES:
            province = group = city = None
            level, parent = "agg", None
        elif name in PROVINCES:
            province, group, city = name, None, None
            level, parent = "province", None
        elif name in SEOUL_GROUPS:
            group = name
            level, parent = "group", "서울특별시"
        elif province == "서울특별시":
            level, parent = "gu", group
        elif name.endswith("시") and province not in (None, "서울특별시") and province not in (
            "부산광역시", "대구광역시", "인천광역시", "전남광주통합특별시", "(구)광주광역시", "대전광역시", "울산광역시"
        ):
            city = name
            level, parent = "city", province
        elif name.endswith(("구", "군")):
            level, parent = "gu", city if city and province in ("경기도", "충청북도", "충청남도", "전북특별자치도", "경상북도", "경상남도") else province
        else:
            level, parent = "city", province
        records.append({"name": name, "level": level, "province": province, "parent": parent})
    frame = pd.DataFrame(records)
    counts = frame["name"].value_counts()
    keys = []
    for row in frame.itertuples():
        if counts[row.name] == 1 or row.province == "서울특별시":
            keys.append(row.name)
        else:
            keys.append(f"{row.parent or row.province} {row.name}")
    frame["key"] = keys
    if frame["key"].duplicated().any():
        dup = frame.loc[frame["key"].duplicated(keep=False), "key"].unique()
        raise ValueError(f"지역 키가 중복됩니다: {list(dup)[:5]}")
    return frame


def _sentiment_scope_chain(row: pd.Series, scopes: set[str]) -> str:
    for candidate in (row["name"], row["parent"], row["province"]):
        if candidate is None or (isinstance(candidate, float) and math.isnan(candidate)):
            continue
        candidate = SENTIMENT_ALIAS.get(candidate, candidate)
        if candidate in scopes:
            return candidate
    return "전국"


def _find_header(rows: list[dict[int, str]], limit: int = 12) -> int | None:
    for idx, row in enumerate(rows[:limit]):
        values = set(row.values())
        if "구분" in values and "서울특별시" in values:
            return idx
    return None


def _parse_index_sheet(rows: list[dict[int, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    header_idx = _find_header(rows)
    if header_idx is None:
        raise ValueError("지수 시트에서 헤더 행('구분', '서울특별시')을 찾지 못했습니다.")
    header = rows[header_idx]
    date_col = next(col for col, value in header.items() if value == "구분")
    columns = sorted(col for col in header if col > date_col)
    names = [header[col] for col in columns]
    hierarchy = build_hierarchy(names)
    col_to_key = dict(zip(columns, hierarchy["key"]))
    records: dict[pd.Timestamp, dict[str, float]] = {}
    for row in rows[header_idx + 1 :]:
        date = _to_date(row.get(date_col, ""))
        if date is None:
            continue
        date = _monday(date)
        bucket = records.setdefault(date, {})
        for col, key in col_to_key.items():
            value = _to_float(row.get(col))
            if not np.isnan(value):
                bucket[key] = value
    wide = pd.DataFrame.from_dict(records, orient="index").sort_index()
    wide = wide.reindex(columns=list(hierarchy["key"]))
    wide.index.name = "date"
    full = pd.date_range(wide.index.min(), wide.index.max(), freq=WEEKLY)
    wide = wide.reindex(full)
    wide.index.name = "date"
    return wide, hierarchy


def _fill_short_gaps(wide: pd.DataFrame, limit: int = 2) -> pd.DataFrame:
    """Linear-fill gaps of at most `limit` weeks *inside* each series' observed span (no edge extrapolation)."""
    return wide.interpolate(method="linear", limit=limit, limit_area="inside")


def _parse_indicator_sheet(rows: list[dict[int, str]]) -> pd.DataFrame:
    if len(rows) < 5:
        return pd.DataFrame()
    group_header = rows[1]
    scope_cols = {col: _clean_scope(text) for col, text in group_header.items() if col > 0}
    records: dict[pd.Timestamp, dict[str, float]] = {}
    for row in rows[4:]:
        date = _to_date(row.get(0, ""))
        if date is None:
            continue
        bucket = records.setdefault(_monday(date), {})
        for col, scope in scope_cols.items():
            value = _to_float(row.get(col + 2))  # 3rd column of each 3-col block is the index
            if not np.isnan(value):
                bucket[scope] = value
    frame = pd.DataFrame.from_dict(records, orient="index").sort_index()
    frame.index.name = "date"
    full = pd.date_range(frame.index.min(), frame.index.max(), freq=WEEKLY)
    return frame.reindex(full).interpolate(limit=2, limit_area="inside").rename_axis("date")


def parse_kb_panel(file_bytes: bytes) -> KBPanel:
    fingerprint = hashlib.sha256(file_bytes).hexdigest()
    sheets = read_workbook_rows(file_bytes)
    warnings: list[str] = []
    sale_name = next((s for s in sheets if "매매지수" in s), None)
    jeonse_name = next((s for s in sheets if "전세지수" in s), None)
    if sale_name is None:
        raise ValueError("매매지수 시트를 찾지 못했습니다. KB 주간시계열 XLSX 형식인지 확인해 주세요.")
    sale, hierarchy = _parse_index_sheet(sheets[sale_name])
    if jeonse_name is not None:
        jeonse, jeonse_hierarchy = _parse_index_sheet(sheets[jeonse_name])
        if list(jeonse_hierarchy["key"]) != list(hierarchy["key"]):
            warnings.append("전세지수 시트의 지역 구성이 매매지수와 달라 공통 지역만 사용합니다.")
        jeonse = jeonse.reindex(columns=sale.columns)
    else:
        jeonse = pd.DataFrame(index=sale.index, columns=sale.columns, dtype=float)
        warnings.append("전세지수 시트를 찾지 못했습니다. 전세 관련 피처는 사용되지 않습니다.")
    sale = _fill_short_gaps(sale)
    jeonse = _fill_short_gaps(jeonse)
    empty = [c for c in sale.columns if sale[c].notna().sum() == 0]
    if empty:
        sale, jeonse = sale.drop(columns=empty), jeonse.drop(columns=empty)
        hierarchy = hierarchy.loc[~hierarchy["key"].isin(empty)].reset_index(drop=True)

    sentiment: dict[str, pd.DataFrame] = {}
    for prefix, indicator in INDICATOR_SHEETS.items():
        sheet = next((s for s in sheets if s.startswith(prefix)), None)
        if sheet is None:
            continue
        frame = _parse_indicator_sheet(sheets[sheet])
        if not frame.empty:
            sentiment[indicator] = frame
    if not sentiment:
        warnings.append("매수우위/거래활발/전세수급 시트를 찾지 못해 심리지표 피처를 사용하지 않습니다.")
    scopes = set().union(*[set(f.columns) for f in sentiment.values()]) if sentiment else set()
    hierarchy = hierarchy.copy()
    hierarchy["sentiment_scope"] = hierarchy.apply(lambda r: _sentiment_scope_chain(r, scopes), axis=1)
    return KBPanel(sale, jeonse, sentiment, hierarchy.set_index("key", drop=False), fingerprint, tuple(warnings))
