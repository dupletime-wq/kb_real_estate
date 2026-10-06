"""Regional demographic series (resident registration population / households, Ministry of the Interior and Safety) mapped onto the KB regions.

Data: kbforecast/data/pop_sigungu_monthly.csv.gz, written by scripts/fetch_population.py (month-end counts for every si/gun/gu, 2008-01 onwards, as the site
serves them today, i.e. final data). Mapping to KB keys is by (province, city, district) NAME because administrative codes changed (for example the 2026 merged
Jeonnam-Gwangju province): rows are first normalised to old-style province names. A KB region that has no row in a month (a district that was abolished or created,
for example the Bucheon districts after 2016) stays empty for that month. Aggregates: Seoul / provinces use their own row, the Seoul half-city groups the sum of their
districts, 전국 and 수도권 the sum of the province rows; the 5/6 metropolitan-city aggregates and 기타지방 are left empty (their member list is not defined here).
Known comparability breaks (mergers): Changwon 2010-07, Cheongju 2014-07, Sejong 2012-07; growth windows that cross these dates are set to NaN for the affected regions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

POP_FILE = Path(__file__).parent / "data" / "pop_sigungu_monthly.csv.gz"
PROVINCE_ALIASES = {
    "강원특별자치도": "강원", "강원도": "강원", "전북특별자치도": "전북", "전라북도": "전북", "제주특별자치도": "제주", "제주도": "제주", "광주광역시": "광주", "(구)광주광역시": "광주",
    "전라남도": "전남", "(구)전라남도": "전남", "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천", "대전광역시": "대전", "울산광역시": "울산",
    "세종특별자치시": "세종", "경기도": "경기", "충청북도": "충북", "충청남도": "충남", "경상북도": "경북", "경상남도": "경남",
}
GWANGJU_DISTRICTS = {"동구", "서구", "남구", "북구", "광산구"}
MERGERS = {"창원시": pd.Timestamp("2010-07-01"), "청주시": pd.Timestamp("2014-07-01"), "세종특별자치시": pd.Timestamp("2012-07-01")}  # comparability breaks (by city / province key)
METRO_AGG_MEMBERS = {"수도권": ("서울", "인천", "경기")}
POP_LAG_DAYS = 45  # assumed: a month is usable once its end is at least 45 days old (an assumption, not a measured release date)


def normalise_name(name: str) -> tuple[str, str]:
    """('province key', 'rest of the name') of a registration row, with province names normalised across years."""
    tokens = name.split()
    prov, rest = tokens[0], tokens[1:]
    if prov == "전남광주통합특별시":
        if not rest:
            return "전남광주", ""
        prov_norm = "광주" if rest[0] in GWANGJU_DISTRICTS else "전남"
    else:
        prov_norm = PROVINCE_ALIASES.get(prov, prov)
    text = " ".join(rest)
    if prov_norm == "인천" and text == "남구":  # renamed Michuhol-gu in 2018-07
        text = "미추홀구"
    return prov_norm, text


def load_population(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or POP_FILE, dtype={"code": str})


def map_to_kb(pop: pd.DataFrame, hierarchy: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """{'pop': month-end x KB key, 'households': ...}."""
    pop = pop.copy()
    keys = pop["name"].map(normalise_name)
    pop["prov"] = keys.str[0]
    pop["rest"] = keys.str[1]
    months = pd.PeriodIndex(sorted(pop["ym"].unique()), freq="M")
    index = months.to_timestamp(how="end").normalize()
    lookup = {field: pop.pivot_table(index="ym", columns=["prov", "rest"], values=field, aggfunc="first").reindex([str(m) for m in months]) for field in ("pop", "households")}
    h = hierarchy.reset_index(drop=True)
    groups = set(h.loc[h["level"] == "group", "key"])
    out = {}
    for field, wide in lookup.items():
        frame = pd.DataFrame(np.nan, index=index, columns=list(h["key"]))
        values = {}
        for _, row in h.iterrows():
            level, name, province, parent = row["level"], row["name"], row["province"], row["parent"]
            prov = PROVINCE_ALIASES.get(province, province) if isinstance(province, str) else None
            col = None
            if row["key"] == "제주특별자치도":  # listed as a city under the province 제주도 in the KB workbook
                col = ("제주", "")
            elif level == "province":
                col = (prov, "")
            elif level == "city":
                col = (prov, name)
            elif level == "gu":
                col = (prov, name if (parent == province or parent in groups) else f"{parent} {name}")
            if col is not None and col in wide.columns:
                values[row["key"]] = wide[col].to_numpy(dtype=float)
        for key, v in values.items():
            frame[key] = v
        # half-city groups of Seoul: sum of their districts (all districts present or the group stays empty)
        for _, row in h[h["level"] == "group"].iterrows():
            members = [k for k in h.loc[h["parent"] == row["key"], "key"] if k in values]
            total_members = int((h["parent"] == row["key"]).sum())
            if members and len(members) == total_members:
                frame[row["key"]] = frame[members].sum(axis=1, min_count=len(members))
        provinces = {p: wide[(p, "")].to_numpy(dtype=float) for p in {c[0] for c in wide.columns} if (p, "") in wide.columns}
        if "전국" in frame.columns:
            core = [p for p in provinces if p != "전남광주"]
            joint = provinces.get("전남광주")
            total = np.nansum([provinces[p] for p in core if p not in ("광주", "전남")], axis=0) if core else np.nan
            parts = [provinces.get(p) for p in ("광주", "전남")]
            if all(p is not None for p in parts):
                gw = np.where(np.isnan(parts[0]) | np.isnan(parts[1]), joint if joint is not None else np.nan, parts[0] + parts[1])
            else:
                gw = joint if joint is not None else np.zeros(len(index))
            frame["전국"] = total + np.nan_to_num(gw)
        if "수도권" in frame.columns:
            frame["수도권"] = np.nansum([provinces.get(p, np.zeros(len(index))) for p in METRO_AGG_MEMBERS["수도권"]], axis=0)
        out[field] = frame
    return out


def _as_of_weekly(monthly: pd.DataFrame, dates: pd.DatetimeIndex, lag_days: int = POP_LAG_DAYS) -> pd.DataFrame:
    """Latest month usable at each week (month end at least `lag_days` old); never interpolated."""
    cutoff = (dates - pd.Timedelta(days=lag_days)).to_numpy()
    idx = np.searchsorted(monthly.index.to_numpy(), cutoff, side="right") - 1
    v = monthly.to_numpy()
    out = np.full((len(dates), v.shape[1]), np.nan)
    ok = idx >= 0
    out[ok] = v[idx[ok]]
    return pd.DataFrame(out, index=dates, columns=monthly.columns)


def population_features(kb, pop: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """pop_g12 / pop_g36 = 12 / 36-month log change of the population, hh_g12 = 12-month log change of households (all as known at the week)."""
    mapped = map_to_kb(load_population() if pop is None else pop, kb.hierarchy)
    dates = kb.sale.index
    out = {}
    for name, field, lag in (("pop_g12", "pop", 12), ("pop_g36", "pop", 36), ("hh_g12", "households", 12)):
        monthly = mapped[field].where(mapped[field] > 0)
        change = np.log(monthly) - np.log(monthly.shift(lag))
        for key, broke in MERGERS.items():  # windows crossing a merger are not comparable
            if key in change.columns:
                change.loc[(change.index >= broke) & (change.index < broke + pd.DateOffset(months=lag)), key] = np.nan
        for key, broke in MERGERS.items():
            for child in kb.hierarchy.index[kb.hierarchy["parent"] == key] if "parent" in kb.hierarchy else []:
                if child in change.columns:
                    change.loc[(change.index >= broke) & (change.index < broke + pd.DateOffset(months=lag)), child] = np.nan
        out[name] = _as_of_weekly(change, dates).reindex(columns=kb.sale.columns)
    return out
