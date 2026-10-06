"""Province-level housing supply pipeline (permits -> starts -> completions) relative to the housing stock, mapped onto the KB regions.

Data: kbforecast/data/supply_sido_monthly.csv.gz (scripts/fetch_supply.py; Ministry of Land, Infrastructure and Transport, province level only, final / revised figures as
served today) and the resident-registration household counts (regional.py) as the stock proxy. Every KB region inherits the series of its province, so the 25 Seoul
districts share one value (the information is about Seoul as a whole, not about a district).
  sup_cmp12     completions of the last 12 months per 1000 households
  sup_start24   starts of the last 24 months per 1000 households   (the pipeline that completes in the next ~2-3 years)
  sup_permit36  permits of the last 36 months per 1000 households
Publication lag: a month counts as known once its end is SUPPLY_LAG_DAYS (60, an assumption) old. Sums need every month of their window (no partial windows).
Provinces whose series is missing in a month (Gwangju / Jeonnam after their 2026-07 merger) stay empty for windows that include it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .regional import PROVINCE_ALIASES, _as_of_weekly, load_population, normalise_name

SUPPLY_FILE = Path(__file__).parent / "data" / "supply_sido_monthly.csv.gz"
SUPPLY_LAG_DAYS = 60


def load_supply(path: Path | None = None) -> pd.DataFrame:
    d = pd.read_csv(path or SUPPLY_FILE)
    d["ym"] = d["ym"].astype(str).str[:7]
    return d


def monthly_wide(supply: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """{'permits' (monthly, un-cumulated), 'starts', 'completions'}: month-end x province."""
    months = pd.period_range(supply["ym"].min(), supply["ym"].max(), freq="M")
    index = months.to_timestamp(how="end").normalize()
    out = {}
    for kind in ("permits", "starts", "completions"):
        wide = supply.pivot_table(index="ym", columns="sido", values=kind, aggfunc="first").reindex([str(m) for m in months])
        wide.index = index
        if kind == "permits":  # year-to-date cumulative -> monthly
            ytd = wide.copy()
            prev = ytd.shift(1)
            january = ytd.index.month == 1
            monthly = ytd - prev
            monthly[january] = ytd[january]
            wide = monthly
        out[kind] = wide
    return out


def households_by_province(pop: pd.DataFrame) -> pd.DataFrame:
    """Month-end households per province (alias names of PROVINCE_ALIASES) plus '전국'; Gwangju / Jeonnam are empty from the 2026-07 merger on."""
    keys = pop["name"].map(normalise_name)
    p = pop.assign(prov=keys.str[0], rest=keys.str[1])
    rows = p[p["rest"] == ""]
    wide = rows.pivot_table(index="ym", columns="prov", values="households", aggfunc="first")
    wide.index = pd.PeriodIndex(wide.index, freq="M").to_timestamp(how="end").normalize()
    national = wide.drop(columns=[c for c in ("전남광주",) if c in wide.columns]).sum(axis=1, min_count=1)
    if "전남광주" in wide.columns:
        national = national + wide["전남광주"].fillna(0)
    wide["전국"] = national
    return wide


def supply_features(kb, supply: pd.DataFrame | None = None, pop: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    sup = monthly_wide(load_supply() if supply is None else supply)
    hh = households_by_province(load_population() if pop is None else pop)
    cols = sorted(set(sup["completions"].columns) & set(hh.columns))
    hh = hh.reindex(sup["completions"].index)[cols].where(lambda x: x > 0)

    def window(kind: str, months: int) -> pd.DataFrame:
        flow = sup[kind][cols]
        return flow.rolling(months, min_periods=months).sum() / hh * 1000.0

    monthly = {"sup_cmp12": window("completions", 12), "sup_start24": window("starts", 24), "sup_permit36": window("permits", 36)}
    h = kb.hierarchy
    sido_of = {}
    for key, row in h.iterrows():
        if key == "전국":
            sido_of[key] = "전국"
        elif row["level"] in ("province", "city", "gu", "group") and isinstance(row["province"], str):
            sido_of[key] = PROVINCE_ALIASES.get(row["province"], row["province"])
    dates = kb.sale.index
    out = {}
    for name, frame in monthly.items():
        weekly = _as_of_weekly(frame, dates, SUPPLY_LAG_DAYS)
        wide = pd.DataFrame(np.nan, index=dates, columns=kb.sale.columns)
        for key, sido in sido_of.items():
            if sido in weekly.columns and key in wide.columns:
                wide[key] = weekly[sido].to_numpy()
        out[name] = wide
    return out
