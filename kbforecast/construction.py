"""Construction-cost proxy (national) from ECOS producer prices of 8 construction inputs plus the construction-industry hourly wage index.

Data: kbforecast/data/construction_cost_ecos.csv.gz (scripts/fetch_construction_cost.py). Items were fixed before any result was seen; the materials composite is the
equal-weight geometric mean of the 8 producer price indices (2020=100). ECOS has no regional construction-cost series, so the proxy is one national time series: a
version for Seoul only (the other regions left empty) is the closest available test of "Seoul construction cost".
  cc_mat_r12   12-month log change of the materials composite
  cc_mat_r36   36-month log change of the materials composite
  cc_wage_r4   4-quarter log change of the construction hourly wage index
Publication lags (assumptions): producer prices 45 days after the month's end, the quarterly wage index 100 days after the quarter's end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .regional import _as_of_weekly

COST_FILE = Path(__file__).parent / "data" / "construction_cost_ecos.csv.gz"
PPI_LAG_DAYS = 45
WAGE_LAG_DAYS = 100


def load_cost(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or COST_FILE, dtype={"period": str})


def composite(cost: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(monthly materials composite level, quarterly wage index), both indexed by period end."""
    ppi = cost[cost["series"].str.startswith("ppi_")].pivot_table(index="period", columns="series", values="value", aggfunc="first")
    ppi.index = pd.PeriodIndex(ppi.index, freq="M").to_timestamp(how="end").normalize()
    level = np.exp(np.log(ppi.where(ppi > 0)).mean(axis=1, skipna=False))
    wage = cost[cost["series"] == "wage_construction"].set_index("period")["value"]
    wage.index = pd.PeriodIndex(wage.index, freq="Q").to_timestamp(how="end").normalize()
    return level.sort_index(), wage.sort_index()


def cost_features(kb, cost: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    from .kb_panel import seoul_region_keys

    level, wage = composite(load_cost() if cost is None else cost)
    lg = np.log(level)
    monthly = pd.DataFrame({"cc_mat_r12": lg - lg.shift(12), "cc_mat_r36": lg - lg.shift(36)})
    quarterly = pd.DataFrame({"cc_wage_r4": np.log(wage) - np.log(wage).shift(4)})
    dates = kb.sale.index
    common = pd.concat([_as_of_weekly(monthly, dates, PPI_LAG_DAYS), _as_of_weekly(quarterly, dates, WAGE_LAG_DAYS)], axis=1)
    seoul = kb.sale.columns.isin(seoul_region_keys(kb.hierarchy))
    out = {}
    for name in ("cc_mat_r12", "cc_mat_r36", "cc_wage_r4"):
        wide = pd.DataFrame(np.repeat(common[name].to_numpy()[:, None], len(kb.sale.columns), axis=1), index=dates, columns=kb.sale.columns)
        out[name] = wide
        only = wide.copy()
        only.loc[:, ~seoul] = np.nan
        out[name + "_seoul"] = only
    return out
