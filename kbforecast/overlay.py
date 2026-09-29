"""Seoul-specific policy-rate overlay on top of the pooled forecast.

Rationale: the pooled model gives every region the same response to shared macro variables, which hides that Seoul
prices are more sensitive to liquidity. A pooled model with the policy rate as an ordinary feature did *not* improve
out-of-sample accuracy, but a separate, single-parameter correction for Seoul does help (mostly in the 2022-23
tightening): it regresses the pooled model's *realised out-of-sample residuals for Seoul* on the 26-week change in the
policy rate, with the slope constrained to be <= 0 (higher rates cannot raise expected returns).

Everything is causal: at each origin the slope only uses residuals whose label window has already closed
(date + horizon <= origin), and the rate change only uses rates announced by that date.

Evidence level (see README): MAE -2.4/-2.7/-3.3% and squared error -6/-7/-9% at 13/26/52 weeks on the Seoul series,
one-sided Diebold-Mariano p ~ 0.12, i.e. suggestive rather than statistically established.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_RATE_FILE = Path(__file__).parent / "data" / "base_rate.csv"
RATE_LAG_DAYS = 1  # a Monetary Policy Board decision is public the day it is made
CHANGE_WEEKS = 26
MIN_TRAIN_ROWS = 1500  # ~ one year of weekly origins across the 28 Seoul series


@dataclass(frozen=True)
class RateSeries:
    frame: pd.DataFrame  # columns [date, value]: change points (plus a final "known through" marker row)
    known_through: pd.Timestamp
    source: str


def load_base_rate(api_key: str | None = None) -> RateSeries:
    """Policy-rate history: bundled public snapshot, refreshed from ECOS when a key is available."""
    snapshot = pd.read_csv(BASE_RATE_FILE, parse_dates=["date"])
    if api_key:
        try:
            from .macro import MACRO_SPECS, fetch_series

            daily = fetch_series(MACRO_SPECS["base_rate"], api_key)
            if not daily.empty:
                changes = daily[daily["value"].diff().fillna(1) != 0][["date", "value"]]
                frame = pd.concat([changes, daily.iloc[[-1]][["date", "value"]]]).drop_duplicates("date", keep="first")
                return RateSeries(frame.reset_index(drop=True), pd.Timestamp(daily["date"].max()), "ECOS")
        except Exception:
            pass  # fall back to the bundled snapshot
    return RateSeries(snapshot, pd.Timestamp(snapshot["date"].max()), "저장소 내 스냅샷")


def rate_change_weekly(rate: RateSeries, index: pd.DatetimeIndex, weeks: int = CHANGE_WEEKS) -> pd.Series:
    """Change in the policy rate over the last `weeks` weeks, as known on each weekly date (step function, no look-ahead)."""
    frame = rate.frame.sort_values("date")
    avail = frame["date"] + pd.Timedelta(days=RATE_LAG_DAYS)
    lookup = pd.DataFrame({"avail": avail.to_numpy(), "value": frame["value"].to_numpy()})
    target = pd.DataFrame({"avail": pd.DatetimeIndex(index)}).sort_values("avail")
    level = pd.merge_asof(target, lookup, on="avail", direction="backward")
    level = pd.Series(level["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(level["avail"])).reindex(index).ffill()
    return level - level.shift(weeks)


def apply_seoul_rate_overlay(
    pred: pd.DataFrame,
    z: pd.Series,
    horizon: int,
    dates: pd.DatetimeIndex,
    seoul_regions: set[str],
    min_rows: int = MIN_TRAIN_ROWS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (frame with pred_raw / pred / overlay columns, slope per origin date).

    `pred` has MultiIndex (date, region) and columns y, pred. Only Seoul rows are adjusted; others keep pred == pred_raw.
    """
    out = pred.copy()
    out["pred_raw"] = out["pred"]
    date_index = out.index.get_level_values("date")
    region = out.index.get_level_values("region")
    is_seoul = np.asarray(region.isin(seoul_regions))
    pos = pd.Series(np.arange(len(dates)), index=dates)
    dpos = pos.reindex(date_index).to_numpy()
    zz = z.reindex(date_index).to_numpy(dtype=float)
    resid = (out["y"] - out["pred_raw"]).to_numpy(dtype=float)
    adj = np.zeros(len(out))
    slopes: dict[pd.Timestamp, float] = {}
    for origin in np.unique(dpos[~np.isnan(dpos)]):
        rows = is_seoul & (dpos == origin)
        if not rows.any():
            continue
        train = is_seoul & (dpos + horizon <= origin) & np.isfinite(resid) & np.isfinite(zz)
        slope = 0.0
        if train.sum() >= min_rows:
            x = zz[train]
            denom = float(np.dot(x, x))
            if denom > 0:
                slope = min(float(np.dot(x, resid[train]) / denom), 0.0)  # sign constraint: tighter policy lowers returns
        zo = zz[rows]
        adj[rows] = np.where(np.isfinite(zo), slope * zo, 0.0)
        slopes[dates[int(origin)]] = slope
    out["overlay"] = adj
    out["pred"] = out["pred_raw"] + out["overlay"]
    return out, pd.Series(slopes, dtype=float)
