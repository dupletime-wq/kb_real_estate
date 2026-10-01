"""Seoul-specific policy-rate overlay on top of the pooled forecast.

Rationale: the pooled model gives every region the same response to shared macro variables, which hides that Seoul
prices are more sensitive to liquidity. A pooled model with the policy rate as an ordinary feature did *not* improve
out-of-sample accuracy, but a separate, single-parameter correction for Seoul does help (mostly in the 2022-23
tightening): it regresses the pooled model's *realised out-of-sample residuals for Seoul* on the 26-week change in the
policy rate, with the slope constrained to be <= 0 (higher rates cannot raise expected returns).

Everything is causal: at each origin the slope only uses residuals whose label window has already closed
(date + horizon <= origin), and the rate change only uses rates announced by that date.

The rate signal is the average of two standardised 26-week changes: the BOK base rate and the 91-day CD rate (a market rate that
moves ahead of the base rate). Evidence level (see README, reproduced by scripts/validate.py): Seoul MAE about -3.3/-4.1/-4.2% at
13/26/52 weeks versus the un-adjusted model (base rate alone: -1.9/-2.9/-4.0%), Diebold-Mariano p ~ 0.10-0.15, i.e. suggestive
rather than statistically established.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_RATE_FILE = Path(__file__).parent / "data" / "base_rate.csv"
CD_FILE = Path(__file__).parent / "data" / "cd91.csv"  # daily 91-day CD rate snapshot
# std of the 26-week change over 2004-2013 (before the first walk-forward origin), used to put the two rates on one scale
SIGMA_BASE = 0.687
SIGMA_CD = 0.775
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


def load_cd91(api_key: str | None = None) -> RateSeries:
    """Daily 91-day CD rate: bundled snapshot, refreshed from ECOS when a key is available."""
    snapshot = pd.read_csv(CD_FILE, parse_dates=["date"])
    if api_key:
        try:
            from .macro import MACRO_SPECS, fetch_series

            daily = fetch_series(MACRO_SPECS["cd91"], api_key)
            if not daily.empty:
                return RateSeries(daily[["date", "value"]].reset_index(drop=True), pd.Timestamp(daily["date"].max()), "ECOS")
        except Exception:
            pass
    return RateSeries(snapshot, pd.Timestamp(snapshot["date"].max()), "저장소 내 스냅샷")


def rate_signal_weekly(base: RateSeries, cd: RateSeries | None, index: pd.DatetimeIndex, weeks: int = CHANGE_WEEKS) -> pd.Series:
    """Seoul rate signal: mean of the standardised 26-week change of the base rate and of the CD rate (base rate alone if no CD)."""
    zb = rate_change_weekly(base, index, weeks) / SIGMA_BASE
    if cd is None:
        return zb
    zc = rate_change_weekly(cd, index, weeks) / SIGMA_CD
    return 0.5 * (zb + zc)


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


def scenario_path(rate: RateSeries, terminal: float, gap_weeks: int = 7, step: float = 0.25) -> pd.DataFrame:
    """Hypothetical future policy-rate moves: `step` per move, one move every `gap_weeks` weeks after the last known date,
    until `terminal` is reached (the last move is shortened if needed). Rows: date, value (level after the move)."""
    frame = rate.frame.sort_values("date")
    level = float(frame["value"].iloc[-1])
    rows = []
    date = pd.Timestamp(rate.known_through)
    while abs(terminal - level) > 1e-9 and len(rows) < 40:
        move = min(step, abs(terminal - level)) * (1.0 if terminal > level else -1.0)
        level = round(level + move, 4)
        date = date + pd.Timedelta(weeks=gap_weeks)
        rows.append({"date": date, "value": level})
    return pd.DataFrame({"date": pd.to_datetime([r["date"] for r in rows]), "value": [r["value"] for r in rows]})


def _weekly_level(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Rate level known on each weekly date (decision/observation published the day before)."""
    frame = frame.sort_values("date")
    lookup = pd.DataFrame({"avail": (frame["date"] + pd.Timedelta(days=RATE_LAG_DAYS)).to_numpy(), "value": frame["value"].to_numpy()})
    target = pd.DataFrame({"avail": pd.DatetimeIndex(index)}).sort_values("avail")
    level = pd.merge_asof(target, lookup, on="avail", direction="backward")
    return pd.Series(level["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(level["avail"])).reindex(index).ffill()


def scenario_adjustments(
    rate: RateSeries,
    terminal: float,
    gap_weeks: int,
    slopes: dict[int, float],
    raw_log_return: dict[int, float],
    cd: RateSeries | None = None,
    tail_weeks: int = 40,
) -> dict:
    """What the Seoul overlay would add over time if the policy rate followed the path to `terminal`.

    The overlay is `slope[h] * signal`, where the signal is the standardised 26-week change of the base rate (averaged with that of
    the CD rate when `cd` is given), so along a hypothetical path we can evaluate it at every later weekly date. In the scenario
    the CD rate is assumed to move one-for-one with the base rate from the last known date on. Returns the path, a weekly timeline
    of the signal and of the adjustment per horizon (%p), and a summary: the peak drag, when it occurs, and when it fades back to zero
    (26 weeks after the last move). `raw_log_return[h]` (the un-adjusted forecast) is only used to show raw + adjustment at the peak:
    this is a sensitivity of the rate term, not a re-forecast of the whole model.

    Because the predictor is a 26-week *change*, the terminal level itself matters only through the pace of moves: a steady
    25bp every ~7 weeks gives about the same 26-week change whether it stops at 3.5%, 3.75% or 4.0%; a higher terminal just
    keeps the drag going for longer.
    """
    path = scenario_path(rate, terminal, gap_weeks)
    frame = rate.frame.sort_values("date")
    frame = frame[frame["date"] < pd.Timestamp(rate.known_through)]  # drop the "known through" marker row
    combined = pd.concat([frame[["date", "value"]], path], ignore_index=True)
    last_move = pd.Timestamp(combined["date"].max())  # last actual (or hypothetical) rate change
    weekly = pd.date_range(frame["date"].min(), max(last_move, pd.Timestamp(rate.known_through)) + pd.Timedelta(weeks=tail_weeks), freq="W-MON")
    base_level = _weekly_level(combined, weekly)
    base_delta = base_level - base_level.shift(CHANGE_WEEKS)
    signal = base_delta / SIGMA_BASE
    if cd is not None:
        known = pd.Timestamp(rate.known_through)
        cd_level = _weekly_level(cd.frame, weekly)
        anchor_date = weekly[weekly <= known][-1]
        shift = base_level - base_level.loc[anchor_date]
        cd_level = cd_level.where(weekly <= anchor_date, cd_level.loc[anchor_date] + shift)  # future CD = last CD + base-rate moves
        signal = 0.5 * (signal + (cd_level - cd_level.shift(CHANGE_WEEKS)) / SIGMA_CD)
    start = weekly[weekly >= pd.Timestamp(rate.known_through)][0]
    timeline = pd.DataFrame({"signal": signal.loc[start:], "base_delta26": base_delta.loc[start:]})
    for h, slope in slopes.items():
        timeline[f"adj_{h}"] = slope * timeline["signal"] * 100.0
    summary = {}
    for h, slope in slopes.items():
        adj = timeline[f"adj_{h}"]
        peak_date = adj.idxmin() if slope < 0 else adj.idxmax()
        peak = float(adj.loc[peak_date])
        summary[h] = {
            "peak_adjustment_pp": peak,
            "peak_date": peak_date,
            "raw_return_pct": (np.exp(raw_log_return[h]) - 1.0) * 100.0,
            "return_at_peak_pct": (np.exp(raw_log_return[h] + peak / 100.0) - 1.0) * 100.0,
        }
    fade = last_move + pd.Timedelta(weeks=CHANGE_WEEKS)
    return {"terminal": terminal, "path": path, "timeline": timeline, "summary": summary, "fade_date": fade,
            "peak_signal": float(timeline["signal"].max()), "peak_base_delta26": float(timeline["base_delta26"].max())}
