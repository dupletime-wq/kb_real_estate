"""Prediction intervals via online (split) conformal calibration on walk-forward residuals.

At every origin only residuals whose label window has already closed (date + h <= origin) are used, so the
calibration never sees the future. Residuals are normalised by a volatility scale so intervals widen in turbulent
regimes and tighten in calm ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def scale_from_vol(vol_weekly: pd.Series, horizon: int) -> pd.Series:
    """Random-walk style horizon scaling of the trailing weekly log-return volatility."""
    return vol_weekly.clip(lower=1e-4) * np.sqrt(float(horizon))


def conformal_quantiles(
    pred: pd.DataFrame,
    scale: pd.Series,
    horizon: int,
    dates: pd.DatetimeIndex,
    lower_q: float = 0.10,
    upper_q: float = 0.90,
    window_weeks: int | None = 260,
    min_calib: int = 400,
) -> pd.DataFrame:
    """Add columns [lo, hi] (return space) to `pred` (index (date, region), columns y/pred).

    lo/hi = pred + z_quantile * scale, where z = (y - pred) / scale over previously-realised rows.
    """
    df = pred.copy()
    df["scale"] = scale.reindex(df.index).to_numpy()
    df["z"] = (df["y"] - df["pred"]) / df["scale"]
    date_index = df.index.get_level_values("date")
    pos = pd.Series(np.arange(len(dates)), index=dates)
    df_pos = pos.reindex(date_index).to_numpy()
    z = df["z"].to_numpy()
    lo = np.full(len(df), np.nan)
    hi = np.full(len(df), np.nan)
    for origin in np.unique(df_pos[~np.isnan(df_pos)]):
        calib = (df_pos + horizon <= origin) & np.isfinite(z)
        if window_weeks is not None:
            calib &= df_pos > origin - window_weeks - horizon
        zc = z[calib]
        rows = df_pos == origin
        if zc.size < min_calib:
            continue
        lo[rows] = np.quantile(zc, lower_q)
        hi[rows] = np.quantile(zc, upper_q)
    df["lo"] = df["pred"] + lo * df["scale"]
    df["hi"] = df["pred"] + hi * df["scale"]
    return df.drop(columns=["z"])


def adaptive_widen(
    df: pd.DataFrame,
    horizon: int,
    dates: pd.DatetimeIndex,
    target_miss: float = 0.20,
    window_weeks: int = 52,
    gain: float = 3.0,
    bounds: tuple[float, float] = (0.85, 1.6),
    min_rows: int = 300,
) -> pd.DataFrame:
    """Adaptive-conformal style feedback: scale interval half-widths by the recently *realised* miss rate.

    Only intervals whose label window closed by the current origin (date + h <= origin) feed back, so this is causal.
    """
    out = df.copy()
    pos = pd.Series(np.arange(len(dates)), index=dates)
    dpos = pos.reindex(out.index.get_level_values("date")).to_numpy()
    y, lo, hi, pred = out["y"].to_numpy(), out["lo"].to_numpy(), out["hi"].to_numpy(), out["pred"].to_numpy()
    miss = np.where(np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi), ((y < lo) | (y > hi)).astype(float), np.nan)
    new_lo, new_hi = lo.copy(), hi.copy()
    for origin in np.unique(dpos[~np.isnan(dpos)]):
        realised = (dpos + horizon <= origin) & (dpos > origin - window_weeks - horizon) & np.isfinite(miss)
        if realised.sum() < min_rows:
            continue
        factor = float(np.clip(1.0 + gain * (miss[realised].mean() - target_miss), *bounds))
        rows = dpos == origin
        new_lo[rows] = pred[rows] - (pred[rows] - lo[rows]) * factor
        new_hi[rows] = pred[rows] + (hi[rows] - pred[rows]) * factor
    out["lo"], out["hi"] = new_lo, new_hi
    return out


def interval_metrics(df: pd.DataFrame, lower_q: float = 0.10, upper_q: float = 0.90) -> dict:
    d = df.dropna(subset=["y", "lo", "hi"])
    if d.empty:
        return {"n": 0}
    inside = ((d["y"] >= d["lo"]) & (d["y"] <= d["hi"])).mean()
    width = (d["hi"] - d["lo"]).mean() * 100
    below = (d["y"] < d["lo"]).mean()
    above = (d["y"] > d["hi"]).mean()
    # interval (Winkler) score at nominal coverage
    alpha = 1.0 - (upper_q - lower_q)
    winkler = (d["hi"] - d["lo"]) + (2 / alpha) * ((d["lo"] - d["y"]).clip(lower=0) + (d["y"] - d["hi"]).clip(lower=0))
    return {
        "n": int(len(d)),
        "coverage": float(inside),
        "below": float(below),
        "above": float(above),
        "width_pp": float(width),
        "winkler_pp": float(winkler.mean() * 100),
    }


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Winkler interval score of a (1 - alpha) interval (lower is better): width plus 2/alpha times the miss distance.

    Rewards narrow intervals only as long as they still cover; reported next to coverage and mean width so a method
    cannot look good by being wide (coverage) or narrow (width) alone. Units of y (log return).
    """
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    return (hi - lo) + (2.0 / alpha) * np.maximum(lo - y, 0.0) + (2.0 / alpha) * np.maximum(y - hi, 0.0)
