"""Candidate A: bias correction of the FINAL forecast by the median of residuals that had already closed at each origin.

Order of operations (explicit, so that the rate overlay and this correction never fix the same bias twice):
    pooled model  ->  Seoul rate overlay (anchors <= 52 weeks)  ->  median-residual correction.
The residual of an earlier forecast is `actual - pre-correction final forecast` (overlay already included), so the correction estimates
only what is still left over after the overlay. A forecast made at origin t may use only residuals with `origin_s + h <= t` (their
target week has been observed), and only labels built from observed prices (rows with no label are skipped). Training residuals are not
used. The correction is `shrink * median(residuals)` with 0 < shrink <= 1 (shrinkage towards zero); `none` is always a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_ROWS = {"global": 1000, "seoul": 300}  # closed residuals required before any correction is applied


@dataclass(frozen=True)
class BiasCfg:
    name: str
    scope: str  # "global": every region, residual pool = all regions; "seoul": Seoul rows only, pool = Seoul rows, others untouched
    shrink: float
    window_weeks: int | None = None  # None = expanding; else only residuals whose origin lies within this many weeks of the newest closed one

    @property
    def is_none(self) -> bool:
        return self.shrink == 0.0


BIAS_CANDIDATES = (
    BiasCfg("none", "global", 0.0),
    BiasCfg("global_x0.5", "global", 0.5),
    BiasCfg("global_x1.0", "global", 1.0),
    BiasCfg("seoul_x0.5", "seoul", 0.5),
    BiasCfg("seoul_x1.0", "seoul", 1.0),
    BiasCfg("seoul_recent156_x0.5", "seoul", 0.5, 156),
)


def median_bias_correction(frame: pd.DataFrame, dates: pd.DatetimeIndex, horizon: int, cfg: BiasCfg, seoul_regions: set[str], min_rows: dict[str, int] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return (frame with 'pred_precorr', corrected 'pred' and 'bias_corr', correction per origin date). Rows need columns y, pred."""
    min_rows = min_rows or MIN_ROWS
    out = frame.copy()
    out["pred_precorr"] = out["pred"]
    out["bias_corr"] = 0.0
    if cfg.is_none:
        return out, pd.Series(dtype=float)
    pos = pd.Series(np.arange(len(dates)), index=dates)
    opos = pos.reindex(out.index.get_level_values("date")).to_numpy()
    resid = (out["y"] - out["pred"]).to_numpy(dtype=float)
    is_seoul = np.asarray(out.index.get_level_values("region").isin(seoul_regions))
    pool = np.ones(len(out), bool) if cfg.scope == "global" else is_seoul
    order = np.argsort(opos, kind="stable")
    s_pos, s_res, s_pool = opos[order], resid[order], pool[order]
    corr = np.zeros(len(out))
    per_origin: dict[pd.Timestamp, float] = {}
    for origin in np.unique(opos[~np.isnan(opos)]):
        hi = int(np.searchsorted(s_pos, origin - horizon, side="right"))  # rows with origin_s + h <= origin
        lo = 0 if cfg.window_weeks is None else int(np.searchsorted(s_pos, origin - horizon - cfg.window_weeks, side="right"))
        r = s_res[lo:hi][s_pool[lo:hi]]
        r = r[np.isfinite(r)]
        value = cfg.shrink * float(np.median(r)) if len(r) >= min_rows[cfg.scope] else 0.0
        rows = (opos == origin) & (pool if cfg.scope == "seoul" else True)
        corr[rows] = value
        per_origin[dates[int(origin)]] = value
    out["bias_corr"] = corr
    out["pred"] = out["pred_precorr"] + corr
    return out, pd.Series(per_origin, dtype=float)


def walk_forward_select(
    frame: pd.DataFrame, dates: pd.DatetimeIndex, horizon: int, seoul_regions: set[str], candidates: tuple[BiasCfg, ...] = BIAS_CANDIDATES,
    windows: tuple[tuple[str, str], ...] = (("2020-01-06", "2021-12-31"), ("2022-01-01", "2023-12-31"), ("2024-01-01", "2035-12-31")), min_improvement: float = 0.01,
    min_rows: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the correction for each external window using only origins whose labels closed before the window starts.

    Inside the selection set the Seoul-28 MAE of every candidate is computed on its own causally corrected forecasts; a candidate replaces
    `none` only if it is better by at least `min_improvement` (relative). Returns (frame with the selected 'pred', selection table).
    Origins before the first window keep the uncorrected forecast.
    """
    has_outside = bool((~np.asarray(frame.index.get_level_values("region").isin(seoul_regions))).any())
    usable = [c for c in candidates if c.scope == "seoul" or c.is_none or has_outside]
    skipped = [c.name for c in candidates if c not in usable]
    corrected = {c.name: median_bias_correction(frame, dates, horizon, c, seoul_regions, min_rows)[0] for c in usable}
    pos = pd.Series(np.arange(len(dates)), index=dates)
    opos = pos.reindex(frame.index.get_level_values("date")).to_numpy()
    seoul_mask = np.asarray(frame.index.get_level_values("region").isin(seoul_regions))
    out = frame.copy()
    out["pred_precorr"] = frame["pred"]
    out["bias_corr"] = 0.0
    rows = []
    for start, end in windows:
        s_pos = int(dates.searchsorted(pd.Timestamp(start)))
        sel = (opos + horizon <= s_pos) & np.isfinite(frame["y"].to_numpy()) & seoul_mask
        maes = {name: float(np.abs(c["pred"].to_numpy()[sel] - c["y"].to_numpy()[sel]).mean()) if sel.any() else np.nan for name, c in corrected.items()}
        best, base = "none", maes.get("none", np.nan)
        if np.isfinite(base):
            cand = {k: v for k, v in maes.items() if k != "none" and np.isfinite(v)}
            if cand:
                k = min(cand, key=cand.get)
                if cand[k] <= base * (1 - min_improvement):
                    best = k
        in_window = (frame.index.get_level_values("date") >= pd.Timestamp(start)) & (frame.index.get_level_values("date") <= pd.Timestamp(end))
        out.loc[in_window, "pred"] = corrected[best].loc[in_window, "pred"].to_numpy()
        out.loc[in_window, "bias_corr"] = corrected[best].loc[in_window, "bias_corr"].to_numpy()
        rows.append({"horizon": horizon, "window": f"{start}..{end}", "selection_rows": int(sel.sum()), "chosen": best, "skipped_candidates": ",".join(skipped), **{f"MAE_sel_{k}": v * 100 for k, v in maes.items()}})
    return out, pd.DataFrame(rows)
