"""Candidate SELECTION validated outside the data it was selected on.

The per-candidate scorecard (a candidate against the baseline over the whole evaluation period) is an exploratory record: the same rows
decide and grade. This module separates the two. For every external window the candidate (or a combination of candidates) is chosen
using ONLY origins whose h-week label had closed before the window starts (`origin + h <= window start`), the choice is frozen, and
the chosen model's forecasts for that window are scored against the baseline. The concatenated external windows grade the selection
PROCEDURE, not any single candidate.

Caveats that stay true: the historical rows were already used by earlier experiments, so the external windows are not untouched data, and
the candidates themselves were designed after looking at these years. The final check is the forward forecast log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evalsuite as E

EXTERNAL_WINDOWS = (("2020-01-06", "2021-12-31"), ("2022-01-01", "2023-12-31"), ("2024-01-01", "2035-12-31"))
MIN_INTERNAL_SEOUL_ROWS = 500
MAX_SINGLE_HORIZON_WORSENING = 2.0  # %, inside the selection period no primary horizon may be worse than this
COMBO_MARGIN = 0.25  # a combination replaces the best single candidate only if its internal score is better by this many points (%)


def _rel(df: pd.DataFrame, base: str, cand: str, mask: np.ndarray) -> tuple[float, int]:
    d = df[mask].dropna(subset=["y", base, cand])
    if d.empty:
        return float("nan"), 0
    mb, mc = float((d[base] - d["y"]).abs().mean()), float((d[cand] - d["y"]).abs().mean())
    return ((mc / mb - 1.0) * 100.0 if mb else float("nan")), int(len(d))


def internal_scores(wides: dict[int, pd.DataFrame], base: str, cands: list[str], start: str, primary: tuple[int, ...] = E.PRIMARY_HORIZONS, min_rows: int = MIN_INTERNAL_SEOUL_ROWS) -> dict[str, dict]:
    """Seoul-28 relative MAE change of each candidate on the selection period of a window, per primary horizon and averaged."""
    out = {}
    for c in cands:
        rels, outs, ns = {}, {}, {}
        ok = True
        for h in primary:
            w = wides.get(h)
            if w is None or c not in w.columns:
                ok = False
                break
            inner = w[w.index.get_level_values("date") <= pd.Timestamp(start) - pd.Timedelta(weeks=h)]  # labels closed before the window starts
            sets = E.region_sets(inner.index.get_level_values("region").unique())
            r, n = _rel(inner, base, c, np.asarray(inner.index.get_level_values("region").isin(sets["서울 28"])))
            if n < min_rows or not np.isfinite(r):
                ok = False
                break
            rels[h], ns[h] = r, n
            if sets["서울 외"]:
                outs[h] = _rel(inner, base, c, np.asarray(inner.index.get_level_values("region").isin(sets["서울 외"])))[0]
        if ok:
            out[c] = {"score": float(np.mean(list(rels.values()))), "rel": rels, "outside": outs, "n_seoul_rows": ns}
    return out


def passes_internal(entry: dict) -> bool:
    """Pre-set internal rule: score at most the adoption threshold, nothing outside Seoul worse than the tolerance, no horizon clearly worse."""
    return bool(
        entry["score"] <= E.ADOPT_MAX_SCORE
        and all(v <= E.OUTSIDE_SEOUL_MAX for v in entry["outside"].values())
        and all(v <= MAX_SINGLE_HORIZON_WORSENING for v in entry["rel"].values())
    )


def run_selection(
    wides: dict[int, pd.DataFrame], base: str, cands: list[str], windows: tuple[tuple[str, str], ...] = EXTERNAL_WINDOWS,
    combo_fn=None, primary: tuple[int, ...] = E.PRIMARY_HORIZONS, extras: tuple[int, ...] = E.EXTRA_HORIZONS, min_rows: int = MIN_INTERNAL_SEOUL_ROWS,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Walk-forward selection. `combo_fn(passing names) -> (combo name, {h: forecasts aligned to wides[h].index})` builds a combination of
    the candidates that passed in this window's selection period; it is only used if it beats the best single candidate there.
    Returns (selection history, {h: external rows with columns y, base, sel, chosen})."""
    wides = {h: w.copy() for h, w in wides.items()}
    history, parts = [], {h: [] for h in list(primary) + list(extras) if h in wides}
    for start, end in windows:
        scores = internal_scores(wides, base, cands, start, primary, min_rows)
        passing = sorted((c for c, e in scores.items() if passes_internal(e)), key=lambda c: scores[c]["score"])
        chosen, reason = base, "no candidate passed the internal rule"
        if passing:
            best = min(passing, key=lambda c: scores[c]["score"])
            chosen, reason = best, f"best passing single candidate (internal score {scores[best]['score']:+.2f}%)"
            if len(passing) >= 2 and combo_fn is not None:
                cname, series = combo_fn(passing)
                for h, s in series.items():
                    if h in wides:
                        wides[h][cname] = s.reindex(wides[h].index)
                combo = internal_scores(wides, base, [cname], start, primary, min_rows).get(cname)
                if combo is not None and passes_internal(combo) and combo["score"] < scores[best]["score"] - COMBO_MARGIN:
                    chosen, reason = cname, f"combination of {passing} beat the best single candidate internally ({combo['score']:+.2f}% vs {scores[best]['score']:+.2f}%)"
                    scores[cname] = combo
        history.append({
            "window": f"{start}..{end}", "chosen": chosen, "reason": reason, "passing_internal": ",".join(passing), "n_candidates_scored": len(scores),
            **{f"internal_score_{c}": e["score"] for c, e in scores.items()},
            **{f"internal_seoul_rows_h{h}": n for h, n in (scores[chosen]["n_seoul_rows"].items() if chosen in scores else [])},
        })
        for h in parts:
            w = wides[h]
            d = w.index.get_level_values("date")
            sel = w[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
            col = base if chosen == base else chosen
            if col not in sel.columns:
                continue
            parts[h].append(pd.DataFrame({"y": sel["y"], "base": sel[base], "sel": sel[col], "chosen": chosen}, index=sel.index))
    ext = {h: pd.concat(p) for h, p in parts.items() if p}
    return pd.DataFrame(history), ext


def external_report(ext: dict[int, pd.DataFrame], n_boot: int = 2000, primary: tuple[int, ...] = E.PRIMARY_HORIZONS, base: str = "baseline") -> dict:
    """Grade the selection procedure on the external windows: the same comparison and the same pre-set rule as for a single candidate."""
    per_h = {}
    for h, f in ext.items():
        groups = {k: v for k, v in E.region_sets(f.index.get_level_values("region").unique()).items() if v}
        per_h[h] = E.compare(f[["y", "base", "sel"]], "base", "sel", h, groups, n_boot)
    return {"per_horizon": per_h, "decision": E.decide({h: r for h, r in per_h.items() if h in primary}, primary=primary), "share_baseline_chosen": {
        h: float((f["chosen"] == base).mean()) for h, f in ext.items()}}
