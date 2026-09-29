"""Reproduce the headline walk-forward validation on a KB weekly workbook.

    python scripts/validate.py path/to/KB_주간시계열.xlsx [--first-origin 2014-01-06]

For every horizon it retrains at every 26-week refit point using only labels that have closed by then, predicts
the next origins, and compares the production blend with a random walk, a 26-week drift extrapolation and a
linear-momentum ridge. Errors are cumulative-return percentage points. For the Seoul series it also reports the
production blend with the Seoul policy-rate overlay (kbforecast/overlay.py) and a Diebold-Mariano test of the
overlay against the un-adjusted blend (squared error, one-sided p).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kbforecast import models as M  # noqa: E402
from kbforecast.engine import blend_model, model_columns  # noqa: E402
from kbforecast.evaluation import WFConfig, baseline_predictions, dm_test, score, walk_forward  # noqa: E402
from kbforecast.features import build_features  # noqa: E402
from kbforecast.kb_panel import parse_kb_panel, seoul_region_keys  # noqa: E402
from kbforecast.overlay import apply_seoul_rate_overlay, load_base_rate, rate_change_weekly  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--first-origin", default="2014-01-06")
    parser.add_argument("--horizons", default="13,26,52")
    args = parser.parse_args()

    kb = parse_kb_panel(args.workbook.read_bytes())
    fs = build_features(kb)
    cols = model_columns(fs)
    seoul = tuple(sorted(seoul_region_keys(kb.hierarchy) & set(kb.sale.columns)))
    dates = fs.log_price.index
    z_rate = rate_change_weekly(load_base_rate(None), dates)

    rows = []
    for h in (int(x) for x in args.horizons.split(",")):
        cfg = WFConfig(horizon=h, first_origin=args.first_origin, eval_step=2, refit_every=26)
        rw, drift = baseline_predictions(fs, h, "rw"), baseline_predictions(fs, h, "drift26")
        linmom = walk_forward(fs, ["r13", "r26", "r52"], M.ridge_model(50.0), cfg)
        blend = walk_forward(fs, cols, blend_model(h), cfg)
        for label, subset in (("Seoul (28 series)", seoul), ("All regions", None)):
            sel = (lambda d: d) if subset is None else (lambda d: d[d.index.get_level_values("region").isin(subset)])
            for name, frame in (("linear momentum", linmom), ("production blend", blend)):
                r = score(sel(frame), name, rw, drift, h, 2, mom_pred=linmom["pred"])
                rows.append({"horizon": h, "set": label, "model": name, "n": r["n"], "MAE_pp": r["MAE_pp"], "skill_vs_drift26": r["skill_vs_drift26"], "skill_vs_linmom": r["skill_vs_linmom"], "DM_p_vs_linmom": r["DM_vs_linmom_p"]})
        adjusted, _ = apply_seoul_rate_overlay(blend, z_rate, h, dates, set(seoul))
        in_seoul = lambda d: d[d.index.get_level_values("region").isin(seoul)]  # noqa: E731
        r = score(in_seoul(adjusted), "blend + Seoul rate overlay", rw, drift, h, 2, mom_pred=linmom["pred"])
        rows.append({"horizon": h, "set": "Seoul (28 series)", "model": r["model"], "n": r["n"], "MAE_pp": r["MAE_pp"], "skill_vs_drift26": r["skill_vs_drift26"], "skill_vs_linmom": r["skill_vs_linmom"], "DM_p_vs_linmom": r["DM_vs_linmom_p"]})
        a, b = in_seoul(adjusted).dropna(subset=["y", "pred"]), in_seoul(blend).dropna(subset=["y", "pred"])
        stat, _ = dm_test((a["pred"] - a["y"]) ** 2, (b["pred"].reindex(a.index) - a["y"]) ** 2, h, 2)
        from scipy import stats as st  # noqa: E402

        print(f"  overlay vs raw blend: MAE {b['pred'].sub(b['y']).abs().mean() * 100:.3f} -> {a['pred'].sub(a['y']).abs().mean() * 100:.3f}, DM t={stat:.2f}, one-sided p={st.norm.cdf(stat):.3f}", flush=True)
        print(f"horizon {h} done", flush=True)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
