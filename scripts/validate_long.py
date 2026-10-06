"""Long-horizon (52/78/104/208-week) re-validation on observed prices only, with the baselines the long horizons deserve.

    python scripts/validate_long.py path/to/KB_주간시계열.xlsx [--horizons 52,78,104,208] [--out-dir validation_runs]
    python scripts/validate_long.py path/to/KB_주간시계열.xlsx --filled      # same, but labels may be built from filled values

What is held fixed (identical rows for every comparison, all causal):
  * labels exist only where the origin price and the price h weeks later were actual observations (`KBPanel.observed`); this is
    applied to model training, the historical-mean baselines, the rate overlay's residual fit and the interval calibration
  * the model is the production one without the Seoul rate overlay (the overlay is off from 78 weeks; it is also evaluated for reference)
  * baselines: no change; the pooled national mean return; each region's own mean shrunk 50/50 towards the pooled mean; the province
    (Seoul = the 28 Seoul series) mean shrunk 50/50 towards the pooled mean. They are refit at the same 26-week refit points on the same
    closed labels as the model (they are walk-forward models), the shrink weight is fixed in advance, not tuned
  * intervals: online conformal on residuals that had closed at each origin, 5%/95% quantiles (a 90% interval), never widened to fit
Reported: MAE / RMSE / mean bias for the Seoul city index, the 25 districts, the two Seoul halves, all 28 Seoul series and all regions;
the same on non-overlapping origins (every h weeks, for every possible starting offset: mean and range; this is NOT an independent-sample
count, neighbouring windows still share the same market regime); coverage, mean width and the 90% interval score.
For the 104-week horizon one extra candidate is evaluated: w * ridge + (1 - w) * shrunk regional mean, with w chosen on an internal,
purged, earlier period and then held fixed on the external period. If it does not help, the current ridge stays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kbforecast.engine import ANCHORS, CONFORMAL_LEVELS, ENGINE_VERSION, RIDGE_ALPHA, blend_model, model_columns  # noqa: E402
from kbforecast.evaluation import PERIODS, WFConfig, baseline_predictions, walk_forward  # noqa: E402
from kbforecast.features import build_features, make_targets  # noqa: E402
from kbforecast.forecastlog import git_commit, library_versions  # noqa: E402
from kbforecast.intervals import conformal_quantiles, interval_score, scale_from_vol  # noqa: E402
from kbforecast.kb_panel import SEOUL_GROUPS, parse_kb_panel, seoul_region_keys  # noqa: E402
from kbforecast.overlay import apply_seoul_rate_overlay, load_base_rate, load_cd91, rate_signal_weekly  # noqa: E402

SHRINK = 0.5  # weight on the regional (own / province) mean in the shrunk baselines; fixed in advance
INTERNAL_END = "2017-12-31"  # combo weight chosen on origins up to here (their 104w labels close by the end of 2019) ...
EXTERNAL_START = "2020-01-06"  # ... and evaluated, held fixed, on origins from here
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def mean_models(group_of: dict[str, str]):
    """Walk-forward 'models' that predict closed-label mean returns (pooled / own region / province), refit on the model's schedule."""

    def pooled(Xtr: pd.DataFrame, ytr: np.ndarray, Xp: pd.DataFrame) -> np.ndarray:
        return np.full(len(Xp), float(np.mean(ytr)))

    def shrunk(by) -> object:
        def fit_predict(Xtr: pd.DataFrame, ytr: np.ndarray, Xp: pd.DataFrame) -> np.ndarray:
            mu = float(np.mean(ytr))
            keys_tr = np.asarray([by(r) for r in Xtr.index.get_level_values("region")])
            means = pd.Series(ytr).groupby(keys_tr).mean()
            own = means.reindex([by(r) for r in Xp.index.get_level_values("region")]).to_numpy()
            return np.where(np.isfinite(own), SHRINK * own + (1.0 - SHRINK) * mu, mu)

        return fit_predict

    return pooled, shrunk(lambda r: r), shrunk(lambda r: group_of.get(r, r))


def zero_model(Xtr: pd.DataFrame, ytr: np.ndarray, Xp: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(Xp))


def metrics(d: pd.DataFrame, pred: str, y: str = "y") -> dict:
    e = (d[pred] - d[y]) * 100.0
    return {"n": int(len(d)), "MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e**2).mean())), "bias": float(e.mean())}


def non_overlapping(d: pd.DataFrame, pred: str, h: int, sets: dict[str, pd.Series]) -> pd.DataFrame:
    """MAE on origins spaced h weeks apart (origins are 2 weeks apart: take every h/2-th), for every start offset."""
    dates = np.sort(d.index.get_level_values("date").unique())
    step = max(h // 2, 1)
    rows = []
    for label, mask in sets.items():
        sub = d[mask.reindex(d.index).to_numpy()]
        maes, counts = [], []
        for k in range(step):
            keep = set(dates[k::step])
            part = sub[sub.index.get_level_values("date").isin(keep)]
            if len(part):
                maes.append(float((part[pred] - part["y"]).abs().mean() * 100.0))
                counts.append(len(keep & set(part.index.get_level_values("date"))))
        if maes:
            rows.append({"set": label, "model": pred, "origins": float(np.mean(counts)), "MAE_mean": float(np.mean(maes)), "MAE_min": float(np.min(maes)), "MAE_max": float(np.max(maes))})
    return pd.DataFrame(rows)


def interval_row(d: pd.DataFrame, lo: str = "lo", hi: str = "hi") -> dict:
    d = d.dropna(subset=["y", lo, hi])
    if d.empty:
        return {"n": 0, "coverage": np.nan, "width": np.nan, "score90": np.nan}
    return {
        "n": int(len(d)), "coverage": float(((d["y"] >= d[lo]) & (d["y"] <= d[hi])).mean()), "width": float((d[hi] - d[lo]).mean() * 100.0),
        "score90": float(interval_score(d["y"], d[lo], d[hi]).mean() * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--horizons", default="52,78,104,208")
    parser.add_argument("--first-origin", default="2014-01-06")
    parser.add_argument("--filled", action="store_true", help="allow labels built from filled values (the earlier, weaker evaluation)")
    parser.add_argument("--out-dir", type=Path, default=Path("validation_runs"))
    parser.add_argument("--combo-horizon", type=int, default=104)
    args = parser.parse_args()

    kb = parse_kb_panel(args.workbook.read_bytes())
    fs = build_features(kb)
    cols = model_columns(fs)
    dates = fs.log_price.index
    seoul_all = seoul_region_keys(kb.hierarchy) & set(kb.sale.columns)
    hier = kb.hierarchy
    seoul_city = {"서울특별시"} & seoul_all
    seoul_groups = set(SEOUL_GROUPS) & seoul_all
    seoul_gu = {k for k in seoul_all if k not in seoul_city and k not in seoul_groups}
    group_of = {k: (p if isinstance(p, str) else k) for k, p in hier["province"].items()}
    pooled_fn, own_fn, province_fn = mean_models(group_of)
    z_rate = rate_signal_weekly(load_base_rate(None), load_cd91(None), dates)
    vol = 0.5 * fs.X["vol52"] + 0.5 * fs.X["vol13"]
    observed = None if args.filled else kb.observed["sale"]
    tag = "filled" if args.filled else "observed"
    run_dir = args.out_dir / f"long_{tag}_{dates[-1]:%Y%m%d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "script": "scripts/validate_long.py", "git_commit": git_commit(Path(__file__).resolve().parents[1]), "versions": library_versions(), "data_fingerprint": kb.fingerprint,
        "last_date": str(dates[-1].date()), "labels": tag, "engine_version": ENGINE_VERSION, "feature_columns": cols, "anchors": list(ANCHORS),
        "ridge_alpha": {str(k): v for k, v in RIDGE_ALPHA.items()}, "refit_every": 26, "eval_step": 2, "first_origin": args.first_origin,
        "baseline_shrink": SHRINK, "interval_levels": {str(k): list(v) for k, v in CONFORMAL_LEVELS.items()}, "interval_window_weeks": 260,
        "combo": {"horizon": args.combo_horizon, "internal_end": INTERNAL_END, "external_start": EXTERNAL_START, "weight_grid": list(WEIGHT_GRID)},
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def idx_set(keys: set[str], index: pd.MultiIndex) -> pd.Series:
        return pd.Series(index.get_level_values("region").isin(keys), index=index)

    all_tables: dict[str, list[pd.DataFrame]] = {k: [] for k in ("point", "nonoverlap", "interval", "period", "filled_effect", "combo")}
    for h in (int(x) for x in args.horizons.split(",")):
        print(f"\n######## horizon {h} weeks ({tag} labels) ########", flush=True)
        y = make_targets(fs.log_price, h, observed)
        cfg = WFConfig(horizon=h, first_origin=args.first_origin, eval_step=2, refit_every=26)
        frames = {
            "model": walk_forward(fs, cols, blend_model(h), cfg, y=y),
            "pooled_mean": walk_forward(fs, ["r1"], pooled_fn, cfg, y=y),
            "own_shrunk": walk_forward(fs, ["r1"], own_fn, cfg, y=y),
            "province_shrunk": walk_forward(fs, ["r1"], province_fn, cfg, y=y),
        }
        overlay_frame, _ = apply_seoul_rate_overlay(frames["model"], z_rate, h, dates, seoul_all)
        frames["model+overlay"] = overlay_frame
        frames["no_change"] = frames["model"].assign(pred=0.0)
        frames["drift26"] = frames["model"].assign(pred=baseline_predictions(fs, h, "drift26").reindex(frames["model"].index).to_numpy())
        if observed is not None:
            filled = walk_forward(fs, cols, blend_model(h), cfg)  # labels may include filled prices (training only; scored on observed rows below)
            frames["model_trained_on_filled_labels"] = filled
        names = list(frames)
        common = frames["model"].dropna(subset=["y", "pred"]).index
        for n in names:
            common = common.intersection(frames[n].dropna(subset=["pred"]).index)
        wide = pd.DataFrame({"y": frames["model"].loc[common, "y"]})
        for n in names:
            wide[n] = frames[n].loc[common, "pred"]
        print(f"evaluation rows (identical for every model): {len(wide)}, origins {wide.index.get_level_values('date').nunique()}, "
              f"{wide.index.get_level_values('date').min().date()} .. {wide.index.get_level_values('date').max().date()}")
        sets = {
            "Seoul city index": idx_set(seoul_city, wide.index), "Seoul 25 districts": idx_set(seoul_gu, wide.index),
            "Seoul 2 halves": idx_set(seoul_groups, wide.index), "Seoul all 28": idx_set(seoul_all, wide.index),
            "All regions": pd.Series(True, index=wide.index),
        }
        point = []
        for label, mask in sets.items():
            sub = wide[mask.to_numpy()]
            if sub.empty:
                continue
            for n in names:
                point.append({"horizon": h, "set": label, "model": n, **metrics(sub, n)})
        point = pd.DataFrame(point)
        all_tables["point"].append(point)
        for label in sets:
            t = point[point["set"] == label].set_index("model")[["n", "MAE", "RMSE", "bias"]]
            if len(t):
                print(f"\n[{label}] MAE / RMSE / mean bias (pred - actual), pp, n={int(t['n'].iloc[0])}")
                print(t[["MAE", "RMSE", "bias"]].round(2).T.to_string())
        # by period (Seoul all 28 and all regions)
        per = []
        for label in ("Seoul all 28", "All regions"):
            sub = wide[sets[label].to_numpy()]
            for pname, (a, b) in PERIODS.items():
                part = sub[(sub.index.get_level_values("date") >= a) & (sub.index.get_level_values("date") <= b)]
                if len(part):
                    per.append({"horizon": h, "set": label, "period": pname, "n": len(part), **{n: float((part[n] - part["y"]).abs().mean() * 100.0) for n in ("model", "pooled_mean", "province_shrunk", "no_change")}})
        per = pd.DataFrame(per)
        all_tables["period"].append(per)
        print("\nMAE by period (pp):")
        print(per.round(2).to_string(index=False))
        # non-overlapping origins
        no = pd.concat([non_overlapping(wide, n, h, {k: sets[k] for k in ("Seoul city index", "Seoul 25 districts", "All regions")}) for n in ("model", "pooled_mean", "province_shrunk", "no_change")])
        no.insert(0, "horizon", h)
        all_tables["nonoverlap"].append(no)
        print("\nnon-overlapping origins (every h weeks; mean and range over start offsets; not an independent-sample count):")
        print(no.round(2).to_string(index=False))
        if observed is not None:
            fe = wide.assign(model_filled=wide["model_trained_on_filled_labels"])
            rows = [{"horizon": h, "set": label, "MAE_model": metrics(fe[m.to_numpy()], "model")["MAE"], "MAE_trained_on_filled": metrics(fe[m.to_numpy()], "model_filled")["MAE"]} for label, m in sets.items() if m.any()]
            all_tables["filled_effect"].append(pd.DataFrame(rows))
            print("\neffect of filled labels in training (both scored on the same observed rows):")
            print(pd.DataFrame(rows).round(2).to_string(index=False))
        # intervals on the model's own (raw, no overlay) residuals, closed labels only
        iv = conformal_quantiles(frames["model"], scale_from_vol(vol, h), h, dates, 0.05, 0.95, window_weeks=260)
        iv_ov = conformal_quantiles(overlay_frame[["y", "pred"]], scale_from_vol(vol, h), h, dates, 0.05, 0.95, window_weeks=260)
        iv = iv.loc[iv.index.isin(common)]
        iv_ov = iv_ov.loc[iv_ov.index.isin(common)]
        irows = []
        for label in ("Seoul city index", "Seoul 25 districts", "Seoul all 28", "All regions"):
            for nm, frame in (("raw model", iv), ("model + overlay", iv_ov)):
                part = frame[sets[label].reindex(frame.index).to_numpy()]
                irows.append({"horizon": h, "set": label, "intervals_from": nm, **interval_row(part)})
        irows = pd.DataFrame(irows)
        all_tables["interval"].append(irows)
        print("\n90% intervals (target coverage 0.90; rows that have a calibrated interval; narrower is not better unless coverage holds):")
        print(irows.round(3).to_string(index=False))
        # combo
        if h == args.combo_horizon:
            internal = wide[wide.index.get_level_values("date") <= INTERNAL_END]
            external = wide[wide.index.get_level_values("date") >= EXTERNAL_START]
            crit = {}
            for w in WEIGHT_GRID:
                d = internal.assign(c=w * internal["model"] + (1 - w) * internal["own_shrunk"])
                crit[w] = metrics(d[sets["Seoul all 28"].reindex(d.index).to_numpy()], "c")["MAE"] + metrics(d, "c")["MAE"]
            w_star = min(crit, key=crit.get)
            config["combo"].update({"internal_criterion_Seoul28_plus_All_MAE": {str(k): v for k, v in crit.items()}, "chosen_weight_on_ridge": w_star, "chosen_before_external_evaluation": True})
            (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n== {h}-week combo: internal origins <= {INTERNAL_END} ({len(internal)} rows), criterion Seoul-28 MAE + all MAE: "
                  + ", ".join(f"w={k}: {v:.2f}" for k, v in crit.items()) + f" -> w* = {w_star} (fixed for the external period)")
            ext = external.assign(combo=w_star * external["model"] + (1 - w_star) * external["own_shrunk"])
            crow = []
            ext_sets = {k: pd.Series(m.reindex(ext.index).to_numpy(), index=ext.index) for k, m in sets.items()}
            for label, mask in ext_sets.items():
                sub = ext[mask.to_numpy()]
                for n in ("model", "combo", "own_shrunk", "province_shrunk", "pooled_mean", "no_change"):
                    crow.append({"horizon": h, "set": label, "model": n, **metrics(sub, n)})
            crow = pd.DataFrame(crow)
            all_tables["combo"].append(crow)
            print(f"external origins {EXTERNAL_START} .. {ext.index.get_level_values('date').max().date()} ({len(ext)} rows):")
            for label in ext_sets:
                t = crow[crow["set"] == label].set_index("model")[["MAE", "RMSE", "bias"]]
                print(f"[{label}]"); print(t.round(2).T.to_string())
            nco = pd.concat([non_overlapping(ext, n, h, {k: ext_sets[k] for k in ("Seoul city index", "Seoul 25 districts", "All regions")}) for n in ("model", "combo", "own_shrunk")])
            print("non-overlapping external origins:"); print(nco.round(2).to_string(index=False))
        # save point-in-time predictions
        out = wide.copy()
        out = out.join(iv[["lo", "hi"]], how="left")
        out.reset_index().to_csv(run_dir / f"predictions_h{h}.csv.gz", index=False, float_format="%.5f", compression="gzip")
    for name, parts in all_tables.items():
        if parts:
            pd.concat(parts).to_csv(run_dir / f"summary_{name}.csv", index=False, float_format="%.4f")
    print(f"\nsaved to {run_dir}")


if __name__ == "__main__":
    main()
