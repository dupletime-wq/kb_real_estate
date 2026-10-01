"""Forward (pre-registered) prediction log: freeze what the model said at each origin, score it once the future is observed.

Walk-forward results on one historical workbook are selected-on-the-same-data evidence (about two dozen candidate features were
tried, a few adopted). The only evidence that is free of that selection is a forecast written down *before* the outcome exists.
Each logging run stores, per origin date and data snapshot:
  <origin>_<data id>.csv   one row per (region, horizon): prediction / interval / baselines, in cumulative log-return units
  <origin>_<data id>.json  metadata: git commit, engine version, data fingerprint, model configuration and its hash, rate-data vintage
A second run on the same origin and the same data is a no-op; a run on the same origin with different (revised) data is a new file,
so point-in-time information is never overwritten. `score_log` evaluates matured rows against prices that were actually observed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .engine import CONFORMAL_LEVELS, ENGINE_VERSION, LONG_HORIZON, PRUNED_FEATURES, RIDGE_ALPHA, EngineFit
from .kb_panel import KBPanel, seoul_region_keys

LOG_COLUMNS = ["origin", "region", "horizon", "pred", "pred_raw", "lo", "hi", "rw", "drift26"]


def git_commit(repo: Path | None = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:  # not a git checkout (e.g. a packaged deployment)
        return "unknown"


def model_config(fit: EngineFit) -> dict:
    """Everything that determines the forecast apart from the data; its hash identifies the frozen candidate."""
    config = {
        "engine_version": ENGINE_VERSION,
        "anchors": list(fit.anchors),
        "feature_columns": list(fit.columns),
        "pruned_features": sorted(PRUNED_FEATURES),
        "ridge_alpha": {str(k): v for k, v in RIDGE_ALPHA.items()},
        "conformal_levels": {str(k): list(v) for k, v in CONFORMAL_LEVELS.items()},
        "long_horizon_ridge_only_from": LONG_HORIZON,
        "settings": fit.settings,
        "overlay": fit.overlay,
    }
    config["hash"] = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return config


def snapshot(fit: EngineFit, regions: set[str] | None = None) -> pd.DataFrame:
    """Live-origin predictions of every anchor horizon for the chosen regions (default: all regions in the fit)."""
    origin = fit.last_date
    rows = []
    for h in fit.anchors:
        df = fit.predictions[h].join(fit.baselines[h])
        try:
            live = df.xs(origin, level="date")
        except KeyError:
            continue
        if regions is not None:
            live = live[live.index.isin(regions)]
        live = live[np.isfinite(live["pred"])]
        raw = live["pred_raw"] if "pred_raw" in live else live["pred"]
        rows.append(
            pd.DataFrame(
                {
                    "origin": origin.date().isoformat(), "region": live.index, "horizon": h, "pred": live["pred"].to_numpy(),
                    "pred_raw": raw.to_numpy(), "lo": live["lo"].to_numpy(), "hi": live["hi"].to_numpy(),
                    "rw": live["rw"].to_numpy(), "drift26": live["drift26"].to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)[LOG_COLUMNS] if rows else pd.DataFrame(columns=LOG_COLUMNS)


def write_log(fit: EngineFit, kb: KBPanel, log_dir: Path, regions: set[str] | None = None, note: str = "") -> tuple[Path | None, str]:
    """Persist the live forecasts. Returns (csv path or None if this origin+data snapshot is already logged, message)."""
    frame = snapshot(fit, regions)
    if frame.empty:
        return None, "기록할 예측이 없습니다."
    data_id = hashlib.sha256(kb.fingerprint.encode()).hexdigest()[:10]
    stem = f"{fit.last_date.date().isoformat()}_{data_id}"
    csv, meta = Path(log_dir) / f"{stem}.csv", Path(log_dir) / f"{stem}.json"
    if csv.exists():
        return None, f"이미 기록됨: {csv.name}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    config = model_config(fit)
    metadata = {
        "origin": fit.last_date.date().isoformat(),
        "logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "data_fingerprint": kb.fingerprint,  # sha256 of the workbook, plus the hub extension date if the hub was used
        "workbook_last_date": str(kb.sale.index.max().date()),
        "rows": int(len(frame)),
        "regions": int(frame["region"].nunique()),
        "model_config": config,
        "note": note,
    }
    frame.to_csv(csv, index=False, float_format="%.6f")
    meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return csv, f"기록 완료: {csv.name} ({len(frame)}행, 모델 설정 {config['hash']})"


def load_log(log_dir: Path) -> pd.DataFrame:
    frames = []
    for csv in sorted(Path(log_dir).glob("*.csv")):
        f = pd.read_csv(csv, parse_dates=["origin"])
        f.insert(0, "log_file", csv.name)
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["log_file", *LOG_COLUMNS])


def score_log(log: pd.DataFrame, kb: KBPanel) -> pd.DataFrame:
    """Attach realised returns to logged forecasts whose target week has arrived; only actually observed prices count.

    A row is scored when the sale index was a real observation both at the origin week and at origin + horizon weeks
    (`kb.observed`; if the panel carries no mask, every non-missing value counts). Errors are cumulative log-return differences.
    """
    if log.empty:
        return log.assign(target_date=pd.NaT, y=np.nan)
    price = np.log(kb.sale)
    seen = kb.observed["sale"] if kb.observed is not None else kb.sale.notna()
    out = log.copy()
    out["target_date"] = out["origin"] + pd.to_timedelta(out["horizon"] * 7, unit="D")
    y = np.full(len(out), np.nan)
    for i, row in enumerate(out.itertuples()):
        if row.region not in price.columns or row.target_date not in price.index or row.origin not in price.index:
            continue
        if not (seen.at[row.origin, row.region] and seen.at[row.target_date, row.region]):
            continue
        y[i] = price.at[row.target_date, row.region] - price.at[row.origin, row.region]
    out["y"] = y
    return out


def summarize(scored: pd.DataFrame, kb: KBPanel) -> pd.DataFrame:
    """Per horizon: MAE of the logged forecast vs the random walk and the 26-week drift, for Seoul series and for all logged regions."""
    done = scored.dropna(subset=["y"])
    if done.empty:
        return pd.DataFrame(columns=["set", "horizon", "origins", "n", "MAE_pp", "raw_MAE_pp", "drift26_MAE_pp", "rw_MAE_pp", "interval_coverage"])
    seoul = seoul_region_keys(kb.hierarchy)
    rows = []
    for label, sub in (("서울", done[done["region"].isin(seoul)]), ("전체", done)):
        for h, g in sub.groupby("horizon"):
            rows.append(
                {
                    "set": label, "horizon": int(h), "origins": int(g["origin"].nunique()), "n": int(len(g)),
                    "MAE_pp": float((g["pred"] - g["y"]).abs().mean() * 100), "raw_MAE_pp": float((g["pred_raw"] - g["y"]).abs().mean() * 100),
                    "drift26_MAE_pp": float((g["drift26"] - g["y"]).abs().mean() * 100), "rw_MAE_pp": float((g["rw"] - g["y"]).abs().mean() * 100),
                    "interval_coverage": float(((g["y"] >= g["lo"]) & (g["y"] <= g["hi"])).mean()),
                }
            )
    return pd.DataFrame(rows)
