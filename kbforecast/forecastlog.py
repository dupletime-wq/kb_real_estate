"""Forward (pre-registered) prediction log: freeze what the model said at each origin, score it once the future is observed.

Walk-forward results on one historical workbook are selected-on-the-same-data evidence (about two dozen candidate features were
tried, a few adopted). The only evidence that is free of that selection is a forecast written down *before* the outcome exists.
Each logging run stores, per origin date, data snapshot AND model configuration:
  <origin>_<data id>_<config id>.csv   one row per (region, horizon): prediction / interval / baselines, in cumulative log-return units,
                                       plus the sale index at the origin as the data showed it then (`origin_price`)
  <origin>_<data id>_<config id>.json  metadata: git commit, library versions, engine version, data fingerprint, model configuration
A second run on the same origin, data and configuration is a no-op. A different configuration (a candidate model) on the same origin
and data is a new file next to the current model's, a run on revised data is a new file too, so nothing is overwritten. Records written
before this scheme (`<origin>_<data id>.csv`) are kept as they are and load as config 'legacy-...'.
The configuration id covers everything that determines the forecast apart from the data: features, HGB settings and early-stopping mode,
blend weights, preprocessing, labels, overlay and bias-correction settings, interval levels (never the data-dependent overlay values).
`score_log` scores matured rows against prices that were actually observed, either with today's values of the index (`latest`) or with the
value each week was FIRST SEEN at by this system (`first_seen`, append-only archive `realized_first_seen.csv`, with the collection lag and a late-collection flag; the origin price is the logged one). A value is only called a first PUBLICATION value (`first_published`) if its vintage was verified separately (`vintage_verified.csv`).
`pair_models` joins two configurations on the same origin / region / horizon / data vintage so they are compared on identical rows.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import models as M
from .engine import CONFORMAL_LEVELS, ENGINE_VERSION, HGB_KW, LONG_HORIZON, PRUNED_FEATURES, RIDGE_ALPHA, EngineFit
from .kb_panel import KBPanel, seoul_region_keys

LOG_COLUMNS = ["origin", "region", "horizon", "pred", "pred_raw", "lo", "hi", "rw", "drift26", "origin_price"]


def git_commit(repo: Path | None = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:  # not a git checkout (e.g. a packaged deployment)
        return "unknown"


def library_versions() -> dict:
    import platform

    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "streamlit"):
        try:
            out["scikit_learn" if name == "sklearn" else name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - optional in some environments
            out["scikit_learn" if name == "sklearn" else name] = "unavailable"
    return out


def structural_config(fit: EngineFit) -> dict:
    """Everything that determines the forecast apart from the data. The identifier is derived from this only."""
    st = fit.settings
    variant = st.get("variant") or {"name": "current"}
    return {
        "config_schema": 2,
        "engine_version": ENGINE_VERSION,
        "anchors": list(fit.anchors),
        "feature_columns": list(fit.columns),
        "pruned_features": sorted(PRUNED_FEATURES),
        "ridge_alpha": {str(k): v for k, v in RIDGE_ALPHA.items()},
        "conformal_levels": {str(k): list(v) for k, v in CONFORMAL_LEVELS.items()},
        "long_horizon_ridge_only_from": LONG_HORIZON,
        "hgb": {"overrides": {str(k): v for k, v in HGB_KW.items()}, "defaults": M.HGB_DEFAULTS, "mode": variant.get("hgb_mode", "auto")},
        "blend": {"ridge": 0.5, "hgb": 0.5, "ridge_only_from_horizon": LONG_HORIZON},
        "preprocessing": {"impute": "train median", "scale": "train mean/std", "winsorize_sigma": 5.0, "hgb_row_stride": M.HGB_DEFAULTS["row_stride"]},
        "labels": "observed-only" if st.get("observed_only_labels") else "filled values allowed",
        "overlay": {"seoul_rate_overlay": bool(st.get("seoul_rate_overlay")), "max_horizon": st.get("overlay_max_horizon")},
        "bias_correction": variant.get("bias"),
        "variant": variant,
        "walk_forward": {k: st.get(k) for k in ("first_origin", "refit_every", "eval_step")},
        "interval_target": st.get("interval_target"),
    }


def config_id(fit: EngineFit) -> str:
    return hashlib.sha256(json.dumps(structural_config(fit), sort_keys=True, default=str).encode()).hexdigest()[:10]


def model_config(fit: EngineFit) -> dict:
    """Full record of the configuration plus the data-dependent overlay values; `hash` identifies the structural part only."""
    config = {**structural_config(fit), "settings": fit.settings, "overlay_info": fit.overlay}
    config["hash"] = config_id(fit)
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
                    "origin_price": np.exp(fit.log_price.loc[origin].reindex(live.index).to_numpy()),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)[LOG_COLUMNS] if rows else pd.DataFrame(columns=LOG_COLUMNS)


def data_id_of(kb: KBPanel) -> str:
    return hashlib.sha256(kb.fingerprint.encode()).hexdigest()[:10]


def write_log(fit: EngineFit, kb: KBPanel, log_dir: Path, regions: set[str] | None = None, note: str = "") -> tuple[Path | None, str]:
    """Persist the live forecasts. Returns (csv path or None if this origin + data + configuration is already logged, message)."""
    frame = snapshot(fit, regions)
    if frame.empty:
        return None, "기록할 예측이 없습니다."
    data_id = data_id_of(kb)
    cfg_id = config_id(fit)
    variant = (fit.settings.get("variant") or {}).get("name", "current")
    stem = f"{fit.last_date.date().isoformat()}_{data_id}_{cfg_id}"
    csv, meta = Path(log_dir) / f"{stem}.csv", Path(log_dir) / f"{stem}.json"
    if csv.exists():
        return None, f"이미 기록됨: {csv.name}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    config = model_config(fit)
    metadata = {
        "origin": fit.last_date.date().isoformat(),
        "logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "versions": library_versions(),
        "data_fingerprint": kb.fingerprint,  # sha256 of the workbook, plus the hub extension date if the hub was used
        "data_id": data_id,
        "workbook_last_date": str(kb.sale.index.max().date()),
        "config_id": cfg_id,
        "variant": variant,
        "rows": int(len(frame)),
        "regions": int(frame["region"].nunique()),
        "model_config": config,
        "note": note,
    }
    frame.insert(0, "variant", variant)
    frame.insert(0, "config_id", cfg_id)
    frame.insert(0, "data_id", data_id)
    frame.to_csv(csv, index=False, float_format="%.6f")
    meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return csv, f"기록 완료: {csv.name} ({len(frame)}행, 변형 {variant}, 설정 {cfg_id})"


def load_log(log_dir: Path) -> pd.DataFrame:
    """All logged records. Legacy files (no config id in the name) get config_id 'legacy-<hash>' and variant 'current'."""
    import re

    frames = []
    for csv in sorted(Path(log_dir).glob("*.csv")):
        if not re.match(r"^\d{4}-\d{2}-\d{2}_", csv.name):  # e.g. realized_first.csv lives next to the logs but is not one
            continue
        f = pd.read_csv(csv, parse_dates=["origin"])
        meta_path = csv.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        parts = csv.stem.split("_")
        if "config_id" not in f.columns:
            legacy_hash = str(meta.get("model_config", {}).get("hash", "unknown"))[:8]
            f.insert(0, "variant", meta.get("variant", "current"))
            f.insert(0, "config_id", meta.get("config_id", f"legacy-{legacy_hash}"))
            f.insert(0, "data_id", meta.get("data_id", parts[1] if len(parts) > 1 else "unknown"))
        if "origin_price" not in f.columns:
            f["origin_price"] = np.nan
        f.insert(0, "log_file", csv.name)
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["log_file", "data_id", "config_id", "variant", *LOG_COLUMNS])


# ----------------------------------------------------------------------------- realised values: first seen, first published, latest
LATE_COLLECTION_DAYS = 14  # a value first collected more than this many days after its week is flagged: it may already have been revised


def record_first_seen(kb: KBPanel, path: Path, since: pd.Timestamp | None = None, today: pd.Timestamp | None = None) -> int:
    """Append, never rewrite: every actually observed (region, week) value of the sale index that the archive has not seen yet.

    This is the value THIS SYSTEM saw first, which is not necessarily the value KB first published: a workbook downloaded weeks later (or a
    first run that back-fills older weeks) may already contain revisions. Each row therefore carries `first_seen_utc`, the data id,
    the newest week of the panel at that time, `collection_lag_days` (first seen minus the week's date) and `late_collection`
    (lag above `LATE_COLLECTION_DAYS`), and the archive has a sidecar `<name>.meta.json` with its creation time. A value is called a first
    PUBLICATION value only if its vintage was confirmed separately (`verify_vintage`); nothing is promoted automatically.
    Returns the number of rows appended.
    """
    price = kb.sale
    seen = kb.observed["sale"] if kb.observed is not None else price.notna()
    long = price.where(seen).stack().rename("value").reset_index()
    long.columns = ["date", "region", "value"]
    if since is not None:
        long = long[long["date"] >= pd.Timestamp(since)]
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["date"])
        key = set(zip(old["region"], old["date"]))
        long = long[[(r, d) not in key for r, d in zip(long["region"], long["date"])]]
    else:
        old = None
    if long.empty:
        return 0
    now = pd.Timestamp(today) if today is not None else pd.Timestamp(datetime.now(timezone.utc).date())
    lag = (now - long["date"]).dt.days
    long = long.assign(
        first_seen_utc=now.strftime("%Y-%m-%d"), data_id=data_id_of(kb), panel_last_date=str(kb.sale.index.max().date()),
        collection_lag_days=lag.to_numpy(), late_collection=(lag > LATE_COLLECTION_DAYS).to_numpy(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(path, mode="a", header=old is None, index=False, float_format="%.6f")
    meta = path.with_suffix(".meta.json")
    if not meta.exists():
        meta.write_text(json.dumps({
            "archive_created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "meaning": "first_seen = first value this system saw; NOT necessarily the first published value",
            "late_collection_days": LATE_COLLECTION_DAYS,
            "first_batch_rows": int(len(long)), "first_batch_late_rows": int(long["late_collection"].sum()),
            "note": "rows of the first batch are mostly late (back-filled); use first_published only for vintages verified via verify_vintage",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(len(long))


def verify_vintage(path: Path, region: str, dates: list, evidence: str, verified_by: str) -> int:
    """Append-only record that the archived first-seen value of these weeks is the first PUBLISHED one (e.g. collected the day it appeared,
    or matched against a dated copy of KB's release). `evidence` must say how; the archive itself is not modified."""
    rows = pd.DataFrame({"region": region, "date": pd.to_datetime(dates), "evidence": evidence, "verified_by": verified_by, "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    path = Path(path)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)
    return int(len(rows))


def score_log(log: pd.DataFrame, kb: KBPanel, realized: str = "latest", first_seen_path: Path | None = None, verified_path: Path | None = None) -> pd.DataFrame:
    """Attach realised returns to logged forecasts whose target week has arrived; only actually observed prices count.

    `y_latest`: today's values of the index at the origin and the target week (a real observation at both). `y_first_seen`: the logged
    origin price against the value the target week was first SEEN at (archive `realized_first_seen.csv`; NaN where either is missing, e.g. for
    records written before `origin_price` existed). `y_first_published`: the same, but only for target weeks whose vintage was verified
    (`vintage_verified.csv`, see `verify_vintage`). `late_target_collection` tells whether the target value was first seen late. `y` is the
    one chosen by `realized` ('latest', 'first_seen' or 'first_published'). Errors are cumulative log-return differences.
    """
    if realized not in ("latest", "first_seen", "first_published"):
        raise ValueError(realized)
    if log.empty:
        return log.assign(target_date=pd.NaT, y=np.nan, y_latest=np.nan, y_first_seen=np.nan, y_first_published=np.nan, late_target_collection=np.nan)
    price = np.log(kb.sale)
    seen = kb.observed["sale"] if kb.observed is not None else kb.sale.notna()
    out = log.copy()
    out["target_date"] = out["origin"] + pd.to_timedelta(out["horizon"] * 7, unit="D")
    first = None
    if first_seen_path is not None and Path(first_seen_path).exists():
        arch = pd.read_csv(first_seen_path, parse_dates=["date"])
        first = arch.set_index(["region", "date"])[["value", "late_collection"]]
    verified: set = set()
    if verified_path is not None and Path(verified_path).exists():
        v = pd.read_csv(verified_path, parse_dates=["date"])
        verified = set(zip(v["region"], v["date"]))
    y_latest = np.full(len(out), np.nan)
    y_seen = np.full(len(out), np.nan)
    y_pub = np.full(len(out), np.nan)
    late = np.full(len(out), np.nan)
    for i, row in enumerate(out.itertuples()):
        if row.region in price.columns and row.target_date in price.index and row.origin in price.index and seen.at[row.origin, row.region] and seen.at[row.target_date, row.region]:
            y_latest[i] = price.at[row.target_date, row.region] - price.at[row.origin, row.region]
        key = (row.region, row.target_date)
        if first is not None and np.isfinite(row.origin_price) and key in first.index:
            y_seen[i] = np.log(first.loc[key, "value"]) - np.log(row.origin_price)
            late[i] = float(first.loc[key, "late_collection"])
            if key in verified:
                y_pub[i] = y_seen[i]
    out["y_latest"], out["y_first_seen"], out["y_first_published"], out["late_target_collection"] = y_latest, y_seen, y_pub, late
    out["y"] = out[{"latest": "y_latest", "first_seen": "y_first_seen", "first_published": "y_first_published"}[realized]]
    return out


def pair_models(scored: pd.DataFrame, base: str, cand: str) -> pd.DataFrame:
    """Join two configurations (config id or variant name) on origin / region / horizon / data vintage: identical rows only."""
    def pick(key: str) -> pd.DataFrame:
        sel = scored[(scored["config_id"] == key) | (scored["variant"] == key)]
        if sel.empty:
            raise KeyError(f"no logged forecasts for {key!r}")
        return sel.drop_duplicates(["origin", "region", "horizon", "data_id", "config_id"])

    keys = ["origin", "region", "horizon", "data_id"]
    a, b = pick(base), pick(cand)
    cols = ["pred", "lo", "hi", "pred_raw"]
    merged = a[keys + ["y", "y_latest", "y_first_seen", "y_first_published", *cols]].merge(b[keys + cols], on=keys, suffixes=("_base", "_cand"))
    return merged


def compare_logged(paired: pd.DataFrame, regions_seoul: set[str] | None = None) -> pd.DataFrame:
    """Point forecast accuracy and interval coverage of two paired configurations, reported in separate columns (they can disagree)."""
    done = paired.dropna(subset=["y"])
    rows = []
    sets = [("전체", done)] + ([("서울", done[done["region"].isin(regions_seoul)])] if regions_seoul else [])
    for label, sub in sets:
        for h, g in sub.groupby("horizon"):
            cov = lambda lo, hi: float(((g["y"] >= g[lo]) & (g["y"] <= g[hi])).mean())  # noqa: E731
            mb, mc = float((g["pred_base"] - g["y"]).abs().mean() * 100), float((g["pred_cand"] - g["y"]).abs().mean() * 100)
            rows.append({"set": label, "horizon": int(h), "origins": int(g["origin"].nunique()), "n": int(len(g)), "MAE_base_pp": mb, "MAE_cand_pp": mc, "MAE_rel_change_pct": (mc / mb - 1) * 100 if mb else np.nan,
                         "coverage_base": cov("lo_base", "hi_base"), "coverage_cand": cov("lo_cand", "hi_cand"), "width_base_pp": float((g["hi_base"] - g["lo_base"]).mean() * 100), "width_cand_pp": float((g["hi_cand"] - g["lo_cand"]).mean() * 100)})
    return pd.DataFrame(rows)


def summarize(scored: pd.DataFrame, kb: KBPanel) -> pd.DataFrame:
    """Per configuration and horizon: MAE of the logged forecast vs the random walk and the 26-week drift, for Seoul series and for all logged regions."""
    done = scored.dropna(subset=["y"])
    cols = ["config_id", "variant", "set", "horizon", "origins", "n", "MAE_pp", "raw_MAE_pp", "drift26_MAE_pp", "rw_MAE_pp", "interval_coverage"]
    if done.empty:
        return pd.DataFrame(columns=cols)
    seoul = seoul_region_keys(kb.hierarchy)
    rows = []
    for (cfg, var), part in done.groupby(["config_id", "variant"]):
        for label, sub in (("서울", part[part["region"].isin(seoul)]), ("전체", part)):
            for h, g in sub.groupby("horizon"):
                rows.append(
                    {
                        "config_id": cfg, "variant": var, "set": label, "horizon": int(h), "origins": int(g["origin"].nunique()), "n": int(len(g)),
                        "MAE_pp": float((g["pred"] - g["y"]).abs().mean() * 100), "raw_MAE_pp": float((g["pred_raw"] - g["y"]).abs().mean() * 100),
                        "drift26_MAE_pp": float((g["drift26"] - g["y"]).abs().mean() * 100), "rw_MAE_pp": float((g["rw"] - g["y"]).abs().mean() * 100),
                        "interval_coverage": float(((g["y"] >= g["lo"]) & (g["y"] <= g["hi"])).mean()),
                    }
                )
    return pd.DataFrame(rows, columns=cols)
