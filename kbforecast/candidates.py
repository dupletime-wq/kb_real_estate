"""Pre-specified candidate features for the next round of tests. All are causal (a value at week t uses data available at t) and none
re-adds information the model already has (levels, changes and drawdowns of the sentiment indices are in `features.py`).

Family B  (sentiment x price interactions; the standardisation is done by the model on the TRAINING window, no threshold is estimated here)
    px_sent     trailing 13-week log return x 13-week change of the buyer-dominance index
    divergence  max(13-week return, 0) x max(-13-week change of the buyer index, 0)   (price still rising while sentiment falls)
    buyer_run   signed number of consecutive weeks (capped at 26) over which the 4-week change of the buyer index kept its sign
Family C  (observation quality; needs `KBPanel.observed`)
    obs_age       weeks since the last actual price observation (0 when this week was observed)
    fill_ratio26  share of the last 26 weeks (after the series started) that were filled, not observed
    Whether a sentiment value was observed or estimated from a parent scope has no history before the hub extension, so it is NOT
    reconstructed: it is a live diagnostic and a forward-logging item only (see report.data_diagnostics).
Family L  (long memory, for 52/104/208-week forecasts; only the sale index): r104 = 104-week log return, pdev156 / pdev260 = log price minus
    its trailing 156 / 260-week mean (at least 104 / 156 weeks of history).
Family J  (valuation, needs the jeonse index): sj_level = log sale - log jeonse (minus log of the jeonse-to-sale ratio), sj_dev156 = its deviation from the
    trailing 156-week mean, sj_z156 = that deviation in units of its trailing 156-week standard deviation. Same definitions as the unused columns of features.py. sj_level_seoul = sj_level for the Seoul series only (others empty).
Family V  (monthly trading volume from MOLIT, Seoul city / two halves / 25 districts only; see trades.py for the point-in-time design)
    vol_rel36     log((latest known month + 1) / (mean of the 36 months before it + 1))
    vol_chg3      log((last 3 known months + 1) / (the 3 months before them + 1))
    vol_px_inter  max(13-week return, 0) x max(-vol_chg3, 0)   (price up while volume falls)
    A month counts as known at week t only when its last day is at least the assumed reporting lag before t (12 weeks before 2020-05-25,
    8 weeks after: assumptions, not measurements; the history is FINAL data) and the weekly value is the latest known month held
    constant, never interpolated towards a later month. Net counts (reported minus cancelled) are used because cancelled rows exist
    only from 2020. Stock-based turnover (volume / housing stock) is not built: no stock history with publication dates is available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import _sentiment_wide
from .kb_panel import KBPanel, seoul_region_keys
from .trades import ASSUMED_LAG_WEEKS, LAG_SWITCH, seoul_weekly

FAMILIES = {
    "B": ("px_sent", "divergence", "buyer_run"),
    "C": ("obs_age", "fill_ratio26"),
    "V": ("vol_rel36", "vol_chg3", "vol_px_inter"),
    "L": ("r104", "pdev156", "pdev260"),  # long memory of the price itself (long-horizon round)
    "J": ("sj_level", "sj_dev156", "sj_z156", "sj_level_seoul"),  # price-to-jeonse valuation level and its deviation from the region's own past (long-horizon round)
}
ALL_CANDIDATE_FEATURES = tuple(n for names in FAMILIES.values() for n in names)


def _run_length(sign: np.ndarray, cap: int) -> np.ndarray:
    """Signed length of the current run of equal non-zero signs along axis 0 (NaN where the sign is unknown)."""
    out = np.full(sign.shape, np.nan)
    run = np.zeros(sign.shape[1])
    last = np.zeros(sign.shape[1])
    for t in range(sign.shape[0]):
        s = sign[t]
        known = np.isfinite(s)
        same = known & (s == last) & (s != 0)
        run = np.where(same, run + 1, np.where(known & (s != 0), 1.0, 0.0))
        last = np.where(known, s, np.nan)
        out[t] = np.where(known, np.sign(s) * np.minimum(run, cap), np.nan)
    return out


def sentiment_price_features(kb: KBPanel) -> dict[str, pd.DataFrame]:
    L = np.log(kb.sale.where(kb.sale > 0))
    r13 = L - L.shift(13)
    out: dict[str, pd.DataFrame] = {}
    if "buyer" not in kb.sentiment:
        return out
    buyer = _sentiment_wide(kb, "buyer", L.index)
    d13 = buyer - buyer.shift(13)
    out["px_sent"] = r13 * d13
    out["divergence"] = r13.clip(lower=0) * (-d13).clip(lower=0)
    sign = np.sign(buyer - buyer.shift(4)).to_numpy(dtype=float)
    out["buyer_run"] = pd.DataFrame(_run_length(sign, 26), index=L.index, columns=L.columns)
    return out


def quality_features(kb: KBPanel) -> dict[str, pd.DataFrame]:
    """Observation-quality features of the sale index. Needs the observed mask; without it nothing is invented."""
    if kb.observed is None:
        return {}
    obs = kb.observed["sale"].reindex(index=kb.sale.index, columns=kb.sale.columns).fillna(False).astype(bool)
    started = obs.cummax()
    age = pd.DataFrame(np.nan, index=obs.index, columns=obs.columns)
    counter = np.full(obs.shape[1], np.nan)
    arr = np.full(obs.shape, np.nan)
    o, st = obs.to_numpy(), started.to_numpy()
    for t in range(obs.shape[0]):
        counter = np.where(o[t], 0.0, np.where(st[t], counter + 1, np.nan))
        arr[t] = counter
    age = pd.DataFrame(arr, index=obs.index, columns=obs.columns)
    filled = ((~obs) & started).astype(float).where(started)
    return {"obs_age": age, "fill_ratio26": filled.rolling(26, min_periods=13).mean()}


def long_memory_features(kb: KBPanel) -> dict[str, pd.DataFrame]:
    L = np.log(kb.sale.where(kb.sale > 0))
    return {
        "r104": L - L.shift(104),
        "pdev156": L - L.rolling(156, min_periods=104).mean(),
        "pdev260": L - L.rolling(260, min_periods=156).mean(),
    }


def valuation_features(kb: KBPanel) -> dict[str, pd.DataFrame]:
    L = np.log(kb.sale.where(kb.sale > 0))
    LJ = np.log(kb.jeonse.where(kb.jeonse > 0))
    if not LJ.notna().any().any():
        return {}
    sj = L - LJ
    dev = sj - sj.rolling(156, min_periods=78).mean()
    # sj_level_seoul: the same level for the Seoul series only (every other region stays empty, like the V family), so its coefficient is learned
    # from Seoul rows alone. Post-hoc structure added after the J_level result was seen (README, long-horizon round).
    sj_seoul = sj.copy()
    sj_seoul.loc[:, ~sj.columns.isin(seoul_region_keys(kb.hierarchy))] = np.nan
    return {"sj_level": sj, "sj_dev156": dev, "sj_z156": dev / sj.rolling(156, min_periods=78).std().replace(0, np.nan), "sj_level_seoul": sj_seoul}


def monthly_net_counts(history: pd.DataFrame) -> pd.DataFrame:
    """Monthly net deals (reported - cancelled) per district code from the daily history; index = month end."""
    days = pd.to_datetime(history["deal_date"])
    net = history["n_all"] - history["n_cancelled"]
    monthly = net.groupby([history["sgg_cd"], days.dt.to_period("M")]).sum().unstack(0).fillna(0.0)
    monthly.index = monthly.index.to_timestamp(how="end").normalize()  # last day of the month
    return monthly


def volume_features(kb: KBPanel, history: pd.DataFrame, extra_lag_weeks: int = 0) -> dict[str, pd.DataFrame]:
    """Family V on the KB weekly calendar for the Seoul series (other regions stay NaN)."""
    dates = kb.sale.index
    monthly = seoul_weekly(monthly_net_counts(history), kb.hierarchy)  # same grouping as the weekly counts: districts, halves, city
    month_ends = monthly.index.to_numpy()
    lag_w = np.where(dates >= LAG_SWITCH, ASSUMED_LAG_WEEKS["from_2020"], ASSUMED_LAG_WEEKS["before_2020"]) + extra_lag_weeks
    cutoff = (dates - pd.to_timedelta(lag_w * 7, unit="D")).to_numpy()
    m_idx = np.searchsorted(month_ends, cutoff, side="right") - 1  # last month that ended on or before the cutoff
    v = monthly.to_numpy()
    n = len(month_ends)
    rel = np.full((len(dates), v.shape[1]), np.nan)
    chg = np.full((len(dates), v.shape[1]), np.nan)
    for i, m in enumerate(m_idx):
        if m < 41 or m >= n:  # 36 months of history for the level, 6 for the 3-month change
            continue
        rel[i] = np.log((v[m] + 1.0) / (v[m - 36:m].mean(axis=0) + 1.0))
        chg[i] = np.log((v[m - 2:m + 1].sum(axis=0) + 1.0) / (v[m - 5:m - 2].sum(axis=0) + 1.0))
    cols = monthly.columns
    r13 = (np.log(kb.sale.where(kb.sale > 0)) - np.log(kb.sale.where(kb.sale > 0)).shift(13)).reindex(columns=cols)
    chg_df = pd.DataFrame(chg, index=dates, columns=cols)
    out = {
        "vol_rel36": pd.DataFrame(rel, index=dates, columns=cols),
        "vol_chg3": chg_df,
        "vol_px_inter": r13.clip(lower=0) * (-chg_df).clip(lower=0),
    }
    return {k: f.reindex(columns=kb.sale.columns) for k, f in out.items()}


def build_candidate_features(kb: KBPanel, names: tuple[str, ...], history: pd.DataFrame | None = None, extra_lag_weeks: int = 0) -> dict[str, pd.DataFrame]:
    """Wide (date x region) frames for the requested candidate names. Unknown names raise; unavailable inputs raise instead of returning empty."""
    unknown = [n for n in names if n not in ALL_CANDIDATE_FEATURES]
    if unknown:
        raise KeyError(f"unknown candidate features: {unknown}")
    out: dict[str, pd.DataFrame] = {}
    if any(n in FAMILIES["B"] for n in names):
        out.update(sentiment_price_features(kb))
    if any(n in FAMILIES["C"] for n in names):
        out.update(quality_features(kb))
    if any(n in FAMILIES["L"] for n in names):
        out.update(long_memory_features(kb))
    if any(n in FAMILIES["J"] for n in names):
        out.update(valuation_features(kb))
    if any(n in FAMILIES["V"] for n in names):
        if history is None:
            raise ValueError("volume features need the trade history (trade_history/<date>/seoul_daily_counts.csv.gz)")
        out.update(volume_features(kb, history, extra_lag_weeks))
    missing = [n for n in names if n not in out]
    if missing:
        raise ValueError(f"inputs for {missing} are not available in this panel (e.g. no observed mask / no buyer index)")
    return {n: out[n] for n in names}


def with_candidates(fs, feats: dict[str, pd.DataFrame]):
    """FeatureSet with the candidate columns appended (existing columns of the same name are replaced)."""
    import dataclasses

    cols = {name: frame.stack(future_stack=True).reindex(fs.X.index).astype("float32") for name, frame in feats.items()}
    X = pd.concat([fs.X.drop(columns=list(cols), errors="ignore"), pd.DataFrame(cols)], axis=1)
    groups = {**fs.groups, "cand": list(cols)}
    return dataclasses.replace(fs, X=X, groups=groups)
