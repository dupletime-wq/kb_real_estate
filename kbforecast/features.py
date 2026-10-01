"""Causal feature engineering for the pooled region panel.

Every feature at row (t, region) only uses information available on date t (KB data up to week t; macro series
via `macro.align_weekly`, which applies publication lags).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .kb_panel import KBPanel

MARKET_REFS = ("서울특별시", "수도권", "경기도", "전국", "6개광역시")
INDICATORS = ("buyer", "sale_txn", "jeonse_supply", "jeonse_txn")
GROUP_ORDER = ("own", "own2", "market", "jeonse", "sentiment", "sent2", "sent3", "macro")


@dataclass(frozen=True)
class FeatureSet:
    X: pd.DataFrame  # long frame, MultiIndex (date, region), float32
    log_price: pd.DataFrame  # wide log sale index
    groups: dict[str, list[str]]
    reference: pd.Series  # region -> benchmark aggregate key used for relative features


def _bcast(series: pd.Series, columns: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(np.repeat(series.to_numpy(dtype=float)[:, None], len(columns), axis=1), index=series.index, columns=columns)


def _reference_map(kb: KBPanel) -> pd.Series:
    cols = set(kb.sale.columns)
    refs = {}
    for row in kb.hierarchy.itertuples():
        if row.level == "group":
            ref = "서울특별시"
        elif row.level in ("province", "agg"):
            ref = "전국"
        else:
            ref = row.parent if row.parent in cols else (row.province if row.province in cols else "전국")
        refs[row.key] = ref if ref in cols else "전국"
    return pd.Series(refs)


def _sentiment_wide(kb: KBPanel, indicator: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = kb.sentiment[indicator].reindex(index)
    scopes = kb.hierarchy["sentiment_scope"].reindex(kb.sale.columns)
    out = {}
    for region, scope in scopes.items():
        out[region] = frame[scope] if scope in frame.columns else np.nan
    return pd.DataFrame(out, index=index)


def _macro_features(macro_w: pd.DataFrame) -> dict[str, pd.Series]:
    m = macro_w
    f: dict[str, pd.Series] = {}

    def has(*names: str) -> bool:
        return all(n in m.columns for n in names)

    if has("base_rate"):
        f["mc_base_lvl"] = m["base_rate"]
        f["mc_base_d26"] = m["base_rate"] - m["base_rate"].shift(26)
        f["mc_base_d52"] = m["base_rate"] - m["base_rate"].shift(52)
    if has("cd91"):
        f["mc_cd_lvl"] = m["cd91"]
        f["mc_cd_d13"] = m["cd91"] - m["cd91"].shift(13)
    if has("ktb3y"):
        f["mc_ktb3_lvl"] = m["ktb3y"]
        f["mc_ktb3_d13"] = m["ktb3y"] - m["ktb3y"].shift(13)
        f["mc_ktb3_d26"] = m["ktb3y"] - m["ktb3y"].shift(26)
    if has("ktb3y", "ktb10y"):
        term = m["ktb10y"] - m["ktb3y"]
        f["mc_term_lvl"] = term
        f["mc_term_d26"] = term - term.shift(26)
    if has("usdkrw"):
        lg = np.log(m["usdkrw"])
        f["mc_fx_r13"] = lg - lg.shift(13)
        f["mc_fx_r26"] = lg - lg.shift(26)
    if has("mortgage_rate"):
        f["mc_mort_lvl"] = m["mortgage_rate"]
        f["mc_mort_d13"] = m["mortgage_rate"] - m["mortgage_rate"].shift(13)
        f["mc_mort_d26"] = m["mortgage_rate"] - m["mortgage_rate"].shift(26)
    if has("mortgage_rate", "base_rate"):
        f["mc_mort_spread"] = m["mortgage_rate"] - m["base_rate"]
    if has("housing_loan"):
        yoy = m["housing_loan"] / m["housing_loan"].shift(52) - 1.0
        f["mc_hloan_yoy"] = yoy
        f["mc_hloan_yoy_d13"] = yoy - yoy.shift(13)
    if has("m2"):
        yoy = m["m2"] / m["m2"].shift(52) - 1.0
        f["mc_m2_yoy"] = yoy
        f["mc_m2_yoy_d13"] = yoy - yoy.shift(13)
    for name, tag in (("csi_house_all", "csih"), ("csi_house_seoul", "csihs"), ("csi_rate_all", "csir"), ("csi_sentiment", "csis")):
        if has(name):
            f[f"mc_{tag}_lvl"] = m[name]
            f[f"mc_{tag}_d13"] = m[name] - m[name].shift(13)
    for name, tag in (("unsold_seoul", "unsS"), ("unsold_nation", "unsN")):
        if has(name):
            lg = np.log(m[name].clip(lower=1.0))
            f[f"mc_{tag}_dev52"] = lg - lg.rolling(52, min_periods=26).mean()
            f[f"mc_{tag}_d26"] = lg - lg.shift(26)
    if has("kospi_m"):
        lg = np.log(m["kospi_m"])
        f["mc_kospi_r13"] = lg - lg.shift(13)
        f["mc_kospi_r26"] = lg - lg.shift(26)
    return f


def build_features(kb: KBPanel, macro_w: pd.DataFrame | None = None) -> FeatureSet:
    L = np.log(kb.sale.where(kb.sale > 0))
    LJ = np.log(kb.jeonse.where(kb.jeonse > 0))
    cols = L.columns
    feats: dict[str, pd.DataFrame] = {}
    groups: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}

    def add(group: str, name: str, frame: pd.DataFrame) -> None:
        feats[name] = frame
        groups[group].append(name)

    def ret(frame: pd.DataFrame, k: int) -> pd.DataFrame:
        return frame - frame.shift(k)

    r = {k: ret(L, k) for k in (1, 4, 8, 13, 26, 52)}
    for k, frame in r.items():
        add("own", f"r{k}", frame)
    add("own", "acc13", r[13] - (L.shift(13) - L.shift(26)))
    add("own", "acc26", r[26] - (L.shift(26) - L.shift(52)))
    dw = L.diff()
    vol13 = dw.rolling(13, min_periods=8).std()
    vol52 = dw.rolling(52, min_periods=26).std()
    add("own", "vol13", vol13)
    add("own", "vol52", vol52)
    add("own", "trend_t13", r[13] / (vol13 * np.sqrt(13.0)).replace(0, np.nan))
    add("own", "dd52", L - L.rolling(52, min_periods=26).max())
    add("own", "up52", L - L.rolling(52, min_periods=26).min())
    add("own", "gap26", L - L.rolling(26, min_periods=13).mean())

    for k in (2, 6, 39):
        add("own2", f"r{k}", ret(L, k))
    add("own2", "dd26", L - L.rolling(26, min_periods=13).max())
    add("own2", "acc8", r[8] - (L.shift(8) - L.shift(16)))
    add("own2", "r4_over_vol", r[4] / (vol13 * 2.0).replace(0, np.nan))

    ref = _reference_map(kb)
    for name in MARKET_REFS:
        if name in cols:
            for k in (13, 26):
                add("market", f"mk_{name}_r{k}", _bcast(r[k][name], cols))
    ref_r13 = pd.DataFrame({c: r[13][ref[c]] for c in cols})
    ref_r26 = pd.DataFrame({c: r[26][ref[c]] for c in cols})
    add("market", "rel13", r[13] - ref_r13)
    add("market", "rel26", r[26] - ref_r26)
    add("market", "cs_rank13", r[13].rank(axis=1, pct=True))
    add("market", "cs_rank26", r[26].rank(axis=1, pct=True))

    if LJ.notna().any().any():
        jr = {k: ret(LJ, k) for k in (4, 13, 26, 52)}
        for k, frame in jr.items():
            add("jeonse", f"jr{k}", frame)
        add("jeonse", "rc26", jr[26] - r[26])
        add("jeonse", "rc52", jr[52] - r[52])
        sj = L - LJ
        dev = sj - sj.rolling(156, min_periods=78).mean()
        add("jeonse", "sj_dev156", dev)
        add("jeonse", "sj_z156", dev / sj.rolling(156, min_periods=78).std().replace(0, np.nan))

    for indicator in INDICATORS:
        if indicator not in kb.sentiment:
            continue
        wide = _sentiment_wide(kb, indicator, L.index)
        add("sentiment", f"{indicator}_lvl", wide)
        add("sentiment", f"{indicator}_d4", wide - wide.shift(4))
        add("sentiment", f"{indicator}_d13", wide - wide.shift(13))
        mean52 = wide.rolling(52, min_periods=26).mean()
        std52 = wide.rolling(52, min_periods=26).std().replace(0, np.nan)
        add("sentiment", f"{indicator}_z52", (wide - mean52) / std52)
        add("sent2", f"{indicator}_d8", wide - wide.shift(8))
        add("sent2", f"{indicator}_d26", wide - wide.shift(26))
        add("sent2", f"{indicator}_ma4dev", wide - wide.rolling(4, min_periods=2).mean())
        add("sent2", f"{indicator}_dev156", wide - wide.rolling(156, min_periods=78).mean())
        # region sentiment relative to its parent province (regional divergence from the wider market)
        parent_scope = kb.hierarchy["province"].reindex(cols).map(lambda v: {"제주도": "제주특별자치도"}.get(v, v))
        prov = pd.DataFrame({c: kb.sentiment[indicator][ps].reindex(L.index) if ps in kb.sentiment[indicator].columns else np.nan
                             for c, ps in parent_scope.items()}, index=L.index)
        add("sent2", f"{indicator}_vs_prov_d13", (wide - wide.shift(13)) - (prov - prov.shift(13)))
        # distance from the recent peak: a sentiment index that has rolled over while prices still rise
        add("sent3", f"{indicator}_dd26", wide - wide.rolling(26, min_periods=13).max())
        if indicator == "buyer":
            add("sent3", "buyer_dd52", wide - wide.rolling(52, min_periods=26).max())
            add("sent3", "buyer_up26", wide - wide.rolling(26, min_periods=13).min())

    if macro_w is not None and not macro_w.empty:
        for name, series in _macro_features(macro_w.reindex(L.index)).items():
            add("macro", name, _bcast(series, cols))

    names = list(feats)
    stacked = np.stack([feats[n].reindex(index=L.index, columns=cols).to_numpy(dtype=np.float32) for n in names], axis=-1)
    T, R, F = stacked.shape
    index = pd.MultiIndex.from_product([L.index, cols], names=["date", "region"])
    X = pd.DataFrame(stacked.reshape(T * R, F), index=index, columns=names)
    X = X.replace([np.inf, -np.inf], np.nan)
    groups = {g: v for g, v in groups.items() if v}
    return FeatureSet(X=X, log_price=L, groups=groups, reference=ref)


def make_targets(log_price: pd.DataFrame, horizon: int, observed: pd.DataFrame | None = None) -> pd.Series:
    """h-week-ahead log return, long-indexed like the feature frame (NaN where the future is unobserved).

    With `observed` (True = actual observation, see `KBPanel.observed`) the target is kept only where both the origin price and the
    price `horizon` weeks later were real observations, so no label is built from a carried-forward (filled) value.
    """
    y = log_price.shift(-horizon) - log_price
    if observed is not None:
        obs = observed.reindex(index=log_price.index, columns=log_price.columns).fillna(False).astype(bool)
        y = y.where(obs & obs.shift(-horizon, fill_value=False))
    return y.stack(future_stack=True).rename(f"y{horizon}").reindex(
        pd.MultiIndex.from_product([log_price.index, log_price.columns], names=["date", "region"])
    )
