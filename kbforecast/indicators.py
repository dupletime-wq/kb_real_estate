"""Rule-based technical signals on a single price series, each validated on its own forward returns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class IndicatorResult:
    frame: pd.DataFrame
    passed_summary: pd.DataFrame
    passed_events: pd.DataFrame


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_indicator_frame(series: pd.Series) -> pd.DataFrame:
    value = series.astype(float)
    frame = pd.DataFrame({"date": value.index, "value": value.to_numpy(dtype=float)})
    frame["ma13"] = value.rolling(13, min_periods=8).mean().to_numpy()
    frame["ma26"] = value.rolling(26, min_periods=13).mean().to_numpy()
    frame["ma52"] = value.rolling(52, min_periods=26).mean().to_numpy()
    bb_mid = value.rolling(20, min_periods=12).mean()
    bb_std = value.rolling(20, min_periods=12).std()
    frame["bb_mid"] = bb_mid.to_numpy()
    frame["bb_upper"] = (bb_mid + 2 * bb_std).to_numpy()
    frame["bb_lower"] = (bb_mid - 2 * bb_std).to_numpy()
    ema12 = value.ewm(span=12, adjust=False).mean()
    ema26 = value.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    frame["macd"] = macd.to_numpy()
    frame["macd_signal"] = macd.ewm(span=9, adjust=False).mean().to_numpy()
    frame["rsi14"] = _rsi(value, 14).to_numpy()
    high9 = value.rolling(9, min_periods=9).max()
    low9 = value.rolling(9, min_periods=9).min()
    tenkan = (high9 + low9) / 2
    high26 = value.rolling(26, min_periods=26).max()
    low26 = value.rolling(26, min_periods=26).min()
    kijun = (high26 + low26) / 2
    high52 = value.rolling(52, min_periods=52).max()
    low52 = value.rolling(52, min_periods=52).min()
    frame["tenkan"] = tenkan.to_numpy()
    frame["kijun"] = kijun.to_numpy()
    frame["span_a"] = ((tenkan + kijun) / 2).to_numpy()
    frame["span_b"] = ((high52 + low52) / 2).to_numpy()
    rolling_mean = value.rolling(26, min_periods=13).mean()
    rolling_std = value.rolling(26, min_periods=13).std()
    frame["zscore26"] = ((value - rolling_mean) / rolling_std.replace(0, np.nan)).to_numpy()
    frame["momentum13"] = value.pct_change(13).to_numpy()
    return frame


def _indicator_definitions(indicators: pd.DataFrame) -> list[tuple[str, str, str, pd.Series]]:
    value = indicators["value"]
    prev_macd = indicators["macd"].shift(1)
    prev_signal = indicators["macd_signal"].shift(1)
    return [
        ("Bollinger", "하단 이탈", "bullish", value <= indicators["bb_lower"]),
        ("Bollinger", "상단 돌파", "bearish", value >= indicators["bb_upper"]),
        (
            "MACD",
            "상향 교차",
            "bullish",
            (indicators["macd"] > indicators["macd_signal"]) & (prev_macd <= prev_signal),
        ),
        (
            "MACD",
            "하향 교차",
            "bearish",
            (indicators["macd"] < indicators["macd_signal"]) & (prev_macd >= prev_signal),
        ),
        ("RSI", "과매도", "bullish", indicators["rsi14"] <= 35),
        ("RSI", "과열", "bearish", indicators["rsi14"] >= 65),
        (
            "Ichimoku",
            "구름 상단 우위",
            "bullish",
            (value > indicators[["span_a", "span_b"]].max(axis=1)) & (indicators["tenkan"] > indicators["kijun"]),
        ),
        (
            "Ichimoku",
            "구름 하단 약세",
            "bearish",
            (value < indicators[["span_a", "span_b"]].min(axis=1)) & (indicators["tenkan"] < indicators["kijun"]),
        ),
        ("Z-score", "저평가", "bullish", indicators["zscore26"] <= -1.5),
        ("Z-score", "고평가", "bearish", indicators["zscore26"] >= 1.5),
        (
            "Momentum",
            "13주 모멘텀 전환",
            "bullish",
            (indicators["momentum13"] > 0) & (indicators["momentum13"].shift(1) <= 0),
        ),
        (
            "Momentum",
            "13주 모멘텀 둔화",
            "bearish",
            (indicators["momentum13"] < 0) & (indicators["momentum13"].shift(1) >= 0),
        ),
    ]


def evaluate_indicators(series: pd.Series, horizon: int) -> IndicatorResult:
    indicators = compute_indicator_frame(series)
    forward_horizon = min(13, horizon)
    future_return = indicators["value"].shift(-forward_horizon) / indicators["value"] - 1.0
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for indicator, signal_name, direction, mask in _indicator_definitions(indicators):
        valid = mask.fillna(False) & future_return.notna()
        signal_returns = future_return.loc[valid]
        if direction == "bearish":
            signed_returns = -signal_returns
            hit = signal_returns < 0
        else:
            signed_returns = signal_returns
            hit = signal_returns > 0
        sample_size = int(valid.sum())
        if sample_size:
            hit_count = int(hit.sum())
            hit_rate = float(hit_count / sample_size)
            try:
                p_binom = float(stats.binomtest(hit_count, sample_size, 0.5, alternative="greater").pvalue)
            except AttributeError:
                p_binom = float(stats.binom_test(hit_count, sample_size, 0.5, alternative="greater"))
            try:
                p_mean = float(stats.ttest_1samp(signed_returns, 0.0, alternative="greater").pvalue)
            except TypeError:
                p_mean = float(stats.ttest_1samp(signed_returns, 0.0).pvalue / 2.0)
            mean_forward = float(signal_returns.mean() * 100.0)
        else:
            hit_count = 0
            hit_rate = 0.0
            p_binom = np.nan
            p_mean = np.nan
            mean_forward = np.nan

        passed = (
            sample_size >= 10
            and hit_rate >= 0.55
            and (np.nanmin([p_binom, p_mean]) <= 0.25)
        )
        summary_rows.append(
            {
                "indicator": indicator,
                "signal": signal_name,
                "direction": "상승 기대" if direction == "bullish" else "하락/둔화 경계",
                "samples": sample_size,
                "hit_rate_pct": hit_rate * 100.0,
                "mean_forward_return_pct": mean_forward,
                "p_value": np.nanmin([p_binom, p_mean]) if sample_size else np.nan,
                "passed": bool(passed),
            }
        )
        if passed:
            signal_dates = indicators.loc[mask.fillna(False), ["date", "value"]].copy()
            signal_dates["indicator"] = indicator
            signal_dates["signal"] = signal_name
            signal_dates["direction"] = direction
            event_rows.append(signal_dates)

    summary = pd.DataFrame(summary_rows)
    passed_summary = (
        summary.loc[summary["passed"]]
        .sort_values(["hit_rate_pct", "samples"], ascending=[False, False])
        .reset_index(drop=True)
    )
    passed_events = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    return IndicatorResult(frame=indicators, passed_summary=passed_summary, passed_events=passed_events)
