from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive, RandomWalkWithDrift, SeasonalNaive


DEFAULT_HORIZONS = (13, 26)
DEFAULT_FEATURE_SCOPE = "KB+Macro"
MODEL_VERSION = "kb-real-estate-v1"
SEASON_LENGTH = 52


@dataclass(frozen=True)
class BacktestReport:
    region: str
    leaderboard: pd.DataFrame
    predictions: pd.DataFrame
    event_summary: pd.DataFrame
    warnings: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ForecastBundle:
    region: str
    history: pd.DataFrame
    forecast_frame: pd.DataFrame
    model_table: pd.DataFrame
    top_models: tuple[str, ...]
    high_risk_score: float
    low_zone_score: float
    turning_points: pd.DataFrame
    warnings: tuple[str, ...]


def _ensure_weekly_frame(region_df: pd.DataFrame) -> pd.DataFrame:
    frame = region_df.copy()
    if "date" not in frame.columns:
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return frame[["date", "value"]]


def _rolling_percentile_last(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return np.nan
    last = valid[-1]
    sorted_valid = np.sort(valid)
    rank = np.searchsorted(sorted_valid, last, side="right")
    return (rank / valid.size) * 100.0


def _ewm(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def build_feature_matrix(region_df: pd.DataFrame, macro_features: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = _ensure_weekly_frame(region_df)
    frame = frame.rename(columns={"value": "y"})
    price = frame["y"].astype(float)
    for lag in range(1, 27):
        frame[f"lag_{lag}"] = price.shift(lag)
    for window in (4, 13, 26):
        frame[f"roll_mean_{window}"] = price.rolling(window=window, min_periods=window).mean()
        frame[f"roll_std_{window}"] = price.rolling(window=window, min_periods=window).std()
    frame["drawdown_26"] = (price / price.rolling(26, min_periods=4).max()) - 1.0
    frame["recovery_26"] = (price / price.rolling(26, min_periods=4).min()) - 1.0
    frame["roc_1"] = price.pct_change(1)
    frame["roc_4"] = price.pct_change(4)
    frame["roc_13"] = price.pct_change(13)
    frame["yoy_52"] = price.pct_change(52)
    frame["ma6"] = price.rolling(6, min_periods=6).mean()
    frame["ma24"] = price.rolling(24, min_periods=24).mean()
    exp12 = _ewm(price, 12)
    exp26 = _ewm(price, 26)
    frame["macd"] = exp12 - exp26
    frame["macd_signal"] = _ewm(frame["macd"], 9)
    frame["macd_hist"] = frame["macd"] - frame["macd_signal"]
    frame["disparity24"] = (price / frame["ma24"]) * 100.0
    ema1 = _ewm(price, 12)
    ema2 = _ewm(ema1, 12)
    ema3 = _ewm(ema2, 12)
    frame["trix"] = ema3.pct_change() * 100.0
    frame["trix_signal"] = _ewm(frame["trix"], 9)
    high_9 = price.rolling(9, min_periods=9).max()
    low_9 = price.rolling(9, min_periods=9).min()
    tenkan = (high_9 + low_9) / 2
    high_26 = price.rolling(26, min_periods=26).max()
    low_26 = price.rolling(26, min_periods=26).min()
    kijun = (high_26 + low_26) / 2
    high_52 = price.rolling(52, min_periods=52).max()
    low_52 = price.rolling(52, min_periods=52).min()
    span_a = (tenkan + kijun) / 2
    span_b = (high_52 + low_52) / 2
    frame["ichimoku_state"] = np.select(
        [
            (span_a > span_b) & (price >= span_a),
            (span_a < span_b) & (price <= span_b),
        ],
        [1.0, -1.0],
        default=0.0,
    )
    if macro_features is not None and not macro_features.empty:
        macro = macro_features.copy()
        macro["date"] = pd.to_datetime(macro["date"]).dt.normalize()
        frame = frame.merge(macro, on="date", how="left")
    return frame


def _infer_macro_frequency(dates: pd.Series) -> str:
    if len(dates) < 2:
        return "M"
    deltas = dates.sort_values().diff().dropna().dt.days
    median_days = deltas.median()
    if median_days >= 60:
        return "Q"
    if median_days >= 20:
        return "M"
    return "W"


def join_macro_asof(base_weekly_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    if macro_df is None or macro_df.empty:
        return pd.DataFrame(columns=["date"])
    base_dates = pd.DataFrame({"date": pd.to_datetime(base_weekly_df["date"]).dt.normalize().drop_duplicates().sort_values()})
    macro = macro_df.copy()
    if "date" not in macro.columns:
        macro = macro.reset_index().rename(columns={macro.index.name or "index": "date"})
    macro["date"] = pd.to_datetime(macro["date"]).dt.normalize()
    value_columns = [column for column in macro.columns if column != "date"]
    aligned = pd.DataFrame({"date": base_dates["date"]})
    for column in value_columns:
        series = macro[["date", column]].dropna().sort_values("date")
        if series.empty:
            continue
        frequency = _infer_macro_frequency(series["date"])
        lag_weeks = (4, 8, 12) if frequency != "Q" else (12, 24)
        release_lag = pd.Timedelta(weeks=4 if frequency == "M" else 6 if frequency == "Q" else 1)
        released = series.assign(date=series["date"] + release_lag)
        merged = pd.merge_asof(base_dates, released, on="date", direction="backward")
        merged[column] = merged[column].ffill()
        for lag in lag_weeks:
            aligned[f"{column}_lag_{lag}w"] = merged[column].shift(lag)
    return aligned


def label_turning_points(
    region_df: pd.DataFrame,
    *,
    window_radius: int = 6,
    lookahead: int = 13,
    min_duration: int = 4,
    threshold_pct: float = 0.02,
) -> pd.DataFrame:
    frame = _ensure_weekly_frame(region_df)
    smoothed = frame["value"].rolling(3, min_periods=1).mean()
    local_max = smoothed == smoothed.rolling((window_radius * 2) + 1, center=True, min_periods=window_radius + 1).max()
    local_min = smoothed == smoothed.rolling((window_radius * 2) + 1, center=True, min_periods=window_radius + 1).min()
    events: list[dict[str, Any]] = []
    for idx, row in frame.iterrows():
        future = frame.iloc[idx + 1 : idx + 1 + lookahead]
        if future.shape[0] < min_duration:
            continue
        current_value = float(row["value"])
        if local_max.iloc[idx]:
            future_drawdown = (future["value"].min() / current_value) - 1.0
            duration_condition = (future["value"].head(min_duration) < current_value).all()
            if future_drawdown <= -threshold_pct and duration_condition:
                events.append({"date": row["date"], "event": "peak", "value": current_value})
        if local_min.iloc[idx]:
            future_recovery = (future["value"].max() / current_value) - 1.0
            duration_condition = (future["value"].head(min_duration) > current_value).all()
            if future_recovery >= threshold_pct and duration_condition:
                events.append({"date": row["date"], "event": "trough", "value": current_value})
    if not events:
        return pd.DataFrame(columns=["date", "event", "value"])
    return pd.DataFrame(events).drop_duplicates(subset=["date", "event"]).sort_values("date").reset_index(drop=True)


def compute_historical_risk_scores(region_df: pd.DataFrame, window: int = 156) -> pd.DataFrame:
    features = build_feature_matrix(region_df)
    disparity = features["disparity24"]
    high_percentile = disparity.rolling(window, min_periods=min(window, 26)).apply(_rolling_percentile_last, raw=True)
    low_percentile = 100.0 - high_percentile
    near_high = (features["y"] / features["y"].rolling(52, min_periods=13).max()).rolling(
        window, min_periods=min(window, 26)
    ).apply(_rolling_percentile_last, raw=True)
    drawdown_depth = (-features["drawdown_26"]).rolling(window, min_periods=min(window, 26)).apply(
        _rolling_percentile_last, raw=True
    )
    high_score = (0.45 * high_percentile) + (0.35 * near_high.fillna(50.0)) + (0.2 * (100.0 - drawdown_depth.fillna(50.0)))
    low_score = (0.45 * low_percentile) + (0.35 * drawdown_depth.fillna(50.0)) + (0.2 * (100.0 - near_high.fillna(50.0)))
    return pd.DataFrame(
        {
            "date": features["date"],
            "high_score": high_score.clip(0.0, 100.0),
            "low_score": low_score.clip(0.0, 100.0),
        }
    )


def _mase_scale(series: pd.Series) -> float:
    diffs = series.diff().abs().dropna()
    if diffs.empty:
        return 1.0
    scale = float(diffs.mean())
    return scale if scale > 0 else 1.0


def _safe_rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(errors.astype(float))))) if not errors.empty else float("nan")


def _build_statsforecast_models(series_length: int) -> tuple[list[Any], list[str]]:
    warnings: list[str] = []
    seasonal_enabled = series_length >= 104
    models: list[Any] = [
        Naive(alias="Naive"),
        RandomWalkWithDrift(alias="Drift"),
        AutoETS(season_length=SEASON_LENGTH if seasonal_enabled else 1, model="ZZZ", alias="AutoETS"),
        AutoARIMA(season_length=SEASON_LENGTH if seasonal_enabled else 1, seasonal=seasonal_enabled, alias="AutoARIMA"),
    ]
    if seasonal_enabled:
        models.insert(2, SeasonalNaive(season_length=SEASON_LENGTH, alias="SeasonalNaive52"))
    else:
        warnings.append("Seasonal models were partially disabled because the series is shorter than 104 weeks.")
    return models, warnings


def _statsforecast_cv(region: str, region_df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    frame = _ensure_weekly_frame(region_df)
    models, warnings = _build_statsforecast_models(len(frame))
    ts_df = frame.rename(columns={"date": "ds", "value": "y"})
    ts_df["unique_id"] = region
    min_train = max(104, 3 * horizon)
    possible_windows = max(1, ((len(ts_df) - min_train - horizon) // 4) + 1)
    n_windows = max(1, min(6, possible_windows))
    sf = StatsForecast(models=models, freq="W-MON", n_jobs=1)
    cv = sf.cross_validation(
        df=ts_df[["unique_id", "ds", "y"]],
        h=horizon,
        n_windows=n_windows,
        step_size=4,
        refit=True,
    )
    cv["horizon_step"] = cv.groupby("cutoff").cumcount() + 1
    return cv, warnings


def _fit_mlforecast(region: str, region_df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    frame = _ensure_weekly_frame(region_df).rename(columns={"date": "ds", "value": "y"})
    frame["unique_id"] = region
    models = {
        "MLForecast-HGBR-p50": HistGradientBoostingRegressor(loss="squared_error", random_state=42, max_depth=4),
        "MLForecast-HGBR-p10": HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.1,
            random_state=42,
            max_depth=4,
        ),
        "MLForecast-HGBR-p90": HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.9,
            random_state=42,
            max_depth=4,
        ),
    }
    forecaster = MLForecast(
        models=models,
        freq="W-MON",
        lags=list(range(1, 27)),
        lag_transforms={
            1: [RollingMean(window_size=4), RollingStd(window_size=4)],
            4: [RollingMean(window_size=13), RollingStd(window_size=13)],
            13: [RollingMean(window_size=26), RollingStd(window_size=26)],
        },
        date_features=[],
        num_threads=1,
    )
    forecaster.fit(frame[["unique_id", "ds", "y"]], max_horizon=horizon, static_features=[])
    predictions = forecaster.predict(horizon)
    return predictions, []


def _mlforecast_cv(region: str, region_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = _ensure_weekly_frame(region_df)
    min_train = max(104, 3 * horizon)
    possible_windows = max(1, ((len(frame) - min_train - horizon) // 4) + 1)
    n_windows = max(1, min(6, possible_windows))
    windows: list[dict[str, Any]] = []
    for window_idx in range(n_windows):
        cutoff_index = min_train - 1 + (window_idx * 4)
        cutoff = frame.iloc[cutoff_index]["date"]
        target_index = cutoff_index + horizon
        if target_index >= len(frame):
            break
        train = frame.iloc[: cutoff_index + 1].copy()
        preds, _ = _fit_mlforecast(region, train, horizon)
        target_row = preds.iloc[horizon - 1]
        actual = float(frame.iloc[target_index]["value"])
        origin_value = float(train.iloc[-1]["value"])
        windows.append(
            {
                "horizon": horizon,
                "cutoff": cutoff,
                "target_date": pd.Timestamp(target_row["ds"]),
                "actual": actual,
                "origin_value": origin_value,
                "MLForecast-HGBR": float(target_row["MLForecast-HGBR-p50"]),
                "MLForecast-HGBR-p10": float(target_row["MLForecast-HGBR-p10"]),
                "MLForecast-HGBR-p90": float(target_row["MLForecast-HGBR-p90"]),
            }
        )
    return pd.DataFrame(windows)


def _score_predictions(
    *,
    actual: pd.Series,
    forecast: pd.Series,
    origin_value: pd.Series,
    scale: float,
) -> dict[str, float]:
    errors = forecast - actual
    predicted_direction = np.sign(forecast - origin_value)
    actual_direction = np.sign(actual - origin_value)
    directional_accuracy = float((predicted_direction == actual_direction).mean()) if len(actual) else float("nan")
    return {
        "mae": float(errors.abs().mean()),
        "rmse": _safe_rmse(errors),
        "mase": float(errors.abs().mean() / scale),
        "directional_accuracy": directional_accuracy,
        "n_windows": int(len(actual)),
    }


def _build_leaderboard(cv_df: pd.DataFrame, scale: float, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_horizon = cv_df.loc[cv_df["horizon_step"] == horizon].copy()
    rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for model in [column for column in final_horizon.columns if column not in {"unique_id", "ds", "cutoff", "y", "horizon_step", "origin_value"}]:
        scored = _score_predictions(
            actual=final_horizon["y"],
            forecast=final_horizon[model],
            origin_value=final_horizon["origin_value"],
            scale=scale,
        )
        rows.append({"horizon": horizon, "model": model, **scored})
        model_predictions = final_horizon[["cutoff", "ds", "y", "origin_value"]].copy()
        model_predictions["model"] = model
        model_predictions["forecast"] = final_horizon[model].astype(float)
        model_predictions["residual"] = model_predictions["forecast"] - model_predictions["y"].astype(float)
        model_predictions["horizon"] = horizon
        model_predictions = model_predictions.rename(columns={"ds": "target_date", "y": "actual"})
        prediction_rows.append(model_predictions)
    leaderboard = pd.DataFrame(rows).sort_values(["horizon", "mase", "mae"]).reset_index(drop=True)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return leaderboard, predictions


def _compute_event_summary(region_df: pd.DataFrame) -> pd.DataFrame:
    scores = compute_historical_risk_scores(region_df)
    events = label_turning_points(region_df)
    if scores.empty:
        return pd.DataFrame(columns=["signal_type", "signals", "hits", "hit_rate"])
    peak_dates = set(pd.to_datetime(events.loc[events["event"] == "peak", "date"]))
    trough_dates = set(pd.to_datetime(events.loc[events["event"] == "trough", "date"]))
    rows = []
    for score_col, event_dates, signal_type in (
        ("high_score", peak_dates, "high"),
        ("low_score", trough_dates, "low"),
    ):
        signals = scores.loc[scores[score_col] >= 80.0, "date"]
        hits = 0
        for signal_date in pd.to_datetime(signals):
            horizon = pd.date_range(signal_date, periods=9, freq="W-MON")[1:]
            if any(date in event_dates for date in horizon):
                hits += 1
        count = int(signals.shape[0])
        rows.append(
            {
                "signal_type": signal_type,
                "signals": count,
                "hits": hits,
                "hit_rate": float(hits / count) if count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_backtests(region: str, region_df: pd.DataFrame, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> BacktestReport:
    warnings: list[str] = []
    frame = _ensure_weekly_frame(region_df)
    scale = _mase_scale(frame["value"])
    leaderboards: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {"series_length": int(len(frame)), "model_version": MODEL_VERSION}
    for horizon in horizons:
        stats_cv, stats_warnings = _statsforecast_cv(region, frame, horizon)
        warnings.extend(stats_warnings)
        cutoff_values = frame.rename(columns={"date": "cutoff", "value": "origin_value"})
        stats_cv = stats_cv.merge(cutoff_values, on="cutoff", how="left")
        leaderboard, predictions = _build_leaderboard(stats_cv, scale, horizon)
        leaderboards.append(leaderboard)
        prediction_frames.append(predictions)

        ml_cv = _mlforecast_cv(region, frame, horizon)
        if not ml_cv.empty:
            ml_scale = _score_predictions(
                actual=ml_cv["actual"],
                forecast=ml_cv["MLForecast-HGBR"],
                origin_value=ml_cv["origin_value"],
                scale=scale,
            )
            leaderboards.append(pd.DataFrame([{"horizon": horizon, "model": "MLForecast-HGBR", **ml_scale}]))
            prediction_frames.append(
                ml_cv.assign(
                    model="MLForecast-HGBR",
                    forecast=ml_cv["MLForecast-HGBR"],
                    residual=ml_cv["MLForecast-HGBR"] - ml_cv["actual"],
                )[["horizon", "cutoff", "target_date", "actual", "origin_value", "model", "forecast", "residual"]]
            )

    leaderboard = pd.concat(leaderboards, ignore_index=True).sort_values(["horizon", "mase", "mae"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    for horizon in horizons:
        subset = leaderboard.loc[leaderboard["horizon"] == horizon]
        if subset.empty:
            continue
        top_models = tuple(subset.nsmallest(2, "mase")["model"].tolist())
        pred_subset = predictions.loc[(predictions["horizon"] == horizon) & (predictions["model"].isin(top_models))]
        if not pred_subset.empty:
            ensemble = pred_subset.groupby(["horizon", "cutoff", "target_date", "actual", "origin_value"], as_index=False)["forecast"].mean()
            ensemble["residual"] = ensemble["forecast"] - ensemble["actual"]
            predictions = pd.concat(
                [
                    predictions,
                    ensemble.assign(model="Top2Ensemble")[
                        ["horizon", "cutoff", "target_date", "actual", "origin_value", "model", "forecast", "residual"]
                    ],
                ],
                ignore_index=True,
            )
            ensemble_score = _score_predictions(
                actual=ensemble["actual"],
                forecast=ensemble["forecast"],
                origin_value=ensemble["origin_value"],
                scale=scale,
            )
            leaderboard = pd.concat(
                [leaderboard, pd.DataFrame([{"horizon": horizon, "model": "Top2Ensemble", **ensemble_score}])],
                ignore_index=True,
            )
    leaderboard = leaderboard.sort_values(["horizon", "mase", "mae"]).reset_index(drop=True)
    event_summary = _compute_event_summary(frame)
    return BacktestReport(
        region=region,
        leaderboard=leaderboard,
        predictions=predictions.reset_index(drop=True),
        event_summary=event_summary,
        warnings=tuple(dict.fromkeys(warnings)),
        metadata=metadata,
    )


def _baseline_forecasts(frame: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
    values = frame["value"].astype(float).to_numpy()
    last = float(values[-1])
    forecasts = {"Naive": np.repeat(last, horizon)}
    drift = (values[-1] - values[0]) / max(1, len(values) - 1) if len(values) > 1 else 0.0
    forecasts["Drift"] = np.array([last + (drift * step) for step in range(1, horizon + 1)], dtype=float)
    if len(values) >= SEASON_LENGTH:
        seasonal_source = values[-SEASON_LENGTH:]
        forecasts["SeasonalNaive52"] = np.array(
            [seasonal_source[(step - 1) % SEASON_LENGTH] for step in range(1, horizon + 1)],
            dtype=float,
        )
    return forecasts


def _statsforecast_forecast(region: str, region_df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    frame = _ensure_weekly_frame(region_df)
    models, warnings = _build_statsforecast_models(len(frame))
    ts_df = frame.rename(columns={"date": "ds", "value": "y"})
    ts_df["unique_id"] = region
    sf = StatsForecast(models=models, freq="W-MON", n_jobs=1)
    forecast = sf.forecast(df=ts_df[["unique_id", "ds", "y"]], h=horizon)
    return forecast, warnings


def _percentile_from_window(series: pd.Series, invert: bool = False, window: int = 156) -> float:
    sample = series.dropna().tail(window)
    if sample.empty:
        return 50.0
    percentile = _rolling_percentile_last(sample.to_numpy(dtype=float))
    return float(100.0 - percentile if invert else percentile)


def forecast_region(
    region: str,
    region_df: pd.DataFrame,
    *,
    horizon: int,
    backtest_report: BacktestReport | None = None,
) -> ForecastBundle:
    frame = _ensure_weekly_frame(region_df)
    warnings: list[str] = []
    baselines = _baseline_forecasts(frame, horizon)
    stats_forecast, stats_warnings = _statsforecast_forecast(region, frame, horizon)
    warnings.extend(stats_warnings)
    ml_preds, _ = _fit_mlforecast(region, frame, horizon)
    forecast_dates = pd.date_range(frame["date"].iloc[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    model_map: dict[str, np.ndarray] = {name: values for name, values in baselines.items()}
    for column in [col for col in stats_forecast.columns if col not in {"unique_id", "ds"}]:
        model_map[column] = stats_forecast[column].to_numpy(dtype=float)
    model_map["MLForecast-HGBR"] = ml_preds["MLForecast-HGBR-p50"].to_numpy(dtype=float)
    model_map["MLForecast-HGBR-p10"] = ml_preds["MLForecast-HGBR-p10"].to_numpy(dtype=float)
    model_map["MLForecast-HGBR-p90"] = ml_preds["MLForecast-HGBR-p90"].to_numpy(dtype=float)

    residual_q10 = -0.01 * frame["value"].iloc[-1]
    residual_q90 = 0.01 * frame["value"].iloc[-1]
    if backtest_report is not None and not backtest_report.leaderboard.empty:
        subset = backtest_report.leaderboard.loc[backtest_report.leaderboard["horizon"] == horizon]
        top_models = tuple(subset.nsmallest(2, "mase")["model"].tolist()) if not subset.empty else tuple(list(model_map)[:2])
        ensemble_residuals = backtest_report.predictions.loc[
            (backtest_report.predictions["horizon"] == horizon) & (backtest_report.predictions["model"] == "Top2Ensemble"),
            "residual",
        ]
        if not ensemble_residuals.empty:
            residual_q10 = float(ensemble_residuals.quantile(0.1))
            residual_q90 = float(ensemble_residuals.quantile(0.9))
    else:
        preferred = [name for name in ("AutoETS", "AutoARIMA", "MLForecast-HGBR", "Naive") if name in model_map]
        top_models = tuple(preferred[:2] or list(model_map)[:2])
        warnings.append("Backtest results are unavailable, so the forecast uses default top-model fallbacks.")

    available_top = [model for model in top_models if model in model_map]
    if not available_top:
        available_top = [name for name in model_map if not name.endswith(("p10", "p90"))][:2]
    ensemble_point = np.mean([model_map[model] for model in available_top], axis=0)

    forecast_frame = pd.DataFrame({"date": forecast_dates, "p50": ensemble_point, "p10": ensemble_point + residual_q10, "p90": ensemble_point + residual_q90})
    if "MLForecast-HGBR-p10" in model_map:
        forecast_frame["p10"] = np.minimum(forecast_frame["p10"], model_map["MLForecast-HGBR-p10"])
    if "MLForecast-HGBR-p90" in model_map:
        forecast_frame["p90"] = np.maximum(forecast_frame["p90"], model_map["MLForecast-HGBR-p90"])

    current_value = float(frame["value"].iloc[-1])
    model_rows = []
    for model, values in model_map.items():
        if model.endswith(("p10", "p90")):
            continue
        last_forecast = float(values[-1])
        model_rows.append({"model": model, "forecast_last": last_forecast, "return_pct": ((last_forecast / current_value) - 1.0) * 100.0})
    model_table = pd.DataFrame(model_rows).sort_values("return_pct")

    features = build_feature_matrix(frame)
    current_components = {
        "overheat": _percentile_from_window(features["disparity24"]),
        "oversold": _percentile_from_window(features["disparity24"], invert=True),
        "near_high": _percentile_from_window(frame["value"] / frame["value"].rolling(52, min_periods=13).max()),
        "drawdown_depth": _percentile_from_window(-features["drawdown_26"]),
    }
    forecast_return = float((forecast_frame["p50"].iloc[-1] / current_value) - 1.0)
    negative_vote = float(np.mean([values[-1] < current_value for name, values in model_map.items() if not name.endswith(("p10", "p90"))]) * 100.0)
    positive_vote = float(np.mean([values[-1] > current_value for name, values in model_map.items() if not name.endswith(("p10", "p90"))]) * 100.0)
    slope_penalty = min(100.0, max(0.0, -forecast_return * 600.0))
    slope_support = min(100.0, max(0.0, forecast_return * 600.0))
    turning_points = label_turning_points(frame)
    recent_event = turning_points.loc[turning_points["date"] >= frame["date"].iloc[-1] - pd.Timedelta(weeks=12)]
    high_turning = 100.0 if not recent_event.loc[recent_event["event"] == "peak"].empty else 40.0
    low_turning = 100.0 if not recent_event.loc[recent_event["event"] == "trough"].empty else 40.0
    high_risk_score = float(
        np.clip(
            (0.25 * current_components["overheat"])
            + (0.2 * slope_penalty)
            + (0.2 * current_components["near_high"])
            + (0.2 * negative_vote)
            + (0.15 * high_turning),
            0.0,
            100.0,
        )
    )
    low_zone_score = float(
        np.clip(
            (0.25 * current_components["oversold"])
            + (0.2 * slope_support)
            + (0.2 * current_components["drawdown_depth"])
            + (0.2 * positive_vote)
            + (0.15 * low_turning),
            0.0,
            100.0,
        )
    )

    return ForecastBundle(
        region=region,
        history=frame,
        forecast_frame=forecast_frame,
        model_table=model_table,
        top_models=tuple(available_top),
        high_risk_score=round(high_risk_score, 1),
        low_zone_score=round(low_zone_score, 1),
        turning_points=turning_points,
        warnings=tuple(dict.fromkeys(warnings)),
    )
