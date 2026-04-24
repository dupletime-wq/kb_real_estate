from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Callable
import warnings
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


APP_TITLE = "서울 부동산 시세 예측 대시보드"
EXCEL_EPOCH = pd.Timestamp("1899-12-30")
WEEKLY_FREQ = "W-MON"
SEASON_LENGTH = 52
NS_MAIN = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NS_REL = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

SEOUL_NORTH = (
    "강북구",
    "광진구",
    "노원구",
    "도봉구",
    "동대문구",
    "마포구",
    "서대문구",
    "성동구",
    "성북구",
    "용산구",
    "은평구",
    "종로구",
    "중구",
    "중랑구",
)
SEOUL_SOUTH = (
    "강남구",
    "강동구",
    "강서구",
    "관악구",
    "구로구",
    "금천구",
    "동작구",
    "서초구",
    "송파구",
    "양천구",
    "영등포구",
)
SEOUL_GROUPS = ("서울특별시", "강북14개구", "강남11개구")
SEOUL_REGIONS = set(SEOUL_GROUPS + SEOUL_NORTH + SEOUL_SOUTH)
CORE_REGIONS = ("서울특별시", "강북14개구", "강남11개구")
MODEL_ORDER = (
    "Naive",
    "SeasonalNaive52",
    "Drift",
    "HoltLog",
    "ARIMA(1,1,0)-log",
    "ARIMA(0,1,1)-log",
    "ML-HGBR",
)
MODEL_LABELS = {
    "Naive": "Naive",
    "SeasonalNaive52": "Seasonal Naive 52주",
    "Drift": "Drift",
    "HoltLog": "Holt 로그 추세",
    "ARIMA(1,1,0)-log": "ARIMA(1,1,0) 로그",
    "ARIMA(0,1,1)-log": "ARIMA(0,1,1) 로그",
    "ML-HGBR": "Lag ML(HGBR)",
}


@dataclass(frozen=True)
class ParsedWorkbook:
    frames: dict[str, pd.DataFrame]
    sheet_table: pd.DataFrame
    warnings: tuple[str, ...]
    fingerprint: str
    source_name: str


@dataclass(frozen=True)
class BacktestResult:
    leaderboard: pd.DataFrame
    predictions: pd.DataFrame
    selected_models: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class IndicatorResult:
    frame: pd.DataFrame
    passed_summary: pd.DataFrame
    passed_events: pd.DataFrame


@dataclass(frozen=True)
class AnalysisResult:
    region: str
    horizon: int
    history: pd.Series
    forecast: pd.DataFrame
    model_forecasts: pd.DataFrame
    backtest: BacktestResult
    indicators: IndicatorResult


def _col_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for char in letters:
        value = (value * 26) + ord(char.upper()) - 64
    return max(0, value - 1)


def _normalise_sheet_path(target: str) -> str:
    target = target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _load_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root.findall("m:si", NS_MAIN):
        values.append("".join(text.text or "" for text in item.findall(".//m:t", NS_MAIN)))
    return values


def _read_cell(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//m:t", NS_MAIN)).strip()
    value = cell.find("m:v", NS_MAIN)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        idx = int(value.text)
        return shared_strings[idx].strip() if 0 <= idx < len(shared_strings) else value.text.strip()
    return value.text.strip()


def _workbook_sheet_paths(zf: ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: _normalise_sheet_path(rel.attrib["Target"])
        for rel in rels.findall("r:Relationship", NS_REL)
    }
    sheets: list[tuple[str, str]] = []
    for sheet in workbook.findall("m:sheets/m:sheet", NS_MAIN):
        sheet_name = sheet.attrib.get("name", "")
        rel_id = sheet.attrib.get(REL_ID)
        if rel_id in rel_map:
            sheets.append((sheet_name, rel_map[rel_id]))
    return sheets


def _read_sheet_rows(zf: ZipFile, path: str, shared_strings: list[str]) -> list[dict[int, str]]:
    root = ET.fromstring(zf.read(path))
    rows: list[dict[int, str]] = []
    for row in root.findall("m:sheetData/m:row", NS_MAIN):
        values: dict[int, str] = {}
        for cell in row.findall("m:c", NS_MAIN):
            value = _read_cell(cell, shared_strings)
            if value:
                values[_col_to_index(cell.attrib.get("r", "A1"))] = value
        rows.append(values)
    return rows


def _classify_metric(sheet_name: str, rows: list[dict[int, str]]) -> str | None:
    preview = " ".join(value for row in rows[:6] for value in row.values())
    if "매매지수" in sheet_name or "매매가격지수" in preview:
        return "매매지수"
    if "전세지수" in sheet_name or "전세가격지수" in preview:
        return "전세지수"
    return None


def _parse_excel_date(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    numeric = pd.to_numeric(text.replace(",", ""), errors="coerce")
    if pd.notna(numeric):
        number = float(numeric)
        if number > 1000:
            return (EXCEL_EPOCH + pd.to_timedelta(number, unit="D")).normalize()
    parsed = pd.to_datetime(text.replace("/", "-").replace(".", "-"), errors="coerce")
    if pd.notna(parsed):
        return pd.Timestamp(parsed).normalize()
    return None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric):
        return None
    result = float(numeric)
    return result if math.isfinite(result) else None


def _region_group(region: str) -> str:
    if region == "서울특별시":
        return "서울 전체"
    if region == "강북14개구" or region in SEOUL_NORTH:
        return "강북권"
    if region == "강남11개구" or region in SEOUL_SOUTH:
        return "강남권"
    return "기타"


def _parse_index_sheet(rows: list[dict[int, str]], *, metric: str, sheet_name: str) -> tuple[pd.DataFrame, list[str]]:
    warnings_out: list[str] = []
    header_idx = None
    for idx, row in enumerate(rows[:10]):
        values = {str(value).strip() for value in row.values()}
        if "구분" in values and "서울특별시" in values:
            header_idx = idx
            break
    if header_idx is None:
        return pd.DataFrame(), [f"{sheet_name}: 서울 지역 헤더를 찾지 못했습니다."]

    header = rows[header_idx]
    date_col = next((col for col, value in header.items() if value == "구분"), min(header))
    stop_candidates = [
        col
        for col, value in header.items()
        if col > date_col and value in {"6개광역시", "부산광역시", "대구광역시"}
    ]
    stop_col = min(stop_candidates) if stop_candidates else (max(header) + 1)

    selected_cols = {
        col: value
        for col, value in header.items()
        if date_col < col < stop_col and value in SEOUL_REGIONS
    }
    if not selected_cols:
        return pd.DataFrame(), [f"{sheet_name}: 서울 관련 컬럼을 찾지 못했습니다."]

    records: list[dict[str, Any]] = []
    failed_values = 0
    for row in rows[header_idx + 1 :]:
        date = _parse_excel_date(row.get(date_col))
        if date is None:
            continue
        date = date - pd.Timedelta(days=int(date.weekday()))
        for col, region in selected_cols.items():
            value = _parse_float(row.get(col))
            if value is None:
                if row.get(col) not in {None, ""}:
                    failed_values += 1
                continue
            records.append(
                {
                    "date": date,
                    "region": region,
                    "region_group": _region_group(region),
                    "value": value,
                    "metric": metric,
                    "sheet_name": sheet_name,
                }
            )

    if not records:
        return pd.DataFrame(), [f"{sheet_name}: 유효한 숫자 데이터가 없습니다."]

    raw = pd.DataFrame(records)
    frames: list[pd.DataFrame] = []
    for region, region_df in raw.groupby("region", sort=False):
        dedup = region_df.groupby("date", as_index=True)["value"].last().sort_index()
        full_index = pd.date_range(dedup.index.min(), dedup.index.max(), freq=WEEKLY_FREQ)
        expanded = dedup.reindex(full_index)
        missing = expanded.isna()
        if missing.any():
            run_ids = missing.ne(missing.shift(fill_value=False)).cumsum()
            max_gap = int(missing.groupby(run_ids).sum().max())
            if max_gap >= 3:
                warnings_out.append(f"{metric} {region}: 3주 이상 결측 구간이 있어 장기 결측은 보간하지 않았습니다.")
            expanded = expanded.interpolate(method="linear", limit=2, limit_direction="both")
        clean = expanded.dropna().to_frame("value")
        clean.index.name = "date"
        clean = clean.reset_index()
        clean["region"] = region
        clean["region_group"] = _region_group(region)
        clean["metric"] = metric
        clean["sheet_name"] = sheet_name
        clean["is_imputed"] = clean["date"].isin(expanded[missing & expanded.notna()].index)
        frames.append(clean)

    if failed_values:
        warnings_out.append(f"{sheet_name}: 숫자로 변환하지 못한 셀 {failed_values:,}개를 제외했습니다.")

    parsed = pd.concat(frames, ignore_index=True).sort_values(["region", "date"]).reset_index(drop=True)
    return parsed, warnings_out


@st.cache_data(show_spinner=False)
def parse_kb_workbook(file_bytes: bytes, source_name: str) -> ParsedWorkbook:
    fingerprint = hashlib.sha256(file_bytes).hexdigest()
    metric_frames: dict[str, pd.DataFrame] = {}
    sheet_rows: list[dict[str, Any]] = []
    warnings_out: list[str] = []

    with ZipFile(BytesIO(file_bytes)) as zf:
        shared_strings = _load_shared_strings(zf)
        for sheet_name, path in _workbook_sheet_paths(zf):
            rows = _read_sheet_rows(zf, path, shared_strings)
            metric = _classify_metric(sheet_name, rows)
            score = 0
            if metric:
                score += 5
            if any("서울특별시" in row.values() for row in rows[:5]):
                score += 2
            sheet_rows.append({"sheet": sheet_name, "metric": metric or "-", "score": score, "rows": len(rows)})
            if metric in {"매매지수", "전세지수"} and metric not in metric_frames:
                parsed, sheet_warnings = _parse_index_sheet(rows, metric=metric, sheet_name=sheet_name)
                warnings_out.extend(sheet_warnings)
                if not parsed.empty:
                    metric_frames[metric] = parsed

    if not metric_frames:
        raise ValueError("매매지수 또는 전세지수 시트를 찾지 못했습니다. KB 주간시계열 XLSX 형식인지 확인해 주세요.")

    sheet_table = pd.DataFrame(sheet_rows).sort_values(["score", "sheet"], ascending=[False, True]).reset_index(drop=True)
    return ParsedWorkbook(
        frames=metric_frames,
        sheet_table=sheet_table,
        warnings=tuple(dict.fromkeys(warnings_out)),
        fingerprint=fingerprint,
        source_name=source_name,
    )


def _series_for_region(frame: pd.DataFrame, region: str) -> pd.Series:
    region_df = frame.loc[frame["region"] == region, ["date", "value"]].copy()
    if region_df.empty:
        raise ValueError(f"{region} 지역 데이터가 없습니다.")
    series = (
        region_df.assign(date=lambda df: pd.to_datetime(df["date"]).dt.normalize())
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")["value"]
        .astype(float)
    )
    full_index = pd.date_range(series.index.min(), series.index.max(), freq=WEEKLY_FREQ)
    return series.reindex(full_index).interpolate(limit=2, limit_direction="both").dropna()


def _mase_scale(series: pd.Series) -> float:
    diffs = series.diff().abs().dropna()
    if diffs.empty:
        return 1.0
    value = float(diffs.mean())
    return value if value > 0 else 1.0


def _as_positive_array(values: Any, horizon: int) -> np.ndarray | None:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < horizon:
        return None
    array = array[:horizon]
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        return None
    return array


def _forecast_naive(train: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(float(train.iloc[-1]), horizon)


def _forecast_seasonal_naive(train: pd.Series, horizon: int) -> np.ndarray | None:
    if len(train) < SEASON_LENGTH:
        return None
    source = train.iloc[-SEASON_LENGTH:].to_numpy(dtype=float)
    return np.asarray([source[(step - 1) % SEASON_LENGTH] for step in range(1, horizon + 1)], dtype=float)


def _forecast_drift(train: pd.Series, horizon: int) -> np.ndarray:
    values = train.to_numpy(dtype=float)
    drift = (values[-1] - values[0]) / max(1, len(values) - 1)
    return np.asarray([values[-1] + drift * step for step in range(1, horizon + 1)], dtype=float)


def _forecast_holt_log(train: pd.Series, horizon: int) -> np.ndarray | None:
    if len(train) < 80 or train.min() <= 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            np.log(train),
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
        forecast = np.exp(np.asarray(model.forecast(horizon), dtype=float))
    return _as_positive_array(forecast, horizon)


def _forecast_arima_log(train: pd.Series, horizon: int, order: tuple[int, int, int]) -> np.ndarray | None:
    if len(train) < 120 or train.min() <= 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(np.log(train), order=order, trend="t").fit(method_kwargs={"warn_convergence": False, "maxiter": 80})
        forecast = np.exp(np.asarray(model.forecast(horizon), dtype=float))
    return _as_positive_array(forecast, horizon)


def _ml_feature_row(history: np.ndarray) -> list[float]:
    lags = (1, 2, 4, 13, 26, 52)
    features = [float(history[-lag]) for lag in lags]
    for window in (4, 13, 26, 52):
        tail = history[-window:]
        features.extend([float(tail.mean()), float(tail.std(ddof=0))])
    features.extend(
        [
            float(history[-1] - history[-4]),
            float(history[-1] - history[-13]),
            float(history[-1] - history[-26]),
            float(len(history)),
        ]
    )
    return features


def _forecast_ml_hgbr(train: pd.Series, horizon: int) -> np.ndarray | None:
    if len(train) < 160 or train.min() <= 0:
        return None
    log_values = np.log(train.to_numpy(dtype=float))
    rows: list[list[float]] = []
    targets: list[float] = []
    for idx in range(52, len(log_values)):
        rows.append(_ml_feature_row(log_values[:idx]))
        targets.append(float(log_values[idx]))
    if len(rows) < 80:
        return None
    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.045,
        max_leaf_nodes=15,
        l2_regularization=0.03,
        random_state=42,
    )
    model.fit(np.asarray(rows), np.asarray(targets))
    history = log_values.copy()
    predictions: list[float] = []
    for _ in range(horizon):
        pred = float(model.predict([_ml_feature_row(history)])[0])
        predictions.append(pred)
        history = np.append(history, pred)
    return _as_positive_array(np.exp(predictions), horizon)


def _forecast_model(model_name: str, train: pd.Series, horizon: int) -> np.ndarray | None:
    try:
        if model_name == "Naive":
            return _as_positive_array(_forecast_naive(train, horizon), horizon)
        if model_name == "SeasonalNaive52":
            return _as_positive_array(_forecast_seasonal_naive(train, horizon), horizon)
        if model_name == "Drift":
            return _as_positive_array(_forecast_drift(train, horizon), horizon)
        if model_name == "HoltLog":
            return _forecast_holt_log(train, horizon)
        if model_name == "ARIMA(1,1,0)-log":
            return _forecast_arima_log(train, horizon, (1, 1, 0))
        if model_name == "ARIMA(0,1,1)-log":
            return _forecast_arima_log(train, horizon, (0, 1, 1))
        if model_name == "ML-HGBR":
            return _forecast_ml_hgbr(train, horizon)
    except Exception:
        return None
    return None


def _cutoff_positions(series_length: int, horizon: int, max_windows: int) -> list[int]:
    min_train = max(156, 52 + horizon)
    last_cutoff = series_length - horizon
    if last_cutoff <= min_train:
        return [last_cutoff] if last_cutoff > 60 else []
    window_count = min(max_windows, max(1, (last_cutoff - min_train) // max(4, horizon // 2) + 1))
    positions = np.linspace(min_train, last_cutoff, num=window_count, dtype=int)
    return sorted({int(pos) for pos in positions if 60 < pos <= last_cutoff})


def _progress(progress: Callable[[int, int, str], None] | None, done: int, total: int, message: str) -> None:
    if progress is not None:
        progress(done, total, message)


def rolling_backtest(
    series: pd.Series,
    *,
    horizon: int,
    max_windows: int = 4,
    progress: Callable[[int, int, str], None] | None = None,
) -> BacktestResult:
    clean = series.astype(float).dropna()
    warnings_out: list[str] = []
    positions = _cutoff_positions(len(clean), horizon, max_windows)
    if not positions:
        warnings_out.append("데이터가 짧아 rolling-origin 검증 창을 충분히 만들지 못했습니다.")
        return BacktestResult(pd.DataFrame(), pd.DataFrame(), tuple(), tuple(warnings_out))

    total = len(positions) * len(MODEL_ORDER)
    done = 0
    prediction_rows: list[dict[str, Any]] = []
    for position in positions:
        train = clean.iloc[:position]
        actual = clean.iloc[position : position + horizon]
        if len(actual) < horizon:
            continue
        origin = float(train.iloc[-1])
        for model_name in MODEL_ORDER:
            forecast = _forecast_model(model_name, train, horizon)
            done += 1
            _progress(progress, done, total, f"{MODEL_LABELS[model_name]} 검증 중")
            if forecast is None:
                continue
            for step, (date, actual_value, predicted_value) in enumerate(
                zip(actual.index, actual.to_numpy(dtype=float), forecast),
                start=1,
            ):
                prediction_rows.append(
                    {
                        "cutoff": train.index[-1],
                        "target_date": date,
                        "step": step,
                        "origin": origin,
                        "actual": float(actual_value),
                        "forecast": float(predicted_value),
                        "model": model_name,
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty or "Naive" not in set(predictions["model"]):
        warnings_out.append("기준 모델 검증 결과가 없어 모델 선별을 수행하지 못했습니다.")
        return BacktestResult(pd.DataFrame(), predictions, tuple(), tuple(warnings_out))

    scale = _mase_scale(clean)
    base = (
        predictions.loc[predictions["model"] == "Naive", ["cutoff", "step", "target_date", "actual", "forecast"]]
        .rename(columns={"forecast": "naive_forecast"})
        .copy()
    )
    rows: list[dict[str, Any]] = []
    for model_name, group in predictions.groupby("model"):
        merged = group.merge(base, on=["cutoff", "step", "target_date", "actual"], how="inner")
        if merged.empty:
            continue
        error = merged["forecast"] - merged["actual"]
        abs_error = error.abs()
        naive_abs_error = (merged["naive_forecast"] - merged["actual"]).abs()
        mae = float(abs_error.mean())
        naive_mae = float(naive_abs_error.mean())
        wape = float(abs_error.sum() / merged["actual"].abs().sum())
        rmse = float(np.sqrt(np.square(error).mean()))
        direction_accuracy = float(
            (
                np.sign(merged["forecast"] - merged["origin"])
                == np.sign(merged["actual"] - merged["origin"])
            ).mean()
        )
        if model_name == "Naive":
            p_value = np.nan
            improvement_pct = 0.0
        else:
            improvement_pct = float((1.0 - (mae / naive_mae)) * 100.0) if naive_mae > 0 else 0.0
            try:
                p_value = float(stats.ttest_rel(naive_abs_error, abs_error, alternative="greater").pvalue)
            except TypeError:
                p_value = float(stats.ttest_rel(naive_abs_error, abs_error).pvalue / 2.0)
        rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "mae": mae,
                "rmse": rmse,
                "mase": float(mae / scale),
                "wape_pct": wape * 100.0,
                "direction_accuracy_pct": direction_accuracy * 100.0,
                "naive_improvement_pct": improvement_pct,
                "p_value_vs_naive": p_value,
                "n_predictions": int(len(merged)),
            }
        )

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        warnings_out.append("모델 성능표를 만들 수 없습니다.")
        return BacktestResult(leaderboard, predictions, tuple(), tuple(warnings_out))

    leaderboard = leaderboard.sort_values(["mase", "wape_pct"]).reset_index(drop=True)
    leaderboard["passed"] = (
        (leaderboard["model"] != "Naive")
        & (leaderboard["naive_improvement_pct"] > 0.0)
        & (leaderboard["p_value_vs_naive"].fillna(1.0) <= 0.25)
        & (leaderboard["direction_accuracy_pct"] >= 45.0)
    )
    selected = tuple(leaderboard.loc[leaderboard["passed"], "model"].head(3).tolist())
    if not selected:
        fallback = str(leaderboard.iloc[0]["model"])
        selected = (fallback,)
        warnings_out.append("Naive 대비 통계적으로 우수한 모델이 없어 최저 오류 모델 1개만 예측에 사용했습니다.")

    return BacktestResult(
        leaderboard=leaderboard,
        predictions=predictions,
        selected_models=selected,
        warnings=tuple(dict.fromkeys(warnings_out)),
    )


def _forecast_full_models(series: pd.Series, horizon: int) -> pd.DataFrame:
    dates = pd.date_range(series.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq=WEEKLY_FREQ)
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        forecast = _forecast_model(model_name, series, horizon)
        if forecast is None:
            continue
        for date, value in zip(dates, forecast):
            rows.append({"date": date, "model": model_name, "forecast": float(value)})
    return pd.DataFrame(rows)


def _forecast_interval(
    point_forecast: np.ndarray,
    model_forecasts: pd.DataFrame,
    backtest: BacktestResult,
    selected_models: tuple[str, ...],
    series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    selected_predictions = backtest.predictions.loc[backtest.predictions["model"].isin(selected_models)].copy()
    if not selected_predictions.empty:
        selected_predictions["relative_error"] = (
            selected_predictions["actual"] - selected_predictions["forecast"]
        ) / selected_predictions["forecast"].replace(0, np.nan)
        q10, q90 = selected_predictions["relative_error"].dropna().quantile([0.10, 0.90]).to_numpy(dtype=float)
    else:
        q10, q90 = np.nan, np.nan
    weekly_vol = float(series.pct_change().dropna().tail(156).std())
    if not math.isfinite(weekly_vol) or weekly_vol <= 0:
        weekly_vol = 0.002
    steps = np.arange(1, len(point_forecast) + 1, dtype=float)
    fallback_band = 1.64 * weekly_vol * np.sqrt(steps)
    if not math.isfinite(q10) or q10 >= 0:
        lower = point_forecast * (1.0 - fallback_band)
    else:
        lower = point_forecast * (1.0 + q10 - (fallback_band * 0.25))
    if not math.isfinite(q90) or q90 <= 0:
        upper = point_forecast * (1.0 + fallback_band)
    else:
        upper = point_forecast * (1.0 + q90 + (fallback_band * 0.25))

    if not model_forecasts.empty:
        spread = model_forecasts.groupby("date")["forecast"].agg(["min", "max"]).reset_index()
        lower = np.minimum(lower, spread["min"].to_numpy(dtype=float))
        upper = np.maximum(upper, spread["max"].to_numpy(dtype=float))
    return np.minimum(lower, point_forecast), np.maximum(upper, point_forecast)


def build_forecast_frame(series: pd.Series, horizon: int, backtest: BacktestResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_forecasts = _forecast_full_models(series, horizon)
    if model_forecasts.empty:
        dates = pd.date_range(series.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq=WEEKLY_FREQ)
        point = _forecast_naive(series, horizon)
        model_forecasts = pd.DataFrame({"date": dates, "model": "Naive", "forecast": point})

    available = set(model_forecasts["model"])
    selected_models = tuple(model for model in backtest.selected_models if model in available)
    if not selected_models:
        selected_models = (str(model_forecasts.groupby("model")["forecast"].last().index[0]),)
    selected_frame = model_forecasts.loc[model_forecasts["model"].isin(selected_models)]
    point_frame = selected_frame.groupby("date", as_index=False)["forecast"].mean().rename(columns={"forecast": "p50"})
    point = point_frame["p50"].to_numpy(dtype=float)
    lower, upper = _forecast_interval(point, model_forecasts, backtest, selected_models, series)
    point_frame["p10"] = lower
    point_frame["p90"] = upper
    point_frame["selected_models"] = ", ".join(MODEL_LABELS.get(model, model) for model in selected_models)
    return point_frame, model_forecasts


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


def run_region_analysis(
    frame: pd.DataFrame,
    region: str,
    horizon: int,
    *,
    max_windows: int = 4,
    progress: Callable[[int, int, str], None] | None = None,
) -> AnalysisResult:
    series = _series_for_region(frame, region)
    backtest = rolling_backtest(series, horizon=horizon, max_windows=max_windows, progress=progress)
    forecast, model_forecasts = build_forecast_frame(series, horizon, backtest)
    indicators = evaluate_indicators(series, horizon)
    return AnalysisResult(
        region=region,
        horizon=horizon,
        history=series,
        forecast=forecast,
        model_forecasts=model_forecasts,
        backtest=backtest,
        indicators=indicators,
    )


def _format_pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}%"


def _format_value(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def _available_regions(frame: pd.DataFrame, group: str) -> list[str]:
    regions = list(frame["region"].drop_duplicates())
    if group == "서울 전체":
        ordered = ["서울특별시"]
    elif group == "강북권":
        ordered = ["강북14개구", *SEOUL_NORTH]
    elif group == "강남권":
        ordered = ["강남11개구", *SEOUL_SOUTH]
    else:
        ordered = [*SEOUL_NORTH, *SEOUL_SOUTH]
    return [region for region in ordered if region in regions]


def make_forecast_chart(result: AnalysisResult) -> go.Figure:
    history = result.history.tail(420)
    forecast = result.forecast
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="실제 지수",
            line=dict(color="#263238", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["p90"],
            mode="lines",
            name="p90",
            line=dict(color="rgba(30, 136, 229, 0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["p10"],
            mode="lines",
            name="p10-p90",
            fill="tonexty",
            fillcolor="rgba(30, 136, 229, 0.16)",
            line=dict(color="rgba(30, 136, 229, 0)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["p50"],
            mode="lines+markers",
            name="예측 p50",
            line=dict(color="#1e88e5", width=3),
        )
    )
    fig.update_layout(
        height=470,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        yaxis_title="KB 가격지수",
    )
    return fig


def make_model_forecast_chart(result: AnalysisResult) -> go.Figure:
    frame = result.model_forecasts.copy()
    frame["model_label"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    fig = go.Figure()
    for label, group in frame.groupby("model_label"):
        fig.add_trace(go.Scatter(x=group["date"], y=group["forecast"], mode="lines", name=label))
    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        yaxis_title="모델별 예측 지수",
    )
    return fig


def make_comparison_chart(frame: pd.DataFrame) -> go.Figure:
    rows: list[dict[str, Any]] = []
    for region in CORE_REGIONS:
        series = _series_for_region(frame, region)
        current = float(series.iloc[-1])
        rows.append(
            {
                "region": region,
                "13주 변화율": ((current / float(series.iloc[-14])) - 1.0) * 100.0 if len(series) > 14 else np.nan,
                "52주 변화율": ((current / float(series.iloc[-53])) - 1.0) * 100.0 if len(series) > 53 else np.nan,
            }
        )
    compare = pd.DataFrame(rows).melt(id_vars="region", var_name="period", value_name="return_pct")
    fig = go.Figure()
    colors = {"13주 변화율": "#00897b", "52주 변화율": "#f9a825"}
    for period, group in compare.groupby("period"):
        fig.add_trace(
            go.Bar(
                x=group["region"],
                y=group["return_pct"],
                name=period,
                marker_color=colors.get(period, "#546e7a"),
                text=[_format_pct(value, 2) for value in group["return_pct"]],
                textposition="outside",
            )
        )
    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=30, b=20),
        barmode="group",
        yaxis_title="변화율",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_technical_chart(result: AnalysisResult) -> go.Figure:
    indicators = result.indicators.frame.tail(420)
    passed = result.indicators.passed_summary
    events = result.indicators.passed_events.copy()
    active_indicators = set(passed["indicator"]) if not passed.empty else set()
    show_macd = "MACD" in active_indicators
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )
    fig.add_trace(
        go.Scatter(x=indicators["date"], y=indicators["value"], name="지수", line=dict(color="#263238", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=indicators["date"], y=indicators["ma13"], name="MA13", line=dict(color="#7e57c2", width=1)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=indicators["date"], y=indicators["ma26"], name="MA26", line=dict(color="#5d4037", width=1)),
        row=1,
        col=1,
    )
    if "Bollinger" in active_indicators:
        fig.add_trace(
            go.Scatter(
                x=indicators["date"],
                y=indicators["bb_upper"],
                name="Bollinger 상단",
                line=dict(color="rgba(0,137,123,0.45)", width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=indicators["date"],
                y=indicators["bb_lower"],
                name="Bollinger 하단",
                line=dict(color="rgba(0,137,123,0.45)", width=1),
                fill="tonexty",
                fillcolor="rgba(0,137,123,0.08)",
            ),
            row=1,
            col=1,
        )
    if "Ichimoku" in active_indicators:
        fig.add_trace(
            go.Scatter(x=indicators["date"], y=indicators["span_a"], name="Ichimoku Span A", line=dict(color="#ef6c00", width=1)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=indicators["date"], y=indicators["span_b"], name="Ichimoku Span B", line=dict(color="#6d4c41", width=1)),
            row=1,
            col=1,
        )
    if not events.empty:
        events = events.loc[events["date"] >= indicators["date"].min()]
        bullish = events.loc[events["direction"] == "bullish"]
        bearish = events.loc[events["direction"] == "bearish"]
        if not bullish.empty:
            fig.add_trace(
                go.Scatter(
                    x=bullish["date"],
                    y=bullish["value"],
                    mode="markers",
                    name="통과 상승 신호",
                    marker=dict(symbol="triangle-up", color="#2e7d32", size=9),
                    text=bullish["indicator"] + " " + bullish["signal"],
                ),
                row=1,
                col=1,
            )
        if not bearish.empty:
            fig.add_trace(
                go.Scatter(
                    x=bearish["date"],
                    y=bearish["value"],
                    mode="markers",
                    name="통과 경계 신호",
                    marker=dict(symbol="triangle-down", color="#c62828", size=9),
                    text=bearish["indicator"] + " " + bearish["signal"],
                ),
                row=1,
                col=1,
            )
    if show_macd:
        fig.add_trace(
            go.Scatter(x=indicators["date"], y=indicators["macd"], name="MACD", line=dict(color="#1565c0", width=1.5)),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=indicators["date"],
                y=indicators["macd_signal"],
                name="MACD Signal",
                line=dict(color="#ef6c00", width=1),
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="KB 가격지수", row=1, col=1)
    if show_macd:
        fig.update_yaxes(title_text="MACD", row=2, col=1)
    else:
        fig.update_yaxes(visible=False, row=2, col=1)
    return fig


def _style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        .app-note {
            padding: 0.75rem 0.9rem;
            border: 1px solid #d7dee8;
            border-radius: 8px;
            background: #f8fafc;
            color: #334155;
            font-size: 0.92rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _data_summary(parsed: ParsedWorkbook, frame: pd.DataFrame, selected_region: str) -> None:
    series = _series_for_region(frame, selected_region)
    latest = float(series.iloc[-1])
    previous_13 = float(series.iloc[-14]) if len(series) > 14 else np.nan
    previous_52 = float(series.iloc[-53]) if len(series) > 53 else np.nan
    cols = st.columns(4)
    cols[0].metric("데이터 기간", f"{series.index.min().date()} ~ {series.index.max().date()}")
    cols[1].metric("서울 지역 수", f"{frame['region'].nunique():,}개")
    cols[2].metric("최근 지수", _format_value(latest, 2))
    cols[3].metric(
        "13주 변화율",
        _format_pct(((latest / previous_13) - 1.0) * 100.0, 2) if pd.notna(previous_13) else "-",
        delta=_format_pct(((latest / previous_52) - 1.0) * 100.0, 2) if pd.notna(previous_52) else None,
        help="큰 숫자는 13주 변화율, 델타는 52주 변화율입니다.",
    )
    st.caption(f"파일 지문: `{parsed.fingerprint[:12]}` · 원본: `{parsed.source_name}`")


def _render_result(result: AnalysisResult) -> None:
    current = float(result.history.iloc[-1])
    forecast_last = float(result.forecast["p50"].iloc[-1])
    forecast_return = ((forecast_last / current) - 1.0) * 100.0
    selected_labels = ", ".join(MODEL_LABELS.get(model, model) for model in result.backtest.selected_models)
    cols = st.columns(4)
    cols[0].metric("선택 지역", result.region)
    cols[1].metric(f"{result.horizon}주 예측 지수", _format_value(forecast_last, 2), delta=_format_pct(forecast_return, 2))
    cols[2].metric("선택 모델", selected_labels or "-")
    cols[3].metric("검증 통과 지표", f"{len(result.indicators.passed_summary):,}개")

    if result.backtest.warnings:
        for message in result.backtest.warnings:
            st.warning(message)

    tab_forecast, tab_models, tab_indicators = st.tabs(["예측", "모델 검증", "기술지표"])
    with tab_forecast:
        st.plotly_chart(make_forecast_chart(result), use_container_width=True)
        with st.expander("모델별 예측선 보기", expanded=False):
            st.plotly_chart(make_model_forecast_chart(result), use_container_width=True)
    with tab_models:
        leaderboard = result.backtest.leaderboard.copy()
        if leaderboard.empty:
            st.info("모델 검증 결과가 없습니다.")
        else:
            display = leaderboard[
                [
                    "model_label",
                    "passed",
                    "mase",
                    "wape_pct",
                    "rmse",
                    "direction_accuracy_pct",
                    "naive_improvement_pct",
                    "p_value_vs_naive",
                    "n_predictions",
                ]
            ].rename(
                columns={
                    "model_label": "모델",
                    "passed": "사용",
                    "mase": "MASE",
                    "wape_pct": "WAPE(%)",
                    "rmse": "RMSE",
                    "direction_accuracy_pct": "방향 적중률(%)",
                    "naive_improvement_pct": "Naive 대비 개선(%)",
                    "p_value_vs_naive": "p-value",
                    "n_predictions": "검증 예측 수",
                }
            )
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "MASE": st.column_config.NumberColumn(format="%.3f"),
                    "WAPE(%)": st.column_config.NumberColumn(format="%.2f"),
                    "RMSE": st.column_config.NumberColumn(format="%.3f"),
                    "방향 적중률(%)": st.column_config.NumberColumn(format="%.1f"),
                    "Naive 대비 개선(%)": st.column_config.NumberColumn(format="%.2f"),
                    "p-value": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            st.caption("`사용=True` 모델만 예측 앙상블에 들어갑니다. 통과 모델이 없으면 최저 오류 모델 1개만 사용합니다.")
    with tab_indicators:
        if result.indicators.passed_summary.empty:
            st.info("이번 데이터와 검증 기준에서 통과한 기술지표 신호가 없습니다. 차트에는 가격 흐름과 보조 이동평균만 표시합니다.")
        else:
            st.dataframe(
                result.indicators.passed_summary.drop(columns=["passed"]).rename(
                    columns={
                        "indicator": "지표",
                        "signal": "신호",
                        "direction": "검증 방향",
                        "samples": "표본 수",
                        "hit_rate_pct": "적중률(%)",
                        "mean_forward_return_pct": "평균 선행수익률(%)",
                        "p_value": "p-value",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "적중률(%)": st.column_config.NumberColumn(format="%.1f"),
                    "평균 선행수익률(%)": st.column_config.NumberColumn(format="%.3f"),
                    "p-value": st.column_config.NumberColumn(format="%.4f"),
                },
            )
        st.plotly_chart(make_technical_chart(result), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="KB", layout="wide")
    _style()
    st.title(APP_TITLE)
    st.markdown(
        """
        <div class="app-note">
        KB 주간시계열 XLSX를 기준으로 서울 아파트 가격지수를 예측합니다.
        결과는 통계 모델의 과거 검증에 기반한 분석 도구이며 투자 권유가 아닙니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("입력")
        uploaded_file = st.file_uploader("KB 주간시계열 XLSX 업로드", type=["xlsx"])
        use_sample = False
        if uploaded_file is None:
            try:
                sample_path = next(Path(".").glob("*_주간시계열.xlsx"))
                use_sample = st.checkbox("로컬 샘플 파일 사용", value=False)
            except StopIteration:
                sample_path = None
        else:
            sample_path = None

    if uploaded_file is None and not use_sample:
        st.info("왼쪽에서 `20260420_주간시계열.xlsx`와 같은 KB 주간시계열 파일을 업로드하세요.")
        return

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        source_name = uploaded_file.name
    elif sample_path is not None:
        file_bytes = sample_path.read_bytes()
        source_name = sample_path.name
    else:
        st.error("읽을 XLSX 파일이 없습니다.")
        return

    try:
        parsed = parse_kb_workbook(file_bytes, source_name)
    except Exception as exc:
        st.error(f"파일을 해석하지 못했습니다: {exc}")
        return

    if parsed.warnings:
        with st.expander("데이터 경고", expanded=False):
            for message in parsed.warnings:
                st.warning(message)

    with st.sidebar:
        st.header("분석 설정")
        metric_options = [metric for metric in ("매매지수", "전세지수") if metric in parsed.frames]
        metric = st.selectbox("분석 지표", metric_options, index=0)
        horizon = st.selectbox("예측 기간", [13, 26, 52], index=1, format_func=lambda value: f"{value}주")
        group = st.selectbox("지역 그룹", ["서울 전체", "강북권", "강남권", "개별 구"], index=0)
        frame = parsed.frames[metric]
        region_options = _available_regions(frame, group)
        if not region_options:
            st.error("선택 가능한 지역이 없습니다.")
            return
        region = st.selectbox("지역", region_options, index=0)
        run_button = st.button("예측 실행", type="primary", use_container_width=True)

    _data_summary(parsed, frame, region)
    st.subheader("서울 권역 최근 변화율")
    st.plotly_chart(make_comparison_chart(frame), use_container_width=True)

    result_key = (parsed.fingerprint, metric, region, horizon)
    if "analysis_cache" not in st.session_state:
        st.session_state["analysis_cache"] = {}

    if run_button:
        with st.status("예측과 검증을 실행하는 중입니다.", expanded=True) as status:
            progress_bar = st.progress(0, text="준비 중")

            def progress_callback(done: int, total: int, message: str) -> None:
                ratio = 0.0 if total <= 0 else min(1.0, done / total)
                progress_bar.progress(ratio, text=message)

            result = run_region_analysis(frame, region, horizon, max_windows=4, progress=progress_callback)
            st.session_state["analysis_cache"][result_key] = result
            progress_bar.progress(1.0, text="완료")
            status.update(label="예측과 검증이 완료되었습니다.", state="complete", expanded=False)

    cached_result = st.session_state["analysis_cache"].get(result_key)
    if cached_result is None:
        st.info("설정을 확인한 뒤 `예측 실행`을 누르면 모델 검증과 예측 차트가 생성됩니다.")
        with st.expander("감지된 시트", expanded=False):
            st.dataframe(parsed.sheet_table, use_container_width=True, hide_index=True)
        return

    st.subheader(f"{region} {metric} 예측 결과")
    _render_result(cached_result)


if __name__ == "__main__":
    main()
