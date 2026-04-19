from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

if __package__ in {None, ""}:
    current_path = Path(__file__).resolve()
    current_dir = current_path.parent
    parent_dir = current_dir.parent
    for candidate in (current_dir, parent_dir):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    try:
        from ingestion import QAReport, load_kb_workbook, workbook_fingerprint
        from modeling import (
            DEFAULT_FEATURE_SCOPE,
            DEFAULT_HORIZONS,
            MODEL_VERSION,
            BacktestReport,
            build_feature_matrix,
            compute_historical_risk_scores,
            forecast_region,
            join_macro_asof,
            run_backtests,
        )
    except ImportError:
        from kb_real_estate_app.ingestion import QAReport, load_kb_workbook, workbook_fingerprint
        from kb_real_estate_app.modeling import (
            DEFAULT_FEATURE_SCOPE,
            DEFAULT_HORIZONS,
            MODEL_VERSION,
            BacktestReport,
            build_feature_matrix,
            compute_historical_risk_scores,
            forecast_region,
            join_macro_asof,
            run_backtests,
        )
else:
    from .ingestion import QAReport, load_kb_workbook, workbook_fingerprint
    from .modeling import (
        DEFAULT_FEATURE_SCOPE,
        DEFAULT_HORIZONS,
        MODEL_VERSION,
        BacktestReport,
        build_feature_matrix,
        compute_historical_risk_scores,
        forecast_region,
        join_macro_asof,
        run_backtests,
    )


APP_TITLE = "KB Weekly Real Estate Forecast Lab"
DEFAULT_REGIONS = ("전국", "강남11개구", "강북14개구", "6개광역시", "성남시")
SESSION_KEYS = (
    "workbook_fingerprint",
    "normalized_df",
    "qa_report",
    "macro_df",
    "forecast_cache_ref",
    "backtest_cache_ref",
    "selected_region",
    "selected_horizon",
)


def configure_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="KB", layout="wide")


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .hero {
            padding: 1.4rem 1.5rem;
            border-radius: 20px;
            color: #f8fafc;
            background:
                radial-gradient(circle at top right, rgba(251, 191, 36, 0.35), transparent 30%),
                linear-gradient(135deg, #111827 0%, #0f766e 45%, #f59e0b 100%);
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0; font-size: 2.1rem; line-height: 1.06; }
        .hero p { margin: 0.55rem 0 0; max-width: 72rem; color: rgba(248,250,252,0.9); }
        .pill {
            display: inline-block;
            padding: 0.28rem 0.62rem;
            margin-right: 0.45rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.15);
            font-size: 0.84rem;
            font-weight: 600;
        }
        .metric-card {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            min-height: 106px;
        }
        .metric-card .label { color: #475569; font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.04em; }
        .metric-card .value { margin-top: 0.2rem; color: #0f172a; font-weight: 700; font-size: 1.7rem; }
        .metric-card .caption { margin-top: 0.25rem; color: #475569; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    for key in SESSION_KEYS:
        st.session_state.setdefault(key, None)
    st.session_state.setdefault("feature_scope", DEFAULT_FEATURE_SCOPE)


def _safe_secrets_get(key: str) -> str | None:
    try:
        return st.secrets.get(key)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_kb_workbook_cached(file_bytes: bytes, source_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized, report = load_kb_workbook(file_bytes, source_name)
    return normalized, report.to_dict()


@st.cache_data(show_spinner=False)
def load_macro_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    macro_df = pd.read_csv(BytesIO(file_bytes))
    if "date" not in macro_df.columns:
        macro_df = macro_df.rename(columns={macro_df.columns[0]: "date"})
    macro_df["date"] = pd.to_datetime(macro_df["date"], errors="coerce")
    macro_df = macro_df.dropna(subset=["date"]).sort_values("date")
    return macro_df.reset_index(drop=True)


def _fetch_ecos_series(api_key: str, stat_code: str, cycle: str, item_code: str) -> pd.DataFrame:
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/500/"
        f"{stat_code}/{cycle}/201001/203001/{item_code}"
    )
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("StatisticSearch", {}).get("row", [])
    if not rows:
        return pd.DataFrame(columns=["date", item_code])
    frame = pd.DataFrame(rows)
    date_col = "TIME"
    value_col = "DATA_VALUE"
    if cycle == "Q":
        frame["date"] = pd.PeriodIndex(frame[date_col], freq="Q").to_timestamp(how="end")
    else:
        frame["date"] = pd.to_datetime(frame[date_col], format="%Y%m", errors="coerce")
    frame[item_code] = pd.to_numeric(frame[value_col], errors="coerce")
    return frame[["date", item_code]].dropna()


@st.cache_data(show_spinner=False)
def load_default_macro_frame(api_key: str | None) -> tuple[pd.DataFrame, str]:
    if not api_key:
        return pd.DataFrame(), "ECOS API key is unavailable, so macro features were disabled."
    try:
        series_specs = (
            ("722Y001", "M", "0101000", "base_rate"),
            ("817Y002", "M", "010150000", "m2"),
        )
        merged: pd.DataFrame | None = None
        for stat_code, cycle, item_code, alias in series_specs:
            frame = _fetch_ecos_series(api_key, stat_code, cycle, item_code).rename(columns={item_code: alias})
            merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
        return (merged.sort_values("date").reset_index(drop=True) if merged is not None else pd.DataFrame(), "")
    except Exception as exc:
        return pd.DataFrame(), f"ECOS macro fetch failed: {exc}"


@st.cache_data(show_spinner=False)
def run_backtests_cached(
    normalized_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    region: str,
    feature_scope: str,
    fingerprint: str,
    model_version: str,
) -> dict[str, Any]:
    del fingerprint, model_version
    region_df = normalized_df.loc[normalized_df["region"] == region, ["date", "value"]].copy()
    if feature_scope == "KB+Macro" and not macro_df.empty:
        macro_features = join_macro_asof(region_df, macro_df)
        region_df = region_df.merge(macro_features, on="date", how="left")
    report = run_backtests(region, region_df[["date", "value"]])
    return {
        "leaderboard": report.leaderboard,
        "predictions": report.predictions,
        "event_summary": report.event_summary,
        "warnings": list(report.warnings),
        "metadata": report.metadata,
    }


@st.cache_data(show_spinner=False)
def forecast_region_cached(
    normalized_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    region: str,
    horizon: int,
    feature_scope: str,
    fingerprint: str,
    model_version: str,
    backtest_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    del fingerprint, model_version
    region_df = normalized_df.loc[normalized_df["region"] == region, ["date", "value"]].copy()
    if feature_scope == "KB+Macro" and not macro_df.empty:
        macro_features = join_macro_asof(region_df, macro_df)
        region_df = region_df.merge(macro_features, on="date", how="left")
    report = None
    if backtest_payload:
        report = BacktestReport(
            region=region,
            leaderboard=backtest_payload["leaderboard"],
            predictions=backtest_payload["predictions"],
            event_summary=backtest_payload["event_summary"],
            warnings=tuple(backtest_payload.get("warnings", [])),
            metadata=backtest_payload.get("metadata", {}),
        )
    bundle = forecast_region(region, region_df[["date", "value"]], horizon=horizon, backtest_report=report)
    return {
        "history": bundle.history,
        "forecast_frame": bundle.forecast_frame,
        "model_table": bundle.model_table,
        "top_models": list(bundle.top_models),
        "high_risk_score": bundle.high_risk_score,
        "low_zone_score": bundle.low_zone_score,
        "turning_points": bundle.turning_points,
        "warnings": list(bundle.warnings),
    }


def _preferred_region_order(regions: list[str]) -> list[str]:
    seen = {region for region in regions}
    ordered = [region for region in DEFAULT_REGIONS if region in seen]
    ordered.extend(sorted(region for region in regions if region not in DEFAULT_REGIONS))
    return ordered


def _render_metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _forecast_chart(payload: dict[str, Any]) -> go.Figure:
    history = payload["history"]
    forecast = payload["forecast_frame"]
    turning_points = payload["turning_points"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history["value"], mode="lines", name="History", line=dict(color="#111827", width=2)))
    fig.add_trace(
        go.Scatter(x=forecast["date"], y=forecast["p90"], mode="lines", line=dict(color="rgba(20,184,166,0.0)"), showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["p10"],
            mode="lines",
            line=dict(color="rgba(20,184,166,0.0)"),
            fill="tonexty",
            fillcolor="rgba(20,184,166,0.18)",
            name="Empirical band",
        )
    )
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["p50"], mode="lines+markers", name="Forecast p50", line=dict(color="#0f766e", width=2)))
    if not turning_points.empty:
        peaks = turning_points.loc[turning_points["event"] == "peak"]
        troughs = turning_points.loc[turning_points["event"] == "trough"]
        if not peaks.empty:
            fig.add_trace(
                go.Scatter(
                    x=peaks["date"],
                    y=peaks["value"],
                    mode="markers",
                    marker=dict(color="#dc2626", size=10, symbol="triangle-down"),
                    name="Peak label",
                )
            )
        if not troughs.empty:
            fig.add_trace(
                go.Scatter(
                    x=troughs["date"],
                    y=troughs["value"],
                    mode="markers",
                    marker=dict(color="#2563eb", size=10, symbol="triangle-up"),
                    name="Trough label",
                )
            )
    fig.update_layout(height=520, margin=dict(l=18, r=18, t=30, b=18), legend=dict(orientation="h"), title="Forecast Fan Chart")
    return fig


def _technical_overlay_chart(region_df: pd.DataFrame) -> go.Figure:
    features = build_feature_matrix(region_df)
    scores = compute_historical_risk_scores(region_df)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.45, 0.18, 0.18, 0.19])
    fig.add_trace(go.Scatter(x=features["date"], y=features["y"], name="Price index", line=dict(color="#111827", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=features["date"], y=features["ma6"], name="MA6", line=dict(color="#2563eb")), row=1, col=1)
    fig.add_trace(go.Scatter(x=features["date"], y=features["ma24"], name="MA24", line=dict(color="#f97316", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=features["date"], y=features["macd"], name="MACD", line=dict(color="#dc2626")), row=2, col=1)
    fig.add_trace(go.Scatter(x=features["date"], y=features["macd_signal"], name="MACD signal", line=dict(color="#0ea5e9")), row=2, col=1)
    fig.add_trace(go.Scatter(x=features["date"], y=features["disparity24"], name="Disparity24", line=dict(color="#16a34a")), row=3, col=1)
    fig.add_trace(go.Scatter(x=features["date"], y=features["trix"], name="TRIX", line=dict(color="#7c3aed")), row=4, col=1)
    fig.add_trace(go.Scatter(x=scores["date"], y=scores["high_score"], name="High risk", line=dict(color="#b91c1c", dash="dot")), row=4, col=1)
    fig.add_trace(go.Scatter(x=scores["date"], y=scores["low_score"], name="Low zone", line=dict(color="#1d4ed8", dash="dot")), row=4, col=1)
    fig.update_layout(height=920, margin=dict(l=18, r=18, t=30, b=18), legend=dict(orientation="h"))
    return fig


def _load_uploaded_inputs() -> None:
    workbook_file = st.file_uploader("Upload KB weekly workbook", type=["xlsx", "xlsm", "xls"])
    macro_file = st.file_uploader("Optional macro CSV", type=["csv"])
    feature_scope = st.radio("Feature scope", ["KB+Macro", "KB Only"], horizontal=True, index=0)
    st.session_state["feature_scope"] = feature_scope
    if workbook_file is None:
        return

    file_bytes = workbook_file.getvalue()
    fingerprint = workbook_fingerprint(file_bytes)
    if st.session_state["workbook_fingerprint"] != fingerprint:
        normalized_df, qa_payload = load_kb_workbook_cached(file_bytes, workbook_file.name)
        st.session_state["workbook_fingerprint"] = fingerprint
        st.session_state["normalized_df"] = normalized_df
        st.session_state["qa_report"] = qa_payload
        st.session_state["forecast_cache_ref"] = {}
        st.session_state["backtest_cache_ref"] = {}

    macro_df = pd.DataFrame()
    macro_warning = ""
    if macro_file is not None and feature_scope == "KB+Macro":
        macro_df = load_macro_csv_cached(macro_file.getvalue())
    elif feature_scope == "KB+Macro":
        macro_df, macro_warning = load_default_macro_frame(_safe_secrets_get("ECOS_API_KEY"))
    st.session_state["macro_df"] = macro_df
    if macro_warning:
        st.info(macro_warning)


def _current_qa_report() -> QAReport | None:
    payload = st.session_state.get("qa_report")
    if not payload:
        return None
    return QAReport(
        blocking_errors=tuple(payload.get("blocking_errors", [])),
        warnings=tuple(payload.get("warnings", [])),
        stats=payload.get("stats", {}),
        detected_layout=payload.get("detected_layout", {}),
    )


def render_data_qa_tab() -> None:
    _load_uploaded_inputs()
    normalized_df = st.session_state.get("normalized_df")
    qa_report = _current_qa_report()
    if qa_report is None:
        st.info("Upload a KB weekly workbook to start the QA pipeline.")
        return
    card_cols = st.columns(4)
    with card_cols[0]:
        _render_metric_card("Blocking errors", str(len(qa_report.blocking_errors)), "The forecast engine stops when this is non-zero.")
    with card_cols[1]:
        _render_metric_card("Warnings", str(len(qa_report.warnings)), "Short gaps and degraded macro/seasonality fall here.")
    with card_cols[2]:
        _render_metric_card("Regions", str(qa_report.stats.get("regions", 0)), "Distinct normalized regional series.")
    with card_cols[3]:
        _render_metric_card("Weeks", str(qa_report.stats.get("unique_weeks", 0)), "Distinct W-MON observations after reindexing.")

    if qa_report.blocking_errors:
        for message in qa_report.blocking_errors:
            st.error(message)
    if qa_report.warnings:
        for message in qa_report.warnings:
            st.warning(message)

    st.markdown("### QA Summary")
    st.json(qa_report.to_dict())
    if normalized_df is not None and not normalized_df.empty:
        st.markdown("### Normalized Preview")
        st.dataframe(normalized_df.head(40), use_container_width=True)
        csv_bytes = normalized_df.to_csv(index=False).encode("utf-8-sig")
        parquet_buffer = BytesIO()
        normalized_df.to_parquet(parquet_buffer, index=False)
        st.download_button("Download normalized CSV", csv_bytes, file_name="kb_normalized.csv", mime="text/csv")
        st.download_button("Download normalized Parquet", parquet_buffer.getvalue(), file_name="kb_normalized.parquet", mime="application/octet-stream")
        st.download_button("Download QA JSON", qa_report.to_json().encode("utf-8"), file_name="kb_qa_report.json", mime="application/json")


def _select_region_and_horizon(normalized_df: pd.DataFrame, prefix: str) -> tuple[str, int]:
    regions = _preferred_region_order(sorted(normalized_df["region"].dropna().unique().tolist()))
    if st.session_state["selected_region"] not in regions:
        st.session_state["selected_region"] = regions[0]
    if st.session_state["selected_horizon"] not in DEFAULT_HORIZONS:
        st.session_state["selected_horizon"] = DEFAULT_HORIZONS[0]
    left, right = st.columns([3, 2])
    with left:
        region = st.selectbox("Region", regions, index=regions.index(st.session_state["selected_region"]), key=f"{prefix}_region")
    with right:
        horizon = st.radio(
            "Forecast horizon",
            list(DEFAULT_HORIZONS),
            horizontal=True,
            index=list(DEFAULT_HORIZONS).index(st.session_state["selected_horizon"]),
            key=f"{prefix}_horizon",
        )
    st.session_state["selected_region"] = region
    st.session_state["selected_horizon"] = horizon
    return region, int(horizon)


def render_forecast_tab() -> None:
    normalized_df = st.session_state.get("normalized_df")
    qa_report = _current_qa_report()
    if normalized_df is None or normalized_df.empty or qa_report is None:
        st.info("Upload a workbook in `Data QA` before opening the forecast dashboard.")
        return
    if qa_report.blocking_errors:
        st.error("Forecasting is disabled until all blocking QA issues are resolved.")
        return
    region, horizon = _select_region_and_horizon(normalized_df, "forecast")
    feature_scope = st.session_state.get("feature_scope", DEFAULT_FEATURE_SCOPE)
    macro_df = st.session_state.get("macro_df")
    if macro_df is None:
        macro_df = pd.DataFrame()
    fingerprint = st.session_state.get("workbook_fingerprint") or "no-fingerprint"
    backtest_payload = None
    if isinstance(st.session_state.get("backtest_cache_ref"), dict):
        backtest_payload = st.session_state["backtest_cache_ref"].get((region, feature_scope))
    with st.spinner("Building forecast bundle..."):
        payload = forecast_region_cached(normalized_df, macro_df, region, horizon, feature_scope, fingerprint, MODEL_VERSION, backtest_payload)
    cards = st.columns(4)
    with cards[0]:
        _render_metric_card("Selected region", region, f"Horizon {horizon} weeks")
    with cards[1]:
        _render_metric_card("High risk", f"{payload['high_risk_score']:.1f}", "Percentile-calibrated overheating / rollover score")
    with cards[2]:
        _render_metric_card("Low zone", f"{payload['low_zone_score']:.1f}", "Percentile-calibrated oversold / trough score")
    with cards[3]:
        _render_metric_card("Top models", ", ".join(payload["top_models"]), "Backtest-ranked when available, fallback ranked otherwise")
    for message in payload["warnings"]:
        st.info(message)
    st.plotly_chart(_forecast_chart(payload), use_container_width=True)
    st.markdown("### Model Table")
    st.dataframe(payload["model_table"], use_container_width=True)
    if backtest_payload is None:
        st.caption("Detailed model ranking and residual-based interval calibration become fully grounded after running the backtest tab.")


def render_backtest_tab() -> None:
    normalized_df = st.session_state.get("normalized_df")
    qa_report = _current_qa_report()
    if normalized_df is None or normalized_df.empty or qa_report is None:
        st.info("Upload a workbook in `Data QA` before opening the backtest dashboard.")
        return
    if qa_report.blocking_errors:
        st.error("Backtesting is disabled until all blocking QA issues are resolved.")
        return
    region, _ = _select_region_and_horizon(normalized_df, "backtest")
    feature_scope = st.session_state.get("feature_scope", DEFAULT_FEATURE_SCOPE)
    macro_df = st.session_state.get("macro_df")
    if macro_df is None:
        macro_df = pd.DataFrame()
    fingerprint = st.session_state.get("workbook_fingerprint") or "no-fingerprint"
    progress = st.progress(0)
    run_button = st.button("Run backtest", use_container_width=True)
    has_cached = isinstance(st.session_state.get("backtest_cache_ref"), dict) and st.session_state["backtest_cache_ref"].get((region, feature_scope))
    if not run_button and not has_cached:
        st.info("Backtests run only when you press `Run backtest`.")
        return
    if run_button:
        progress.progress(15)
        with st.spinner("Running rolling-origin backtests..."):
            payload = run_backtests_cached(normalized_df, macro_df, region, feature_scope, fingerprint, MODEL_VERSION)
        progress.progress(100)
        if not isinstance(st.session_state["backtest_cache_ref"], dict):
            st.session_state["backtest_cache_ref"] = {}
        st.session_state["backtest_cache_ref"][(region, feature_scope)] = payload
    payload = st.session_state["backtest_cache_ref"][(region, feature_scope)]
    for warning in payload.get("warnings", []):
        st.warning(warning)
    st.markdown("### Leaderboard")
    st.dataframe(payload["leaderboard"], use_container_width=True)
    st.markdown("### Event Hit Rates")
    st.dataframe(payload["event_summary"], use_container_width=True)
    if not payload["predictions"].empty:
        st.markdown("### Prediction Residuals")
        st.dataframe(payload["predictions"].head(40), use_container_width=True)


def render_technical_overlay_tab() -> None:
    normalized_df = st.session_state.get("normalized_df")
    qa_report = _current_qa_report()
    if normalized_df is None or normalized_df.empty or qa_report is None:
        st.info("Upload a workbook in `Data QA` before opening the technical overlay.")
        return
    region, _ = _select_region_and_horizon(normalized_df, "technical")
    region_df = normalized_df.loc[normalized_df["region"] == region, ["date", "value"]].copy()
    st.plotly_chart(_technical_overlay_chart(region_df), use_container_width=True)


def main() -> None:
    configure_page()
    apply_style()
    initialize_session_state()
    st.markdown(
        """
        <div class="hero">
            <span class="pill">KB Weekly Excel</span>
            <span class="pill">QA-First</span>
            <span class="pill">StatsForecast + MLForecast</span>
            <h1>KB Weekly Real Estate Forecast Lab</h1>
            <p>
                Upload a KB weekly timeseries workbook, validate its structure, and then inspect medium-horizon
                forecasts, rolling-origin backtests, turning-point labels, and percentile-based high/low risk scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Data QA", "Forecast", "Backtest", "Technical Overlay"])
    with tabs[0]:
        render_data_qa_tab()
    with tabs[1]:
        render_forecast_tab()
    with tabs[2]:
        render_backtest_tab()
    with tabs[3]:
        render_technical_overlay_tab()


if __name__ == "__main__":
    main()
