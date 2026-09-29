from __future__ import annotations

from pathlib import Path
import hashlib
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from kbforecast.engine import ENGINE_VERSION, EngineFit, fit_engine, forecast_region, validation_summary
from kbforecast.hub import HubData, HubReport, extend_kb_panel, fetch_hub
from kbforecast.indicators import IndicatorResult, evaluate_indicators
from kbforecast.kb_panel import KBPanel, parse_kb_panel, seoul_region_keys
from kbforecast.macro import ecos_api_key, load_macro, macro_weekly
from kbforecast.overlay import RateSeries, load_base_rate, scenario_adjustments

APP_TITLE = "KB 부동산 시세 예측 대시보드"
CACHE_DIR = Path(".cache")
HUB_TTL_SECONDS = 6 * 3600
HORIZONS = (13, 26, 52)
PROVINCE_ORDER = (
    "서울특별시", "경기도", "인천광역시", "부산광역시", "대구광역시", "대전광역시", "울산광역시", "(구)광주광역시",
    "전남광주통합특별시", "세종특별자치시", "강원특별자치도", "충청북도", "충청남도", "전북특별자치도",
    "(구)전라남도", "경상북도", "경상남도", "제주도",
)


# ----------------------------------------------------------------------------- data / engine plumbing
@st.cache_data(show_spinner=False)
def load_panel(file_bytes: bytes) -> KBPanel:
    return parse_kb_panel(file_bytes)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_rate(api_key: str) -> RateSeries:
    return load_base_rate(api_key or None)


def _hub_data(kb: KBPanel) -> HubData:
    """Hub download (about 1~2 minutes) cached on disk for a few hours."""
    path = CACHE_DIR / "hub_raw.pkl"
    try:
        if path.exists() and time.time() - path.stat().st_mtime < HUB_TTL_SECONDS:
            return pickle.loads(path.read_bytes())
    except Exception:
        pass
    hub = fetch_hub(kb.hierarchy)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(hub))
    except Exception:
        pass
    return hub


def extend_with_hub(kb: KBPanel) -> tuple[KBPanel, HubReport | None, str | None]:
    """Append the weeks KB 데이터허브 has published after the workbook. Never raises: on any problem the workbook is used as is."""
    store = st.session_state.setdefault("hub_ext", {})
    if kb.fingerprint in store:
        return store[kb.fingerprint]
    try:
        with st.status("KB 데이터허브에서 워크북 이후의 최신 주간 자료를 확인하는 중입니다 (1~2분, 이후 캐시).", expanded=False) as status:
            extended, report = extend_kb_panel(kb, _hub_data(kb))
            status.update(label=report.message, state="complete" if report.applied else "error")
        result: tuple[KBPanel, HubReport | None, str | None] = (extended, report, None if report.applied else report.message)
    except Exception as exc:  # noqa: BLE001 - the workbook alone is always a valid input
        result = (kb, None, f"KB 데이터허브 자료를 가져오지 못해 업로드한 파일만 사용합니다 ({exc})")
    store[kb.fingerprint] = result
    return result


def _engine_cache_path(fingerprint: str, target: str, use_macro: bool, rate: RateSeries | None) -> Path:
    macro_tag = "macro" if use_macro else "base"
    rate_tag = f"rate{rate.known_through:%Y%m%d}" if rate is not None else "norate"
    tag = hashlib.sha1(fingerprint.encode()).hexdigest()[:12]  # includes the hub extension date
    return CACHE_DIR / f"engine_{tag}_{target}_{macro_tag}_{rate_tag}_{ENGINE_VERSION}.pkl"


def _load_engine_from_disk(path: Path) -> EngineFit | None:
    try:
        if path.exists():
            return pickle.loads(path.read_bytes())
    except Exception:
        return None
    return None


def _save_engine_to_disk(path: Path, fit: EngineFit) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(fit))
    except Exception:
        pass  # read-only filesystems just skip the persistent cache


def get_engine(
    kb: KBPanel, target: str, use_macro: bool, api_key: str, rate: RateSeries | None = None
) -> tuple[EngineFit | None, list[str]]:
    """Fit (or reuse) the pooled engine for this workbook. Returns (fit, warnings)."""
    key = (kb.fingerprint, target, use_macro, rate.known_through if rate is not None else None)
    store = st.session_state.setdefault("engine_fits", {})
    if key in store:
        return store[key], []
    warnings: list[str] = []
    path = _engine_cache_path(kb.fingerprint, target, use_macro, rate)
    fit = _load_engine_from_disk(path)
    if fit is None:
        model_kb = kb if target == "sale" else kb.swap_target()
        macro_w = None
        if use_macro:
            macro, problems = load_macro(api_key, cache_dir=CACHE_DIR / "ecos")
            warnings.extend(problems)
            if macro:
                macro_w = macro_weekly(macro, model_kb.sale.index)
            else:
                use_macro = False
        with st.status("전국 패널 예측 엔진을 학습하고 검증하는 중입니다 (파일당 최초 1회, 약 3~4분).", expanded=True) as status:
            bar = st.progress(0.0, text="준비 중")

            def progress(done: int, total: int, message: str) -> None:
                bar.progress(min(1.0, done / max(total, 1)), text=message)

            fit = fit_engine(model_kb, macro_w, use_macro=use_macro, rate=rate, progress=progress)
            status.update(label="예측 엔진 준비 완료", state="complete", expanded=False)
        _save_engine_to_disk(path, fit)
    store[key] = fit
    return fit, warnings


# ----------------------------------------------------------------------------- formatting helpers
def _format_pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}%"


def _format_value(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def _series(kb: KBPanel, target: str, region: str) -> pd.Series:
    panel = kb.sale if target == "sale" else kb.jeonse
    return panel[region].dropna()


def region_groups(kb: KBPanel) -> dict[str, list[str]]:
    h = kb.hierarchy
    groups: dict[str, list[str]] = {}
    agg = [k for k in h.index[h["level"] == "agg"]]
    groups["집계 지수 (전국·수도권 등)"] = agg
    for province in PROVINCE_ORDER:
        keys = [k for k in h.index if (h.loc[k, "province"] == province or k == province) and h.loc[k, "level"] != "agg"]
        keys = [k for k in keys if k in kb.sale.columns and kb.sale[k].notna().sum() >= 120]
        if keys:
            groups[province] = keys
    return groups


# ----------------------------------------------------------------------------- charts
def make_forecast_chart(history: pd.Series, path: pd.DataFrame, y_title: str) -> go.Figure:
    history = history.tail(420)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="실제 지수", line=dict(color="#263238", width=2)))
    fig.add_trace(go.Scatter(x=path["date"], y=path["p90"], mode="lines", line=dict(color="rgba(30,136,229,0)"), hoverinfo="skip", showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=path["date"], y=path["p10"], mode="lines", name="예측구간", fill="tonexty",
            fillcolor="rgba(30,136,229,0.16)", line=dict(color="rgba(30,136,229,0)"),
        )
    )
    fig.add_trace(go.Scatter(x=path["date"], y=path["p50"], mode="lines", name="예측 중앙값", line=dict(color="#1e88e5", width=3)))
    fig.update_layout(
        height=470, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified", yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_comparison_chart(kb: KBPanel) -> go.Figure:
    names = [n for n in ("전국", "수도권", "서울특별시", "강북14개구", "강남11개구") if n in kb.sale.columns]
    rows = []
    for name in names:
        s = kb.sale[name].dropna()
        rows.append(
            {
                "region": name,
                "13주 변화율": (s.iloc[-1] / s.iloc[-14] - 1) * 100 if len(s) > 14 else np.nan,
                "52주 변화율": (s.iloc[-1] / s.iloc[-53] - 1) * 100 if len(s) > 53 else np.nan,
            }
        )
    data = pd.DataFrame(rows).melt(id_vars="region", var_name="period", value_name="ret")
    colors = {"13주 변화율": "#00897b", "52주 변화율": "#f9a825"}
    fig = go.Figure()
    for period, group in data.groupby("period"):
        fig.add_trace(
            go.Bar(x=group["region"], y=group["ret"], name=period, marker_color=colors.get(period, "#546e7a"),
                   text=[_format_pct(v, 2) for v in group["ret"]], textposition="outside")
        )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), barmode="group", yaxis_title="변화율",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


def make_contribution_chart(contrib: pd.Series) -> go.Figure:
    contrib = contrib.sort_values()
    colors = ["#c62828" if v < 0 else "#2e7d32" for v in contrib.values]
    fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h", marker_color=colors,
                           text=[f"{v:+.2f}%p" for v in contrib.values], textposition="outside"))
    span = float(np.abs(contrib.values).max()) or 1.0
    fig.update_traces(cliponaxis=False)
    fig.update_layout(height=280, margin=dict(l=20, r=60, t=20, b=20), xaxis_title="예측 수익률 기여도 (%p)",
                      xaxis_range=[min(0.0, float(contrib.min())) - span * 0.25, max(0.0, float(contrib.max())) + span * 0.25])
    return fig


def make_sentiment_chart(kb: KBPanel, region: str) -> go.Figure | None:
    if not kb.sentiment:
        return None
    scope = kb.hierarchy.loc[region, "sentiment_scope"] if region in kb.hierarchy.index else "전국"
    labels = {"buyer": "매수우위지수", "sale_txn": "매매거래활발지수", "jeonse_supply": "전세수급지수", "jeonse_txn": "전세거래활발지수"}
    fig = make_subplots(rows=2, cols=2, subplot_titles=[labels[k] for k in labels if k in kb.sentiment], vertical_spacing=0.16)
    for i, key in enumerate([k for k in labels if k in kb.sentiment]):
        frame = kb.sentiment[key]
        if scope not in frame.columns:
            continue
        s = frame[scope].dropna().tail(156)
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", showlegend=False, line=dict(width=2)), row=i // 2 + 1, col=i % 2 + 1)
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def make_technical_chart(history: pd.Series, ind: IndicatorResult) -> go.Figure:
    indicators = ind.frame.tail(420)
    passed = ind.passed_summary
    events = ind.passed_events.copy()
    active = set(passed["indicator"]) if not passed.empty else set()
    show_macd = "MACD" in active
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=indicators["date"], y=indicators["value"], name="지수", line=dict(color="#263238", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=indicators["date"], y=indicators["ma13"], name="MA13", line=dict(color="#7e57c2", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=indicators["date"], y=indicators["ma26"], name="MA26", line=dict(color="#5d4037", width=1)), row=1, col=1)
    if "Bollinger" in active:
        fig.add_trace(go.Scatter(x=indicators["date"], y=indicators["bb_upper"], name="Bollinger 상단", line=dict(color="rgba(0,137,123,0.45)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=indicators["date"], y=indicators["bb_lower"], name="Bollinger 하단", line=dict(color="rgba(0,137,123,0.45)", width=1),
                                 fill="tonexty", fillcolor="rgba(0,137,123,0.08)"), row=1, col=1)
    if "Ichimoku" in active:
        fig.add_trace(go.Scatter(x=indicators["date"], y=indicators["span_a"], name="Ichimoku Span A", line=dict(color="#ef6c00", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=indicators["date"], y=indicators["span_b"], name="Ichimoku Span B", line=dict(color="#6d4c41", width=1)), row=1, col=1)
    if not events.empty:
        events = events.loc[events["date"] >= indicators["date"].min()]
        for direction, symbol, color, label in (("bullish", "triangle-up", "#2e7d32", "통과 상승 신호"), ("bearish", "triangle-down", "#c62828", "통과 경계 신호")):
            sub = events.loc[events["direction"] == direction]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["date"], y=sub["value"], mode="markers", name=label,
                                         marker=dict(symbol=symbol, color=color, size=9), text=sub["indicator"] + " " + sub["signal"]), row=1, col=1)
    if show_macd:
        fig.add_trace(go.Scatter(x=indicators["date"], y=indicators["macd"], name="MACD", line=dict(color="#1565c0", width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=indicators["date"], y=indicators["macd_signal"], name="MACD Signal", line=dict(color="#ef6c00", width=1)), row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=35, b=20), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
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
        .app-note { padding: .75rem .9rem; border: 1px solid #d7dee8; border-radius: 8px; background: #f8fafc; color: #334155; font-size: .92rem; }
        div[data-testid="stMetric"] { border: 1px solid #e2e8f0; border-radius: 8px; padding: .75rem .85rem; background: #fff; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- tabs
def _overlay_note(fit: EngineFit, region: str, horizon: int) -> None:
    """Explain the Seoul policy-rate adjustment for the selected region's forecast (only shown when it applies)."""
    if not fit.overlay:
        return
    anchor = min(fit.anchors, key=lambda a: abs(a - horizon))
    info = fit.overlay.get(anchor)
    if info is None or region not in seoul_region_keys(fit.hierarchy):
        return
    try:
        live = fit.predictions[anchor].xs(fit.last_date, level="date").loc[region]
    except KeyError:
        return
    adj = float(live.get("overlay", 0.0)) * 100
    change = info["rate_change_26w"]
    change_text = "확인 불가" if pd.isna(change) else f"{change:+.2f}%p"
    st.info(
        f"**서울 기준금리 보정** · 최근 26주 기준금리 변화 {change_text} (자료 {info['rate_source']}, {info['rate_known_through']}까지) → "
        f"{anchor}주 예측 수익률에 **{adj:+.2f}%p** 반영 (계수 {info['slope']:.2f}, 부호는 '금리↑ → 수익률↓'로 제한). "
        "과거 검증에서 서울 평균오차를 2~3% 줄였지만 통계적 유의성은 약합니다(단측 p≈0.13). '예측 근거' 탭과 '검증' 탭에서 보정 전후를 볼 수 있습니다."
    )


def _rate_scenario(fit: EngineFit, region: str, horizon: int, rate: RateSeries | None) -> None:
    """Seoul-only what-if: how the policy-rate term would evolve if the base rate follows a hypothetical path."""
    if not fit.overlay or rate is None or region not in seoul_region_keys(fit.hierarchy):
        return
    anchors = [h for h in HORIZONS if h in fit.overlay]
    slopes = {h: fit.overlay[h]["slope"] for h in anchors}
    try:
        raw = {h: float(fit.predictions[h].xs(fit.last_date, level="date").loc[region, "pred_raw"]) for h in anchors}
    except KeyError:
        return
    with st.expander("기준금리 시나리오 시뮬레이션 (서울)"):
        current = float(rate.frame.sort_values("date")["value"].iloc[-1])
        c1, c2 = st.columns(2)
        terminal = c1.number_input("최종 기준금리 (%)", min_value=1.0, max_value=7.0, value=max(3.75, current), step=0.25, format="%.2f")
        gap = c2.slider("인상 간격 (주, 1회 0.25%p)", min_value=4, max_value=26, value=7, help="금통위는 연 8회(약 6~7주 간격)입니다.")
        candidates = sorted({round(current, 2), 3.5, 3.75, 4.0, round(terminal, 2)})
        runs = {c: scenario_adjustments(rate, c, gap, slopes, raw) for c in candidates}
        anchor = min(anchors, key=lambda a: abs(a - horizon))
        fig = go.Figure()
        for c, run in runs.items():
            name = f"{c:.2f}%" + (" (현재 유지)" if abs(c - current) < 1e-9 else "")
            fig.add_trace(go.Scatter(x=run["timeline"].index, y=run["timeline"][f"adj_{anchor}"], mode="lines", name=name,
                                     line=dict(width=3 if abs(c - terminal) < 1e-9 else 1.5)))
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), yaxis_title=f"{anchor}주 예측에 더해지는 금리 보정 (%p)",
                          hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, width="stretch")
        rows = []
        for c, run in runs.items():
            info = run["summary"][anchor]
            rows.append({
                "최종 금리(%)": c, "26주 금리 변화 최대(%p)": run["peak_delta26"],
                "보정 최대 하락(%p)": info["peak_adjustment_pp"], "그 시점": info["peak_date"].date(),
                "보정 소멸 시점": run["fade_date"].date() if run["peak_delta26"] > 0 else None,
                "보정 전 예측(%)": info["raw_return_pct"], "보정 최대 시 예측(%)": info["return_at_peak_pct"],
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("최종 금리(%)", "26주 금리 변화 최대(%p)", "보정 최대 하락(%p)", "보정 전 예측(%)", "보정 최대 시 예측(%)")})
        st.caption(
            "**이 시뮬레이션이 말해주는 것과 아닌 것.** 보정 항은 '기준금리 *26주 변화*'에 비례하므로, 같은 속도로 올리는 동안에는 최종 금리가 3.5%든 3.75%든 4.0%든 "
            "최대 하락폭이 같고 (높을수록 하락 압력이 더 오래 지속), 인상을 멈추고 26주가 지나면 보정이 0으로 돌아갑니다. 금리 *수준*의 누적 효과는 이 모형에 들어 있지 않습니다. "
            "'보정 최대 시 예측'은 그 시점에도 다른 요인은 지금 예측(보정 전)과 같다고 둔 단순 합산이며 모형을 다시 학습한 예측이 아닙니다. 계수는 과거 몇 차례 금리 사이클에서 추정되어 불확실합니다(p≈0.13)."
        )


def _validation_tab(fit: EngineFit, kb: KBPanel, region: str, horizon: int) -> None:
    seoul = tuple(sorted(seoul_region_keys(kb.hierarchy)))
    frames = []
    for label, regions in ((f"{region} (선택 지역)", (region,)), ("서울 28개 지역 평균", seoul), ("전국 패널 전체", None)):
        v = validation_summary(fit, regions, horizons=tuple(h for h in HORIZONS if h in fit.anchors))
        if not v.empty:
            frames.append(v.assign(대상=label))
    if not frames:
        st.info("검증 결과가 없습니다.")
        return
    table = pd.concat(frames)
    table["개선율(vs 추세연장, %)"] = table["skill_vs_drift26"] * 100
    table = table.rename(
        columns={"horizon": "예측 기간(주)", "n": "검증 표본", "model_MAE_pp": "모델 평균오차(%p)", "drift26_MAE_pp": "추세연장 평균오차(%p)",
                 "randomwalk_MAE_pp": "무변화 평균오차(%p)", "interval_coverage": "구간 적중률"}
    )
    table["구간 적중률"] = table["구간 적중률"] * 100
    cols = ["대상", "예측 기간(주)", "검증 표본", "모델 평균오차(%p)", "추세연장 평균오차(%p)", "무변화 평균오차(%p)", "개선율(vs 추세연장, %)", "구간 적중률"]
    st.dataframe(
        table[cols], width="stretch", hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.2f") for c in cols[3:]} | {"구간 적중률": st.column_config.NumberColumn(format="%.1f%%")},
    )
    if fit.overlay:
        seoul_rows = table[table["대상"] == "서울 28개 지역 평균"].copy()
        if not seoul_rows.empty:
            seoul_rows["보정 효과(%)"] = (seoul_rows["모델 평균오차(%p)"] / seoul_rows["raw_model_MAE_pp"] - 1) * 100
            st.markdown("**서울 기준금리 보정 전/후 (서울 28개 지역 평균 오차, %p)**")
            st.dataframe(
                seoul_rows[["예측 기간(주)", "raw_model_MAE_pp", "모델 평균오차(%p)", "보정 효과(%)"]].rename(
                    columns={"raw_model_MAE_pp": "보정 전", "모델 평균오차(%p)": "보정 후"}
                ),
                width="stretch", hide_index=True,
                column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("보정 전", "보정 후", "보정 효과(%)")},
            )
            st.caption(
                "보정은 서울 시리즈의 과거 검증 잔차를 '기준금리 26주 변화'에 회귀한 단일 계수(≤0)로 만듭니다. 각 시점에서는 그때까지 라벨이 확정된 잔차만 씁니다. "
                "개선은 금리 인상기(2022–23)에 집중되어 있고 통계적으로는 단측 p≈0.13 수준이라, 경제적 판단(서울의 유동성 민감도)에 근거해 적용한 것입니다."
            )
    first, last = table["from"].min().date(), table["to"].max().date()
    st.caption(
        f"매 시점마다 그 시점까지의 데이터만으로 다시 학습해 미래를 예측하고 실제와 비교한 결과입니다 (walk-forward, {first} ~ {last}). "
        "평균오차는 예측 기간 누적 수익률의 절대오차(%p)이며, '추세연장'은 최근 26주 상승률을 그대로 이어 붙인 단순 기준선입니다. "
        "구간 적중률은 실제값이 예측구간(80% 목표) 안에 들어온 비율입니다 — 급변 국면에서는 목표보다 낮아질 수 있습니다."
    )


def _rank_tab(fit: EngineFit, kb: KBPanel, horizon: int) -> None:
    anchor = min(fit.anchors, key=lambda a: abs(a - horizon))
    pred = fit.predictions[anchor]
    try:
        live = pred.xs(fit.last_date, level="date")
    except KeyError:
        st.info("최신 시점 예측이 없습니다.")
        return
    live = live.dropna(subset=["pred"])
    table = pd.DataFrame(
        {
            "지역": live.index,
            "예측 수익률(%)": (np.exp(live["pred"]) - 1) * 100,
            "하단(%)": (np.exp(live["lo"].fillna(live["pred"])) - 1) * 100,
            "상단(%)": (np.exp(live["hi"].fillna(live["pred"])) - 1) * 100,
        }
    )
    table["구분"] = table["지역"].map(lambda k: kb.hierarchy.loc[k, "level"] if k in kb.hierarchy.index else "")
    table["시도"] = table["지역"].map(lambda k: (kb.hierarchy.loc[k, "province"] or "") if k in kb.hierarchy.index else "")
    level_names = {"agg": "집계", "province": "시도", "group": "서울 권역", "gu": "구·군", "city": "시"}
    table["구분"] = table["구분"].map(level_names).fillna("")
    kinds = st.multiselect("지역 구분", sorted(table["구분"].unique()), default=[k for k in ("구·군", "시") if k in set(table["구분"])])
    view = table[table["구분"].isin(kinds)] if kinds else table
    st.dataframe(view.sort_values("예측 수익률(%)", ascending=False), width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("예측 수익률(%)", "하단(%)", "상단(%)")})
    st.caption(f"{anchor}주 후 누적 수익률 예측(기준일 {fit.last_date.date()}). 하단·상단은 예측구간입니다.")


# ----------------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="KB", layout="wide")
    _style()
    st.title(APP_TITLE)
    st.markdown(
        """
        <div class="app-note">
        KB 주간시계열 XLSX의 전국 지역 전체를 하나의 패널로 학습해 주간 가격지수의 13·26·52주 뒤 변화를 예측합니다.
        모든 성능 수치는 과거 시점마다 재학습해 미래를 맞혀 본 walk-forward 검증 결과이며, 투자 권유가 아닙니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("입력")
        uploaded = st.file_uploader("KB 주간시계열 XLSX 업로드", type=["xlsx"])
        use_hub = st.checkbox(
            "KB 데이터허브 최신 주간 자료 보강", value=True,
            help="업로드한 파일 이후에 KB 데이터허브(data.kbland.kr)가 공개한 주간 가격지수·심리지표를 붙여 예측 기준일을 앞당깁니다. "
                 "지역별로 워크북과 값이 정확히 일치하는 경우에만 사용하며, 실패하면 업로드한 파일만 씁니다. 비공식 공개 API를 사용합니다.",
        )
        sample_path = None
        use_sample = False
        if uploaded is None:
            try:
                sample_path = next(Path(".").glob("*_주간시계열.xlsx"))
                use_sample = st.checkbox("로컬 샘플 파일 사용", value=False)
            except StopIteration:
                sample_path = None
    if uploaded is None and not use_sample:
        st.info("왼쪽에서 `20260420_주간시계열.xlsx`와 같은 KB 주간시계열 파일을 업로드하세요.")
        return
    file_bytes = uploaded.getvalue() if uploaded is not None else sample_path.read_bytes()
    source_name = uploaded.name if uploaded is not None else sample_path.name
    try:
        kb = load_panel(file_bytes)
    except Exception as exc:
        st.error(f"파일을 해석하지 못했습니다: {exc}")
        return
    for message in kb.warnings:
        st.warning(message)
    hub_report, hub_problem = None, None
    if use_hub:
        kb, hub_report, hub_problem = extend_with_hub(kb)
    if hub_report is not None and hub_report.applied:
        st.info(hub_report.message + f" · 원본 워크북 기준일 {(hub_report.new_dates[0] - pd.Timedelta(weeks=1)).date()}")
    elif hub_problem:
        st.caption(f"KB 데이터허브 보강 미적용: {hub_problem}")

    groups = region_groups(kb)
    with st.sidebar:
        st.header("분석 설정")
        target_label = st.selectbox("분석 지표", ["매매지수", "전세지수"], index=0)
        target = "sale" if target_label == "매매지수" else "jeonse"
        horizon = st.selectbox("예측 기간", list(HORIZONS), index=1, format_func=lambda v: f"{v}주")
        group = st.selectbox("권역", list(groups), index=list(groups).index("서울특별시") if "서울특별시" in groups else 0)
        region = st.selectbox("지역", groups[group], index=0)
        seoul_overlay = st.checkbox(
            "서울 기준금리 보정", value=True, disabled=target != "sale",
            help="서울은 유동성에 더 민감하다는 판단으로, 풀링 예측 위에 서울 시리즈에만 '기준금리 26주 변화'에 대한 보정(부호 제약)을 더합니다. "
                 "과거 검증에서 오차가 소폭 줄었지만(단측 p≈0.13) 통계적으로 확정된 수준은 아닙니다. 매매지수에만 적용됩니다.",
        )
        with st.expander("고급 (실험)"):
            use_macro = st.checkbox(
                "한국은행 ECOS 거시지표 피처 포함", value=False,
                help="검증에서 예측력이 개선되지 않았고 오히려 과적합 경향이 있어 기본은 끕니다.",
            )
            api_key = st.text_input("ECOS 인증키", value=ecos_api_key(), type="password") if use_macro else ""

    rate = load_rate(api_key or ecos_api_key()) if (seoul_overlay and target == "sale") else None
    if rate is not None and rate.known_through < kb.last_date:
        st.sidebar.warning(
            f"기준금리 자료가 {rate.known_through.date()}까지만 반영되어 있어 그 이후 금리 변경은 보정에 들어가지 않습니다 "
            "(ECOS 인증키를 넣으면 최신 자료로 갱신됩니다)."
        )
    fit, fit_warnings = get_engine(kb, target, use_macro and bool(api_key), api_key, rate)
    for message in fit_warnings:
        st.sidebar.warning(message)

    series = _series(kb, target, region)
    latest = float(series.iloc[-1])
    prev13 = float(series.iloc[-14]) if len(series) > 14 else np.nan
    prev52 = float(series.iloc[-53]) if len(series) > 53 else np.nan
    top = st.columns(4)
    top[0].metric("최근 기준일", str(series.index.max().date()), help=f"데이터 시작: {series.index.min().date()}")
    top[1].metric("분석 지역 수", f"{kb.sale.shape[1]:,}개")
    top[2].metric("최근 지수", _format_value(latest, 2))
    top[3].metric("13주 변화율", _format_pct((latest / prev13 - 1) * 100, 2) if pd.notna(prev13) else "-",
                  delta=_format_pct((latest / prev52 - 1) * 100, 2) if pd.notna(prev52) else None,
                  help="큰 숫자는 13주 변화율, 델타는 52주 변화율입니다.")
    st.caption(f"파일 지문: `{kb.fingerprint[:12]}` · 원본: `{source_name}`")
    st.subheader("주요 권역 최근 변화율")
    st.plotly_chart(make_comparison_chart(kb), width="stretch")

    st.subheader(f"{region} {target_label} 예측 결과")
    try:
        fc = forecast_region(fit, region, horizon)
    except (ValueError, KeyError) as exc:
        st.warning(f"이 지역은 예측할 수 없습니다: {exc}")
        fc = None

    tab_names = ["예측", "예측 근거", "검증", "지역 순위", "기술지표"]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        if fc is None:
            st.info("데이터 이력이 충분한 다른 지역을 선택해 주세요.")
        else:
            end = fc.path.iloc[-1]
            cols = st.columns(3)
            cols[0].metric(f"{horizon}주 후 예측 지수", _format_value(end["p50"], 2), delta=_format_pct((end["p50"] / fc.last_value - 1) * 100, 2))
            cols[1].metric("예측구간 (하단 ~ 상단)", f"{_format_value(end['p10'], 1)} ~ {_format_value(end['p90'], 1)}")
            cols[2].metric("기준일", str(fc.origin.date()))
            st.plotly_chart(make_forecast_chart(series, fc.path, f"KB {target_label}"), width="stretch")
            _overlay_note(fit, region, horizon)
            _rate_scenario(fit, region, horizon, rate)
            with st.expander("기간별 예측 수익률", expanded=False):
                st.dataframe(
                    fc.anchor_table.rename(columns={"h": "기간(주)", "pred_pct": "예측 수익률(%)", "lo_pct": "하단(%)", "hi_pct": "상단(%)"}),
                    width="stretch", hide_index=True,
                    column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("예측 수익률(%)", "하단(%)", "상단(%)")},
                )
    with tabs[1]:
        anchor = min(fit.anchors, key=lambda a: abs(a - horizon))
        contrib = fit.contributions.get(anchor)
        if contrib is None or contrib.empty or region not in contrib.index:
            st.info("이 지역의 기여도 분해를 계산할 수 없습니다.")
        else:
            st.markdown(f"**{anchor}주 예측에서 각 정보군이 미친 영향** (선형 모형 절반의 분해, 평균 대비 %p)")
            st.plotly_chart(make_contribution_chart(contrib.loc[region]), width="stretch")
            st.caption("양수는 예측을 위로, 음수는 아래로 미는 요인입니다. 블렌드의 다른 절반(트리 모형)은 비선형이라 분해하지 않았습니다.")
        fig = make_sentiment_chart(kb, region)
        if fig is not None:
            st.markdown("**KB 심리지표 (최근 3년, 해당 지역이 속한 권역 기준)**")
            st.plotly_chart(fig, width="stretch")
    with tabs[2]:
        _validation_tab(fit, kb, region, horizon)
    with tabs[3]:
        _rank_tab(fit, kb, horizon)
    with tabs[4]:
        ind = evaluate_indicators(series, min(13, horizon))
        if ind.passed_summary.empty:
            st.info("이번 데이터와 검증 기준에서 통과한 기술지표 신호가 없습니다. 차트에는 가격 흐름과 보조 이동평균만 표시합니다.")
        else:
            st.dataframe(
                ind.passed_summary.drop(columns=["passed"]).rename(
                    columns={"indicator": "지표", "signal": "신호", "direction": "검증 방향", "samples": "표본 수",
                             "hit_rate_pct": "적중률(%)", "mean_forward_return_pct": "평균 선행수익률(%)", "p_value": "p-value"}
                ),
                width="stretch", hide_index=True,
                column_config={"적중률(%)": st.column_config.NumberColumn(format="%.1f"),
                               "평균 선행수익률(%)": st.column_config.NumberColumn(format="%.3f"),
                               "p-value": st.column_config.NumberColumn(format="%.4f")},
            )
        st.plotly_chart(make_technical_chart(series, ind), width="stretch")


if __name__ == "__main__":
    main()
