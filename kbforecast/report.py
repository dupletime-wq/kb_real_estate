"""Display tables for the app, kept out of the Streamlit script so they can be tested."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import EngineFit, validation_summary

VALIDATION_LABELS = {
    "horizon": "예측 기간(주)", "n": "검증 표본", "model_MAE_pp": "모델 평균오차(%p)", "drift26_MAE_pp": "추세연장 평균오차(%p)",
    "randomwalk_MAE_pp": "무변화 평균오차(%p)", "histmean_MAE_pp": "과거 평균수익률 평균오차(%p)", "interval_coverage": "구간 적중률",
    "interval_width_pp": "구간 평균 폭(%p)", "interval_score90_pp": "90% 구간점수(%p)",
}
VALIDATION_COLUMNS = (
    "대상", "예측 기간(주)", "검증 표본", "모델 평균오차(%p)", "추세연장 평균오차(%p)", "무변화 평균오차(%p)", "과거 평균수익률 평균오차(%p)",
    "개선율(vs 추세연장, %)", "구간 적중률", "구간 평균 폭(%p)", "90% 구간점수(%p)",
)


def validation_table(fit: EngineFit, region: str, seoul: tuple[str, ...], horizons: tuple[int, ...]) -> tuple[pd.DataFrame, list[str]]:
    """(full table with display names, display columns that exist). Missing columns are skipped instead of raising."""
    frames = []
    for label, regions in ((f"{region} (선택 지역)", (region,)), ("서울 28개 지역 평균", seoul), ("전국 패널 전체", None)):
        v = validation_summary(fit, regions, horizons=tuple(h for h in horizons if h in fit.anchors))
        if not v.empty:
            frames.append(v.assign(대상=label))
    if not frames:
        return pd.DataFrame(), []
    table = pd.concat(frames)
    table["개선율(vs 추세연장, %)"] = table["skill_vs_drift26"] * 100
    table = table.rename(columns=VALIDATION_LABELS)
    if "구간 적중률" in table:
        table["구간 적중률"] = table["구간 적중률"] * 100
    return table, [c for c in VALIDATION_COLUMNS if c in table.columns]


def data_diagnostics(kb, hub_report=None, hub_problem: str | None = None, rate=None, cd=None, today: pd.Timestamp | None = None) -> dict:
    """What the forecast is built on, and what is stale, filled or estimated. Nothing is inferred beyond what the panel records.

    Returns {"headline": DataFrame(item, value, status), "regions": per region last real observation / weeks since / filled share of the
    last 26 weeks, "sentiment": per indicator and scope the last date and whether the newest weeks are observed or estimated}.
    Sentiment values added from a parent scope by the hub extension are listed as estimated; for the workbook's own history there is no
    estimated/observed record, so those weeks are reported as 'workbook (observed)'.
    """
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    last = kb.sale.index.max()
    age_weeks = (today - last).days / 7.0
    rows = [("가격 패널 마지막 주", str(last.date()), "오래됨" if age_weeks > 3 else "정상"), ("오늘 기준 경과 주수", f"{age_weeks:.1f}", "오래됨" if age_weeks > 3 else "정상")]
    if hub_report is not None and hub_report.applied:
        rows.append(("KB 데이터허브 보강", f"{len(hub_report.new_dates)}주 추가 ({hub_report.new_dates[0].date()}~{hub_report.new_dates[-1].date()}), 일치 지역 {hub_report.matched_regions}/{hub_report.total_regions}", "적용"))
        if hub_report.ended_regions:
            rows.append(("허브 시계열이 끝난 지역", ", ".join(hub_report.ended_regions[:8]) + (" …" if len(hub_report.ended_regions) > 8 else ""), "주의"))
        if hub_report.proxied_sentiment:
            rows.append(("추정(상위 범위 변화폭)으로 채운 심리 범위", ", ".join(hub_report.proxied_sentiment), "추정"))
    elif hub_problem:
        rows.append(("KB 데이터허브 보강", f"미적용: {hub_problem}", "미적용"))
    for name, series in (("기준금리", rate), ("CD91", cd)):
        if series is not None:
            behind = (last - series.known_through).days
            rows.append((f"{name} 자료 기준일", f"{series.known_through.date()} (패널 마지막 주보다 {behind}일 이전)" if behind > 0 else f"{series.known_through.date()}", "오래됨" if behind > 7 else "정상"))
    headline = pd.DataFrame(rows, columns=["항목", "값", "상태"])

    obs = kb.observed["sale"] if kb.observed is not None else None
    price = kb.sale
    recs = []
    for region in price.columns:
        col = price[region]
        real = obs[region] if obs is not None else col.notna()
        last_real = real[real].index.max() if real.any() else pd.NaT
        started = real.cummax()
        filled26 = float(((~real) & started).iloc[-26:].mean()) if len(real) else np.nan
        weeks_since = float((last - last_real).days / 7.0) if pd.notna(last_real) else np.nan
        recs.append({"region": region, "last_real_observation": last_real, "weeks_since_real_observation": weeks_since, "filled_share_last_26w": filled26,
                     "status": "관측 없음" if pd.isna(last_real) else ("최근 관측 공백" if weeks_since > 0 else ("채움 있음" if filled26 > 0 else "정상"))})
    regions = pd.DataFrame(recs)

    est = set(hub_report.proxied_sentiment) if (hub_report is not None and hub_report.applied) else set()
    n_new = len(hub_report.new_dates) if (hub_report is not None and hub_report.applied) else 0
    srows = []
    for indicator, frame in kb.sentiment.items():
        for scope in frame.columns:
            col = frame[scope].dropna()
            srows.append({"indicator": indicator, "scope": scope, "last_value_date": col.index.max() if len(col) else pd.NaT,
                          "weeks_behind_panel": float((last - col.index.max()).days / 7.0) if len(col) else np.nan,
                          "newest_weeks": f"추정 {n_new}주 (상위 범위 변화폭)" if scope in est else "워크북/허브 값"})
    return {"headline": headline, "regions": regions, "sentiment": pd.DataFrame(srows)}
