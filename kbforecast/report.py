"""Display tables for the app, kept out of the Streamlit script so they can be tested."""
from __future__ import annotations

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
