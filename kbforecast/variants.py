"""Named model variants: the current production model and the candidates that are tested or logged next to it.

A variant changes ONLY what is listed here; everything else (labels, refit schedule, overlay, intervals) is the production setting. The
fields feed the model-configuration id of the forecast log (forecastlog.model_config), so two variants can never share a log record.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .calibration import BiasCfg


@dataclass(frozen=True)
class EngineVariant:
    name: str = "current"
    extra_features: tuple[str, ...] = ()  # names from candidates.ALL_CANDIDATE_FEATURES, appended to the model columns
    hgb_mode: str = "auto"  # "auto" = scikit-learn default early stopping (production), "fixed" = no early stopping, "timeval" = purged time-ordered validation
    ridge_mode: str = "fixed"  # "fixed" = the validated alpha per horizon (production), "alpha_cv" = purged time-validated alpha, "group_alpha_cv" = one alpha for Seoul and one for the rest
    bias: BiasCfg | None = None  # median-residual correction applied after the rate overlay
    volume_extra_lag_weeks: int = 0  # sensitivity of the volume features to the assumed reporting lag

    def as_dict(self) -> dict:
        d = asdict(self)
        d["extra_features"] = list(self.extra_features)
        return d

    @property
    def is_current(self) -> bool:
        return self == CURRENT


CURRENT = EngineVariant()

NAMED_VARIANTS: dict[str, EngineVariant] = {
    "current": CURRENT,
    "B_px_sent": EngineVariant("B_px_sent", ("px_sent",)),
    "B_divergence": EngineVariant("B_divergence", ("divergence",)),
    "B_buyer_run": EngineVariant("B_buyer_run", ("buyer_run",)),
    "B_all": EngineVariant("B_all", ("px_sent", "divergence", "buyer_run")),
    "C_obs_age": EngineVariant("C_obs_age", ("obs_age",)),
    "C_fill_ratio": EngineVariant("C_fill_ratio", ("fill_ratio26",)),
    "C_both": EngineVariant("C_both", ("obs_age", "fill_ratio26")),
    "D_hgb_fixed": EngineVariant("D_hgb_fixed", hgb_mode="fixed"),
    "D_hgb_timeval": EngineVariant("D_hgb_timeval", hgb_mode="timeval"),
    "V_rel36": EngineVariant("V_rel36", ("vol_rel36",)),
    "V_chg3": EngineVariant("V_chg3", ("vol_chg3",)),
    "V_px_inter": EngineVariant("V_px_inter", ("vol_px_inter",)),
    "V_all": EngineVariant("V_all", ("vol_rel36", "vol_chg3", "vol_px_inter")),
    # long-horizon round (52 / 104 / 208 weeks)
    "R_alpha_cv": EngineVariant("R_alpha_cv", ridge_mode="alpha_cv"),
    "R_group_alpha": EngineVariant("R_group_alpha", ridge_mode="group_alpha_cv"),
    "L_longmem": EngineVariant("L_longmem", ("r104", "pdev156", "pdev260")),
    "L_valuation": EngineVariant("L_valuation", ("sj_level", "sj_dev156", "sj_z156")),
    # post-hoc ablation of L_valuation (added after its result was seen; counted in the experiment log)
    "J_level": EngineVariant("J_level", ("sj_level",)),
    "J_dev": EngineVariant("J_dev", ("sj_dev156", "sj_z156")),
    # post-hoc structure (added after J_level's result was seen): the valuation level enters for the Seoul series only
    "J_level_seoul": EngineVariant("J_level_seoul", ("sj_level_seoul",)),
    # exchange-rate round (2026-10-05): two pre-specified candidates, one common feature pair for all regions and the same pair for Seoul only
    "X_fx": EngineVariant("X_fx", ("fx_r26", "fx_dev156")),
    "X_fx_seoul": EngineVariant("X_fx_seoul", ("fx_r26_seoul", "fx_dev156_seoul")),
    "L_all": EngineVariant("L_all", ("r104", "pdev156", "pdev260", "sj_level", "sj_dev156", "sj_z156")),
}
