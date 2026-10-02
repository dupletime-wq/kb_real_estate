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
}
