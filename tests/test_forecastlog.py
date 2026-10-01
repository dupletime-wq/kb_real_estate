import json

import numpy as np
import pandas as pd

from kbforecast.engine import EngineFit
from kbforecast.forecastlog import load_log, score_log, summarize, write_log
from tests.synthetic import make_panel


def _fit_at(kb, origin_pos: int, h_list=(4, 13)) -> EngineFit:
    """An EngineFit whose live-origin predictions are fixed numbers (no model fitted)."""
    price = np.log(kb.sale.iloc[: origin_pos + 1])
    origin = price.index[-1]
    index = pd.MultiIndex.from_product([[origin], price.columns], names=["date", "region"])
    preds, bases = {}, {}
    for h in h_list:
        preds[h] = pd.DataFrame({"y": np.nan, "pred": 0.01 * h, "pred_raw": 0.01 * h, "lo": -0.02, "hi": 0.05 * h}, index=index)
        bases[h] = pd.DataFrame({"rw": 0.0, "drift26": 0.005 * h}, index=index)
    return EngineFit(price, tuple(h_list), ["r1"], preds, bases, {}, kb.hierarchy, kb.fingerprint, origin, False, {"first_origin": "x"}, {})


def test_log_is_written_once_per_origin_and_data(tmp_path):
    kb = make_panel()
    fit = _fit_at(kb, 300)
    path, _ = write_log(fit, kb, tmp_path)
    assert path is not None and path.exists()
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["data_fingerprint"] == kb.fingerprint and meta["model_config"]["hash"] and "git_commit" in meta
    again, message = write_log(fit, kb, tmp_path)
    assert again is None and "이미" in message  # same origin, same data: nothing is overwritten
    revised = type(kb)(kb.sale, kb.jeonse, kb.sentiment, kb.hierarchy, kb.fingerprint + "+revised", kb.warnings)
    other, _ = write_log(fit, revised, tmp_path)
    assert other is not None and other != path  # revised data is a separate point-in-time record


def test_scoring_uses_only_matured_rows_with_observed_prices(tmp_path):
    kb = make_panel()
    fit = _fit_at(kb, 300)
    write_log(fit, kb, tmp_path)
    log = load_log(tmp_path)
    region = log["region"].iloc[0]
    target = kb.sale.index[300] + pd.Timedelta(weeks=4)
    obs = kb.sale.notna()
    obs.loc[target, region] = False  # the 4-week-ahead price of this region was only a filled value
    masked = type(kb)(kb.sale, kb.jeonse, kb.sentiment, kb.hierarchy, kb.fingerprint, kb.warnings, {"sale": obs, "jeonse": kb.jeonse.notna()})
    scored = score_log(log, masked)
    row4 = scored[(scored["region"] == region) & (scored["horizon"] == 4)].iloc[0]
    assert np.isnan(row4["y"])  # not scored against a filled price
    other = scored[(scored["region"] != region) & (scored["horizon"] == 4)].iloc[0]
    assert np.isclose(other["y"], np.log(kb.sale.at[target, other["region"]]) - np.log(kb.sale.at[kb.sale.index[300], other["region"]]))
    early = _fit_at(kb, kb.sale.shape[0] - 3)  # origin 3 weeks before the end: the 4- and 13-week targets have not happened yet
    write_log(early, kb, tmp_path / "early")
    assert score_log(load_log(tmp_path / "early"), kb)["y"].isna().all()
    assert not summarize(scored, kb).empty
