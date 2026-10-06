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


def test_two_configurations_on_the_same_origin_and_data_never_collide(tmp_path):
    import dataclasses

    from kbforecast.forecastlog import config_id

    kb = make_panel()
    cur = _fit_at(kb, 300)
    cand = dataclasses.replace(cur, columns=["r1", "px_sent"], settings={"first_origin": "x", "variant": {"name": "B_px_sent", "extra_features": ["px_sent"], "hgb_mode": "auto", "bias": None}})
    p1, _ = write_log(cur, kb, tmp_path)
    p2, _ = write_log(cand, kb, tmp_path)
    assert p1 is not None and p2 is not None and p1 != p2 and p1.exists() and p2.exists()  # side by side
    again, msg = write_log(cand, kb, tmp_path)
    assert again is None and "이미" in msg  # the same configuration on the same origin and data is a no-op
    log = load_log(tmp_path)
    assert set(log["variant"]) == {"current", "B_px_sent"} and log["config_id"].nunique() == 2 and set(log["data_id"]) == {log["data_id"].iloc[0]}
    assert config_id(cur) != config_id(cand)
    # the identifier ignores data-dependent overlay values but not settings: same structure -> same id
    assert config_id(dataclasses.replace(cur, overlay={4: {"slope": -0.1}})) == config_id(cur)
    assert config_id(dataclasses.replace(cur, settings={"first_origin": "x", "refit_every": 26})) != config_id(cur)


def test_hgb_mode_blend_and_bias_settings_change_the_configuration_id():
    import dataclasses

    from kbforecast.forecastlog import config_id

    kb = make_panel()
    cur = _fit_at(kb, 300)
    ids = {config_id(dataclasses.replace(cur, settings={"first_origin": "x", "variant": {"name": n, **v}})) for n, v in
           {"a": {"hgb_mode": "auto"}, "b": {"hgb_mode": "timeval"}, "c": {"hgb_mode": "auto", "bias": {"name": "seoul_x0.5", "shrink": 0.5}}, "d": {"hgb_mode": "auto", "extra_features": ["obs_age"]}}.items()}
    assert len(ids) == 4


def test_pairing_uses_identical_origin_region_horizon_and_data_only(tmp_path):
    import dataclasses

    from kbforecast.forecastlog import pair_models, score_log, compare_logged

    kb = make_panel()
    cur = _fit_at(kb, 300)
    cand = dataclasses.replace(cur, settings={"first_origin": "x", "variant": {"name": "B_px_sent"}})
    write_log(cur, kb, tmp_path)
    write_log(cand, kb, tmp_path, regions={"서울특별시", "강남구"})  # the candidate was logged for fewer regions
    kb_other = type(kb)(kb.sale, kb.jeonse, kb.sentiment, kb.hierarchy, kb.fingerprint + "+revised", kb.warnings)
    write_log(cand, kb_other, tmp_path)  # same candidate on a different data vintage
    scored = score_log(load_log(tmp_path), kb)
    pairs = pair_models(scored, "current", "B_px_sent")
    assert set(pairs["region"]) == {"서울특별시", "강남구"} and pairs["data_id"].nunique() == 1  # only rows both models logged on the same vintage
    table = compare_logged(pairs, {"서울특별시", "강남구"})
    assert {"MAE_base_pp", "MAE_cand_pp", "coverage_base", "coverage_cand"} <= set(table.columns)  # point accuracy and coverage are separate columns


def test_first_release_values_are_archived_append_only_and_scored_separately(tmp_path):
    from kbforecast.forecastlog import record_first_seen, score_log

    kb = make_panel()
    fit = _fit_at(kb, 300)
    write_log(fit, kb, tmp_path)
    archive = tmp_path / "realized_first_seen.csv"
    origin = fit.last_date
    n1 = record_first_seen(kb, archive, since=origin)
    assert n1 > 0
    target = origin + pd.Timedelta(weeks=4)
    revised_sale = kb.sale.copy()
    revised_sale.loc[target, "서울특별시"] *= 1.10  # KB later revises that week
    kb2 = type(kb)(revised_sale, kb.jeonse, kb.sentiment, kb.hierarchy, kb.fingerprint, kb.warnings)
    assert record_first_seen(kb2, archive, since=origin) == 0  # nothing new, and the first value is not overwritten
    scored = score_log(load_log(tmp_path), kb2, realized="first_seen", first_seen_path=archive)
    row = scored[(scored["region"] == "서울특별시") & (scored["horizon"] == 4)].iloc[0]
    expected_first = np.log(kb.sale.at[target, "서울특별시"]) - np.log(row["origin_price"])
    assert np.isclose(row["y_first_seen"], expected_first, atol=1e-5)
    assert np.isclose(row["y_latest"], np.log(revised_sale.at[target, "서울특별시"]) - np.log(kb.sale.at[origin, "서울특별시"]))
    assert not np.isclose(row["y_first_seen"], row["y_latest"]) and row["y"] == row["y_first_seen"]
    assert np.isnan(row["y_first_published"])  # nothing is promoted to "first published" without a verified vintage


def test_records_written_before_the_config_scheme_still_load():
    from pathlib import Path

    log = load_log(Path(__file__).resolve().parents[1] / "forecast_log")
    assert len(log) > 0 and log["config_id"].str.startswith("legacy-").any() and (log["variant"] == "current").any()


def test_first_seen_archive_records_collection_lag_and_only_verified_vintages_count_as_published(tmp_path):
    from kbforecast.forecastlog import record_first_seen, score_log, verify_vintage

    kb = make_panel()
    fit = _fit_at(kb, 300)
    write_log(fit, kb, tmp_path)
    origin = fit.last_date
    archive, verified = tmp_path / "realized_first_seen.csv", tmp_path / "vintage_verified.csv"
    target = origin + pd.Timedelta(weeks=4)
    # first collection happens 40 days after the origin week, i.e. long after the early target weeks were published
    n = record_first_seen(kb, archive, since=origin, today=origin + pd.Timedelta(days=40))
    arch = pd.read_csv(archive, parse_dates=["date"])
    assert n == len(arch) and {"first_seen_utc", "collection_lag_days", "late_collection", "panel_last_date", "data_id"} <= set(arch.columns)
    row = arch[(arch["region"] == "서울특별시") & (arch["date"] == target)].iloc[0]
    assert row["collection_lag_days"] == 12 and bool(row["late_collection"]) is False  # target = origin + 28 days, seen at day 40
    first_week = arch[(arch["region"] == "서울특별시") & (arch["date"] == origin)].iloc[0]
    assert first_week["collection_lag_days"] == 40 and bool(first_week["late_collection"]) is True  # the back-filled week is flagged late
    meta = json.loads(archive.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["first_batch_late_rows"] > 0 and "NOT necessarily the first published" in meta["meaning"]
    scored = score_log(load_log(tmp_path), kb, realized="first_published", first_seen_path=archive, verified_path=verified)
    assert scored["y_first_seen"].notna().any() and scored["y_first_published"].isna().all() and scored["y"].isna().all()  # no verified vintage yet
    verify_vintage(verified, "서울특별시", [target], evidence="collected on the publication day and matched to the dated KB release", verified_by="test")
    scored = score_log(load_log(tmp_path), kb, realized="first_published", first_seen_path=archive, verified_path=verified)
    hit = scored[(scored["region"] == "서울특별시") & (scored["horizon"] == 4)].iloc[0]
    assert np.isclose(hit["y_first_published"], hit["y_first_seen"]) and scored["y_first_published"].notna().sum() == 1
