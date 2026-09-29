"""Small synthetic KB-like panel for fast, deterministic tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kbforecast.kb_panel import KBPanel, build_hierarchy


def make_panel(weeks: int = 420, seed: int = 0) -> KBPanel:
    rng = np.random.default_rng(seed)
    names = ["전국", "서울특별시", "강북14개구", "강북구", "노원구", "강남11개구", "강남구", "서초구", "경기도", "수원시", "성남시"]
    hierarchy = build_hierarchy(names)
    index = pd.date_range("2008-04-07", periods=weeks, freq="W-MON")
    common = rng.normal(0.001, 0.004, size=weeks).cumsum()
    sale = {}
    for name in hierarchy["key"]:
        sale[name] = 100 * np.exp(common + rng.normal(0.0005, 0.003, size=weeks).cumsum())
    sale = pd.DataFrame(sale, index=index)
    jeonse = sale.mul(np.exp(rng.normal(0, 0.002, size=sale.shape).cumsum(axis=0)))
    scopes = ["전국", "서울특별시", "강북14개구", "강남11개구", "경기도"]
    sentiment = {
        key: pd.DataFrame({s: 50 + rng.normal(0, 5, size=weeks).cumsum() * 0.2 for s in scopes}, index=index)
        for key in ("buyer", "sale_txn", "jeonse_supply", "jeonse_txn")
    }
    hierarchy = hierarchy.assign(sentiment_scope=[
        {"강북구": "강북14개구", "노원구": "강북14개구", "강남구": "강남11개구", "서초구": "강남11개구", "수원시": "경기도", "성남시": "경기도"}.get(k, k if k in scopes else "전국")
        for k in hierarchy["key"]
    ])
    return KBPanel(sale, jeonse, sentiment, hierarchy.set_index("key", drop=False), "synthetic", ())
