"""Pooled-panel regressors used by the direct multi-horizon forecaster."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

# A model function fits on (X_train, y_train) and returns predictions for X_pred.
ModelFn = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]


def _prepare(X_train: pd.DataFrame, X_pred: pd.DataFrame, clip: float = 5.0):
    """Median-impute with train medians, standardize with train stats, winsorize to +/-clip sigma."""
    med = X_train.median()
    mu_src = X_train.fillna(med)
    mu = mu_src.mean()
    sd = mu_src.std().replace(0, 1.0).fillna(1.0)
    a = ((mu_src - mu) / sd).clip(-clip, clip).to_numpy(dtype=np.float64)
    b = ((X_pred.fillna(med) - mu) / sd).clip(-clip, clip).to_numpy(dtype=np.float64)
    return np.nan_to_num(a), np.nan_to_num(b)


def ridge_model(alpha: float = 300.0) -> ModelFn:
    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame) -> np.ndarray:
        a, b = _prepare(X_train, X_pred)
        y_mu = float(np.mean(y_train))
        model = Ridge(alpha=alpha).fit(a, y_train - y_mu)
        return model.predict(b) + y_mu

    return fit_predict


def hgb_model(
    max_iter: int = 200,
    learning_rate: float = 0.04,
    max_leaf_nodes: int = 8,
    min_samples_leaf: int = 200,
    l2: float = 5.0,
    seed: int = 0,
    row_stride: int = 2,
) -> ModelFn:
    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame) -> np.ndarray:
        idx = np.arange(0, len(X_train), row_stride)
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2,
            random_state=seed,
        )
        model.fit(X_train.iloc[idx].to_numpy(dtype=np.float32), y_train[idx])
        return model.predict(X_pred.to_numpy(dtype=np.float32))

    return fit_predict


def blend(models: list[ModelFn], weights: list[float] | None = None) -> ModelFn:
    w = np.asarray(weights if weights is not None else [1.0] * len(models), dtype=float)
    w = w / w.sum()

    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame) -> np.ndarray:
        return sum(wi * m(X_train, y_train, X_pred) for wi, m in zip(w, models))

    return fit_predict
