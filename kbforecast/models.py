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
    """Median-impute with train medians, standardize with train stats, winsorize to +/-clip sigma (numpy fast path)."""
    a = X_train.to_numpy(dtype=np.float64, copy=True)
    b = X_pred.to_numpy(dtype=np.float64, copy=True)
    med = np.nanmedian(a, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for arr in (a, b):
        rows, cols = np.where(np.isnan(arr))
        arr[rows, cols] = med[cols]
    mu = a.mean(axis=0)
    sd = a.std(axis=0, ddof=1)
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
    a = np.clip((a - mu) / sd, -clip, clip)
    b = np.clip((b - mu) / sd, -clip, clip)
    return np.nan_to_num(a), np.nan_to_num(b)


def ridge_model(alpha: float = 300.0) -> ModelFn:
    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame) -> np.ndarray:
        a, b = _prepare(X_train, X_pred)
        y_mu = float(np.mean(y_train))
        model = Ridge(alpha=alpha).fit(a, y_train - y_mu)
        return model.predict(b) + y_mu

    return fit_predict


HGB_DEFAULTS = dict(max_iter=200, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=200, l2=5.0, seed=0, row_stride=2)


def _hgb(max_iter: int, learning_rate: float, max_leaf_nodes: int, min_samples_leaf: int, l2: float, seed: int, early_stopping) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2, random_state=seed, early_stopping=early_stopping,
    )


ALPHA_GRID = (3e3, 1e4, 3e4, 1e5, 3e5, 1e6)


def _std_pair(a: pd.DataFrame, b: pd.DataFrame):
    return _prepare(a, b)


def ridge_model_timeval(purge_weeks: int, alphas: tuple[float, ...] = ALPHA_GRID, val_share: float = 0.2, group_fn=None, record: list | None = None) -> ModelFn:
    """Ridge whose shrinkage is chosen on a purged, time-ordered validation block inside the training window (MAE, the evaluation metric).

    The newest `val_share` of the training dates validates, fit rows end `purge_weeks` (the horizon) before it. The winning alpha refits on all rows.
    With `group_fn(region) -> label` every group gets its own alpha, chosen on that group's validation rows, and predicts its own rows: the
    pooled coefficients are the same family of models, only the amount of shrinkage differs by group (e.g. Seoul vs the rest).
    """

    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame) -> np.ndarray:
        regions_tr = np.asarray(X_train.index.get_level_values("region"))
        regions_pr = np.asarray(X_pred.index.get_level_values("region"))
        lab_tr = np.asarray([group_fn(r) for r in regions_tr]) if group_fn else np.zeros(len(regions_tr), dtype=object)
        lab_pr = np.asarray([group_fn(r) for r in regions_pr]) if group_fn else np.zeros(len(regions_pr), dtype=object)
        fit_rows, val_rows = time_split_masks(X_train.index.get_level_values("date"), purge_weeks, val_share)
        best = {g: float(alphas[len(alphas) // 2]) for g in np.unique(lab_tr)}
        if fit_rows.sum() >= 2000 and val_rows.sum() >= 500:
            a, b = _prepare(X_train[fit_rows], X_train[val_rows])
            yt = y_train[fit_rows]
            yv = y_train[val_rows]
            mu = float(np.mean(yt))
            lv = lab_tr[val_rows]
            errs = {g: [] for g in best}
            for alpha in alphas:
                pred = Ridge(alpha=alpha).fit(a, yt - mu).predict(b) + mu
                for g in best:
                    m = lv == g
                    errs[g].append(float(np.mean(np.abs(pred[m] - yv[m]))) if m.any() else np.inf)
            best = {g: float(alphas[int(np.argmin(e))]) if np.isfinite(min(e)) else best[g] for g, e in errs.items()}
        A, B = _prepare(X_train, X_pred)
        mu_all = float(np.mean(y_train))
        out = np.empty(len(X_pred))
        for alpha in sorted(set(best.values())):
            groups = [g for g, v in best.items() if v == alpha]
            sel = np.isin(lab_pr, groups)
            if sel.any():
                out[sel] = Ridge(alpha=alpha).fit(A, y_train - mu_all).predict(B[sel]) + mu_all
        unseen = ~np.isin(lab_pr, list(best))  # a group with no training rows keeps the middle alpha
        if unseen.any():
            out[unseen] = Ridge(alpha=float(alphas[len(alphas) // 2])).fit(A, y_train - mu_all).predict(B[unseen]) + mu_all
        if record is not None:
            record.append({"alpha_by_group": {str(g): v for g, v in best.items()}, "n_train": int(len(X_train))})
        return out

    return fit_predict


def hgb_model(
    max_iter: int = 200,
    learning_rate: float = 0.04,
    max_leaf_nodes: int = 8,
    min_samples_leaf: int = 200,
    l2: float = 5.0,
    seed: int = 0,
    row_stride: int = 2,
    early_stopping: bool | str = "auto",
    record: list | None = None,
) -> ModelFn:
    """HistGradientBoosting on every `row_stride`-th training row.

    `early_stopping="auto"` is scikit-learn's default and what production has always used: with more than 10,000 training rows it holds out a
    RANDOM 10% of the rows (not the latest dates, not whole origins), so the iteration count is chosen on a validation set that is a
    time-mixed sample of the training window. That is a statement about how the count is chosen, not evidence of leakage into an outside
    evaluation. `record` (a list) receives {"n_train", "n_iter", "max_iter", "early_stopping_active"} per fit so real runs can report it.
    """

    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame) -> np.ndarray:
        idx = np.arange(0, len(X_train), row_stride)
        # a column that is entirely NaN / constant in this training window (e.g. a survey that starts later) breaks
        # HGB's binning; such a column carries no information here, so leave it out for this fit
        usable = [c for c in X_train.columns if X_train[c].iloc[idx].nunique(dropna=True) >= 2]
        if not usable:
            return np.full(len(X_pred), float(np.mean(y_train)))
        X_train, X_pred = X_train[usable], X_pred[usable]
        model = _hgb(max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2, seed, early_stopping)
        model.fit(X_train.iloc[idx].to_numpy(dtype=np.float32), y_train[idx])
        if record is not None:
            record.append({"n_train": int(len(idx)), "n_iter": int(model.n_iter_), "max_iter": int(max_iter), "early_stopping_active": bool(getattr(model, "do_early_stopping_", False))})
        return model.predict(X_pred.to_numpy(dtype=np.float32))

    return fit_predict


def time_split_masks(dates: pd.DatetimeIndex, purge_weeks: int, val_share: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """(fit rows, validation rows): the newest `val_share` of the distinct dates validate; fit rows end `purge_weeks` before they start,
    so no fit label window reaches into the validation block."""
    uniq = np.sort(pd.DatetimeIndex(dates).unique())
    cut = uniq[int(len(uniq) * (1.0 - val_share))]
    d = pd.DatetimeIndex(dates)
    return np.asarray(d < cut - pd.Timedelta(weeks=purge_weeks)), np.asarray(d >= cut)


def hgb_model_timeval(
    purge_weeks: int,
    max_iter: int = 400,
    learning_rate: float = 0.04,
    max_leaf_nodes: int = 8,
    min_samples_leaf: int = 200,
    l2: float = 5.0,
    seed: int = 0,
    row_stride: int = 2,
    val_share: float = 0.2,
    min_iter: int = 20,
    record: list | None = None,
) -> ModelFn:
    """HGB whose number of iterations is chosen on a purged, time-ordered validation block inside the training window.

    The newest `val_share` of the training dates is the validation block; training rows whose label window (`purge_weeks` = the
    horizon) reaches into it are dropped. The model is grown to `max_iter` without early stopping, the iteration with the lowest
    validation MSE is kept (at least `min_iter`), and the final model is refit on all rows with that many iterations.
    """

    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame) -> np.ndarray:
        usable = [c for c in X_train.columns if X_train[c].iloc[::row_stride].nunique(dropna=True) >= 2]
        if not usable:
            return np.full(len(X_pred), float(np.mean(y_train)))
        Xt, Xp = X_train[usable], X_pred[usable]
        fit_rows, val = time_split_masks(Xt.index.get_level_values("date"), purge_weeks, val_share)
        best = max_iter
        if fit_rows.sum() >= 2000 and val.sum() >= 500:
            tr_idx = np.flatnonzero(fit_rows)[::row_stride]
            va_idx = np.flatnonzero(val)[::row_stride]
            probe = _hgb(max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2, seed, False)
            probe.fit(Xt.iloc[tr_idx].to_numpy(dtype=np.float32), y_train[tr_idx])
            yv = y_train[va_idx]
            errs = [float(np.mean((p - yv) ** 2)) for p in probe.staged_predict(Xt.iloc[va_idx].to_numpy(dtype=np.float32))]
            best = int(max(min_iter, int(np.argmin(errs)) + 1))
        idx = np.arange(0, len(Xt), row_stride)
        final = _hgb(best, learning_rate, max_leaf_nodes, min_samples_leaf, l2, seed, False)
        final.fit(Xt.iloc[idx].to_numpy(dtype=np.float32), y_train[idx])
        if record is not None:
            record.append({"n_train": int(len(idx)), "n_iter": int(best), "max_iter": int(max_iter), "early_stopping_active": False, "selected_on": "purged time-ordered block"})
        return final.predict(Xp.to_numpy(dtype=np.float32))

    return fit_predict


def blend(models: list[ModelFn], weights: list[float] | None = None) -> ModelFn:
    w = np.asarray(weights if weights is not None else [1.0] * len(models), dtype=float)
    w = w / w.sum()

    def fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame) -> np.ndarray:
        return sum(wi * m(X_train, y_train, X_pred) for wi, m in zip(w, models))

    return fit_predict
