"""Random Forest, SVR and MLP surrogates, each with a small hyper-parameter grid.

All three predict the same two targets, [Cl, log10(Cd)], and are wrapped so that inputs and
targets are standardised inside the model (a saved model is therefore self-contained).
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

TARGET_NAMES = ["Cl", "log10_Cd"]


def _wrap(reg):
    return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())


def candidates(fast: bool = False, seed: int = 0, n_jobs: int = 1) -> dict:
    """name -> (estimator, param_grid). `fast=True` shrinks grids for quick smoke runs."""
    rf = _wrap(RandomForestRegressor(n_estimators=150 if fast else 400, random_state=seed, n_jobs=n_jobs))
    rf_grid = {"regressor__min_samples_leaf": [1, 3], "regressor__max_features": [0.6, 1.0]}

    svr = _wrap(Pipeline([("x", StandardScaler()), ("m", MultiOutputRegressor(SVR(kernel="rbf")))]))
    svr_grid = {"regressor__m__estimator__C": [3, 30] if fast else [1, 10, 100],
                "regressor__m__estimator__gamma": [0.05, 0.2] if fast else [0.02, 0.05, 0.2],
                "regressor__m__estimator__epsilon": [0.02, 0.1]}

    mlp = _wrap(Pipeline([("x", StandardScaler()),
                          ("m", MLPRegressor(max_iter=2000, early_stopping=True, n_iter_no_change=30,
                                             random_state=seed))]))
    mlp_grid = {"regressor__m__hidden_layer_sizes": [(64, 64)] if fast else [(64, 64), (128, 128), (64, 64, 64)],
                "regressor__m__alpha": [1e-4, 1e-2]}

    return {"random_forest": (rf, rf_grid), "svr": (svr, svr_grid), "mlp": (mlp, mlp_grid)}


def tuned(estimator, grid, n_inner: int = 3, n_jobs: int = 1) -> GridSearchCV:
    return GridSearchCV(estimator, grid, cv=GroupKFold(n_splits=n_inner),
                        scoring="neg_mean_absolute_error", n_jobs=n_jobs, refit=True)


def fit_tuned(estimator, grid, X, Y, groups, n_inner: int = 3, n_jobs: int = 1) -> GridSearchCV:
    gs = tuned(estimator, grid, n_inner, n_jobs)
    gs.fit(X, Y, groups=groups)
    return gs


def save_bundle(path, model, feature_set, feature_names, source, task, X_train, extra=None):
    bundle = {
        "model": model,
        "feature_set": feature_set,
        "feature_names": feature_names,
        "target_names": TARGET_NAMES,
        "source": source,
        "task": task,
        "train_min": np.asarray(X_train).min(axis=0),
        "train_max": np.asarray(X_train).max(axis=0),
        "n_train": int(len(X_train)),
        **(extra or {}),
    }
    joblib.dump(bundle, path)


def load_bundle(path):
    return joblib.load(path)


def predict_coefficients(bundle, X):
    """Return (Cl, Cd) in physical units."""
    y = bundle["model"].predict(X)
    y = np.atleast_2d(y)
    return y[:, 0], 10.0 ** y[:, 1]
