"""Metrics: MAE overall and by angle-of-attack band, plus rank correlation (what design needs)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from .models import tuned

HIGH_ALPHA_DEG = 10.0  # "high angle of attack" threshold. AirfRANS only goes up to 15 deg.


def _rho(a, b):
    return float(spearmanr(a, b).statistic) if len(a) > 2 else np.nan


def metrics_for(df: pd.DataFrame, cl_pred, cd_pred) -> dict:
    cl, cd = df["Cl"].to_numpy(), df["Cd"].to_numpy()
    ld, ld_pred = cl / cd, cl_pred / cd_pred
    return {
        "n": len(df),
        "MAE_Cl": float(np.mean(np.abs(cl - cl_pred))),
        "MAE_Cd_counts": float(np.mean(np.abs(cd - cd_pred)) * 1e4),   # 1 drag count = 1e-4
        "MAPE_Cd_pct": float(np.mean(np.abs(cd - cd_pred) / cd) * 100),
        "MAE_LD": float(np.mean(np.abs(ld - ld_pred))),
        "spearman_Cl": _rho(cl, cl_pred),
        "spearman_Cd": _rho(cd, cd_pred),
        "spearman_LD": _rho(ld, ld_pred),
    }


def banded_metrics(df: pd.DataFrame, cl_pred, cd_pred, thr: float = HIGH_ALPHA_DEG) -> pd.DataFrame:
    """Rows: all, alpha < thr, alpha >= thr."""
    a = df["alpha_deg"].to_numpy()
    bands = {"all": np.ones(len(df), bool), f"alpha<{thr:g}": a < thr, f"alpha>={thr:g}": a >= thr}
    rows = []
    for name, m in bands.items():
        if m.sum() == 0:
            continue
        rows.append({"band": name, **metrics_for(df[m], np.asarray(cl_pred)[m], np.asarray(cd_pred)[m])})
    return pd.DataFrame(rows)


def nested_oof_predictions(estimator, grid, X, Y, groups, n_outer=5, n_inner=3):
    """Out-of-fold predictions where hyper-parameters are re-tuned inside every outer fold.

    Splits are grouped by `groups` (geometry id), so no airfoil is ever in both the
    training and validation side of a fold.
    """
    oof = np.zeros_like(Y, dtype=float)
    for tr, va in GroupKFold(n_splits=n_outer).split(X, Y, groups):
        gs = tuned(estimator, grid, n_inner)
        gs.fit(X[tr], Y[tr], groups=groups[tr])
        oof[va] = gs.predict(X[va])
    return oof
