"""Turn (geometry, flow condition) into the numeric input vector the models see."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import geometry_row, naca4_surfaces

# Geometry descriptors, each with a direct aerodynamic meaning (see docs/physics_primer.md)
GEOM_PHYSICS = ["t_max", "m_max", "x_m", "z75", "alpha_L0_deg"]
GEOM_CST = [f"cst_{s}{i}" for s in ("u", "l") for i in range(6)]

FEATURE_SETS = {
    # alpha, log10(Re), 5 geometry descriptors, and alpha - alpha_L0 (thin-airfoil effective AoA)
    "physics": ["alpha_deg", "log10_Re"] + GEOM_PHYSICS + ["alpha_eff_deg"],
    # same flow inputs, but the shape is given by 12 CST coefficients
    "cst": ["alpha_deg", "log10_Re"] + GEOM_CST,
}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log10_Re"] = np.log10(out["Re"])
    out["alpha_eff_deg"] = out["alpha_deg"] - out["alpha_L0_deg"]
    return out


def feature_matrix(df: pd.DataFrame, feature_set: str = "physics"):
    names = FEATURE_SETS[feature_set]
    df = add_derived(df)
    return df[names].to_numpy(dtype=float), names


def targets(df: pd.DataFrame) -> np.ndarray:
    """Model targets: [Cl, log10(Cd)]. Drag spans a decade or more and must stay positive."""
    return np.column_stack([df["Cl"].to_numpy(float), np.log10(df["Cd"].to_numpy(float))])


def to_coefficients(y: np.ndarray):
    """Inverse of `targets`: returns (Cl, Cd) in physical units."""
    y = np.atleast_2d(y)
    return y[:, 0], 10.0 ** y[:, 1]


def frame_from_naca4(m_pct, p_tenths, t_pct, alpha_deg, re) -> pd.DataFrame:
    """Build the model-input table for NACA 4-digit shapes (arrays broadcast together)."""
    m, p, t, a, r = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in (m_pct, p_tenths, t_pct, alpha_deg, re)])
    cache: dict = {}
    rows = []
    for mi, pi_, ti in zip(m.ravel(), p.ravel(), t.ravel()):
        key = (round(float(mi), 9), round(float(pi_), 9), round(float(ti), 9))
        if key not in cache:
            cache[key] = geometry_row(*naca4_surfaces(*key))
        rows.append(cache[key])
    df = pd.DataFrame(rows)
    df["alpha_deg"] = a.ravel()
    df["Re"] = r.ravel()
    return df
