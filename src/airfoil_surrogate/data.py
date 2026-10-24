"""Dataset table: schema, loading, AirfRANS extraction, and a synthetic stand-in for testing.

The rest of the pipeline reads only the CSV table described by `REQUIRED`, so any CFD or
XFOIL dataset can be plugged in by producing the same columns.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import (NU_AIR, contour_to_surfaces, geometry_row, naca4_surfaces,
                       reynolds_from_velocity)

REQUIRED = ["sim_id", "source", "geometry_id", "Re", "alpha_deg", "Cl", "Cd",
            "t_max", "m_max", "x_m", "z75", "alpha_L0_deg"]
SPLIT_COLS = ["split_full", "split_scarce", "split_aoa", "split_reynolds"]

SYNTHETIC_BANNER = (
    "\n" + "!" * 78 + "\n"
    "  SYNTHETIC STAND-IN DATA. Labels come from a thin-airfoil formula, NOT from CFD.\n"
    "  This run only proves the code works. Nothing it prints or plots is a result.\n"
    + "!" * 78 + "\n"
)


def load_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if (df["Cd"] <= 0).any():
        raise ValueError("Non-positive Cd found; check the extraction.")
    return df


def announce(df: pd.DataFrame) -> str:
    """Return the data source tag and print a loud warning if the data is synthetic."""
    src = str(df["source"].iloc[0])
    if src == "synthetic":
        print(SYNTHETIC_BANNER, file=sys.stderr)
    return src


def split_masks(df: pd.DataFrame, task: str):
    """Boolean (train, test) masks for an AirfRANS task: full | scarce | aoa | reynolds."""
    if task == "scarce":
        train = df["split_scarce"] == "train"
        test = df["split_full"] == "test"
    else:
        col = f"split_{task}"
        if col not in df.columns:
            raise ValueError(f"No column {col}; tasks available: full, scarce, aoa, reynolds")
        train, test = df[col] == "train", df[col] == "test"
    return train.to_numpy(), test.to_numpy()


# --------------------------------------------------------------------------
# Synthetic stand-in (code testing only)
# --------------------------------------------------------------------------
def make_synthetic(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        m = rng.uniform(0, 7)
        p = rng.uniform(0, 7)
        p = 0.0 if p < 1.5 else p
        m = 0.0 if p == 0.0 else m
        t = rng.uniform(5, 20)
        u = rng.uniform(31, 94)  # m/s -> Re about 2e6 .. 6e6
        alpha = rng.uniform(-5, 15)
        g = geometry_row(*naca4_surfaces(m, p, t))
        a_eff = alpha - g["alpha_L0_deg"]
        cl = 0.93 * 0.1097 * (1 + 0.77 * g["t_max"]) * a_eff - 0.0006 * max(a_eff - 10, 0) ** 2
        re = float(reynolds_from_velocity(u))
        cd0 = 0.0045 * (1 + 2.2 * g["t_max"] + 60 * g["t_max"] ** 4) * (re / 4e6) ** -0.12
        cd = (cd0 + 0.0075 * (a_eff / 10) ** 2) * np.exp(rng.normal(0, 0.03))
        cl += rng.normal(0, 0.01)
        rows.append(dict(sim_id=f"synthetic_{i:04d}", source="synthetic", geometry_id=f"synthetic_{i:04d}",
                         family="naca4", p1=m, p2=p, p3=t, p4=np.nan, U=u, Re=re, alpha_deg=alpha,
                         Cl=cl, Cd=cd, **g))
    df = pd.DataFrame(rows)
    is_test = rng.random(n) < 0.2
    df["split_full"] = np.where(is_test, "test", "train")
    scarce = df.index[df["split_full"] == "train"][:200]
    df["split_scarce"] = np.where(df.index.isin(scarce), "train", "")
    df["split_aoa"] = np.where((df["alpha_deg"] >= -2.5) & (df["alpha_deg"] <= 12.5), "train", "test")
    df["split_reynolds"] = np.where((df["Re"] >= 3e6) & (df["Re"] <= 5e6), "train", "test")
    return df


# --------------------------------------------------------------------------
# AirfRANS extraction (run this on a machine that can download the dataset)
# --------------------------------------------------------------------------
def parse_simulation_name(name: str):
    """'airFoil2D_SST_<U>_<alpha deg>_<naca params...>' -> (U, alpha_deg, params)."""
    parts = name.split("_")
    return float(parts[2]), float(parts[3]), [float(v) for v in parts[4:]]


def _find_root(root: str | os.PathLike) -> Path:
    root = Path(root)
    for cand in (root, root / "Dataset"):
        if (cand / "manifest.json").exists():
            return cand
    raise FileNotFoundError(f"manifest.json not found under {root}. Download the dataset first "
                            f"(scripts/01_build_dataset.py --download).")


def _one_simulation(root: str, name: str) -> dict:
    import airfrans as af
    from airfrans.naca_generator import naca_generator

    u, alpha, params = parse_simulation_name(name)
    sim = af.Simulation(root=root, name=name, T=298.15)
    (cd, cdp, cdv), (cl, clp, clv) = sim.force_coefficient(reference=True)
    contour = naca_generator(np.array(params), nb_samples=200, verbose=False)
    geom = geometry_row(*contour_to_surfaces(contour))
    row = dict(sim_id=name, source="airfrans", geometry_id=name,
               family="naca4" if len(params) == 3 else "naca5",
               p1=params[0], p2=params[1], p3=params[2], p4=params[3] if len(params) == 4 else np.nan,
               U=u, Re=float(reynolds_from_velocity(u)), alpha_deg=alpha,
               Cl=float(cl), Cd=float(cd), Cd_pressure=float(cdp), Cd_viscous=float(cdv), Cl_pressure=float(clp))
    row.update(geom)
    return row


def build_airfrans_table(root, out_csv, n_jobs: int = 2, limit: int | None = None) -> pd.DataFrame:
    """Integrate surface forces of every reference simulation into one table of (Cl, Cd).

    AirfRANS stores fields (velocity, pressure, turbulent viscosity) per mesh node, not
    coefficients. `Simulation.force_coefficient(reference=True)` integrates the OpenFOAM
    pressure and wall shear stress over the airfoil surface to give Cd and Cl.
    """
    from joblib import Parallel, delayed

    root = _find_root(root)
    manifest = json.loads((root / "manifest.json").read_text())
    names = sorted({n for split in manifest.values() for n in split})
    if limit:
        names = names[:limit]
    rows = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_one_simulation)(str(root), n) for n in names)
    df = pd.DataFrame(rows)

    def tag(key):
        members = set(manifest.get(key, []))
        return df["sim_id"].map(lambda n: n in members)

    df["split_full"] = np.select([tag("full_train"), tag("full_test")], ["train", "test"], default="")
    df["split_scarce"] = np.where(tag("scarce_train"), "train", "")
    df["split_aoa"] = np.select([tag("aoa_train"), tag("aoa_test")], ["train", "test"], default="")
    df["split_reynolds"] = np.select([tag("reynolds_train"), tag("reynolds_test")], ["train", "test"], default="")

    # sanity checks: catch a wrong tuple order or a broken extraction early
    assert (df["Cd"] > 0).all(), "non-positive Cd: extraction problem"
    assert (df["Cl"].abs() < 3).all(), "implausible Cl: extraction problem"
    assert (df["Cl"].abs() > df["Cd"]).mean() > 0.9, "Cl/Cd look swapped"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(df)} simulations "
          f"(Re {df.Re.min():.2e}..{df.Re.max():.2e}, alpha {df.alpha_deg.min():.1f}..{df.alpha_deg.max():.1f} deg)")
    return df
