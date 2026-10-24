"""Reader for the NASA Turbulence Modeling Resource NACA 0012 experimental files.

Files (downloaded by scripts/fetch_experimental.py from tmbwg.github.io/turbmodels/NACA0012_validation/):
  CLCD_Ladson_expdata.dat   Ladson (NASA TM 4074), tripped, Re = 6e6   <- the right one for fully turbulent RANS
  0012.abbottdata.cl.dat    Abbott & von Doenhoff, untripped (digitised)
  CL_Gregory_expdata.dat    Gregory & O'Reilly, tripped, Re = 3e6 (digitised)
  0012.mccroskeydata.cl.dat McCroskey best-fit CL(alpha) line

The Ladson file has columns alpha (deg), Cl, Cd. `read_table` parses every numeric row and the
validation script prints the first rows it read so the column mapping can be confirmed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FILES = {
    "ladson": "CLCD_Ladson_expdata.dat",
    "abbott_cl": "0012.abbottdata.cl.dat",
    "gregory_cl": "CL_Gregory_expdata.dat",
    "mccroskey_cl": "0012.mccroskeydata.cl.dat",
}
BASE_URL = "https://tmbwg.github.io/turbmodels/NACA0012_validation/"


def read_table(path) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.replace(",", " ").split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue  # header / comment line
    if not rows:
        return None
    width = min(len(r) for r in rows)
    return np.array([r[:width] for r in rows])


def load_all(directory) -> dict[str, np.ndarray]:
    out = {}
    for key, fname in FILES.items():
        tab = read_table(Path(directory) / fname)
        if tab is not None:
            out[key] = tab
    return out
