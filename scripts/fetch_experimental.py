"""Download the NACA 0012 experimental data files from the NASA Turbulence Modeling Resource."""
import argparse
import urllib.request
from pathlib import Path

from airfoil_surrogate.experimental import BASE_URL, FILES

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/experimental")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for fname in FILES.values():
    dest = out / fname
    urllib.request.urlretrieve(BASE_URL + fname, dest)
    first = dest.read_text(errors="ignore").splitlines()[:4]
    print(f"{fname}: {dest.stat().st_size} bytes; first lines: {first}")
print("Check the first lines above: which column is alpha, CL, CD? The validation script prints them again.")
