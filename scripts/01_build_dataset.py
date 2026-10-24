"""Step 1: build the table of simulations (geometry descriptors, flow condition, Cl, Cd).

Real data (needs internet access to data.isir.upmc.fr, a few GB, and `pip install airfrans`):
    python scripts/01_build_dataset.py --download --root data/airfrans_raw --out data/airfrans.csv

Code test only (synthetic stand-in, never a result):
    python scripts/01_build_dataset.py --synthetic 1000 --out data/synthetic.csv
"""
import argparse

from airfoil_surrogate.data import SYNTHETIC_BANNER, build_airfrans_table, make_synthetic

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/airfrans.csv")
ap.add_argument("--root", default="data/airfrans_raw", help="where the AirfRANS download lives")
ap.add_argument("--download", action="store_true", help="download the pre-processed AirfRANS dataset first")
ap.add_argument("--synthetic", type=int, default=0, metavar="N", help="write N synthetic rows instead")
ap.add_argument("--limit", type=int, default=None, help="only process the first N simulations (debugging)")
ap.add_argument("--n-jobs", type=int, default=2)
args = ap.parse_args()

if args.synthetic:
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    make_synthetic(args.synthetic).to_csv(args.out, index=False)
    print(SYNTHETIC_BANNER, f"wrote {args.out}")
else:
    if args.download:
        import airfrans as af
        af.dataset.download(root=args.root, file_name="Dataset", unzip=True, OpenFOAM=False)
    build_airfrans_table(args.root, args.out, n_jobs=args.n_jobs, limit=args.limit)
