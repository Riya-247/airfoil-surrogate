"""Step 4: NACA 0012 lift curve, surrogate vs experiment, extrapolated through stall.

    python scripts/fetch_experimental.py              # once, on a machine with internet access
    python scripts/04_validate_naca0012.py --data data/airfrans.csv --task full

Shows, for the same trained models: (a) Cl(alpha), (b) Cd(alpha), (c) the random forest's own
uncertainty estimate (spread across trees) next to its actual error against experiment.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from airfoil_surrogate import plotstyle as ps
from airfoil_surrogate.data import announce, load_dataset
from airfoil_surrogate.experimental import load_all
from airfoil_surrogate.features import feature_matrix, frame_from_naca4
from airfoil_surrogate.models import load_bundle, predict_coefficients

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/airfrans.csv")
ap.add_argument("--task", default="full")
ap.add_argument("--re", type=float, default=6.0e6, help="Ladson tripped data are at Re = 6e6")
ap.add_argument("--alpha-max", type=float, default=22.0)
ap.add_argument("--experimental", default="data/experimental")
ap.add_argument("--ladson-cols", type=int, nargs=3, default=[0, 1, 2], metavar=("ALPHA", "CL", "CD"),
                help="column indices in CLCD_Ladson_expdata.dat (verify against the printed rows!)")
ap.add_argument("--all-experiments", action="store_true", help="also draw Abbott and Gregory data")
args = ap.parse_args()

df = load_dataset(args.data)
source = announce(df)
out_dir = Path("results") / source / args.task
out_dir.mkdir(parents=True, exist_ok=True)
bundles = {m: load_bundle(Path("models") / source / args.task / f"{m}.joblib") for m in ("random_forest", "svr", "mlp")}
a_train_max = bundles["random_forest"]["alpha_train_max"]

alphas = np.arange(-5.0, args.alpha_max + 1e-9, 0.5)
frame = frame_from_naca4(0.0, 0.0, 12.0, alphas, args.re)
X, names = feature_matrix(frame, bundles["random_forest"]["feature_set"])
i_re = names.index("log10_Re")
b = bundles["random_forest"]
if not (b["train_min"][i_re] <= np.log10(args.re) <= b["train_max"][i_re] + 0.01):
    print(f"WARNING: Re={args.re:.2e} is outside the training Reynolds range.")
pred = {m: predict_coefficients(bd, X) for m, bd in bundles.items()}

# random-forest spread across trees: the model's own idea of how unsure it is
rf_model = bundles["random_forest"]["model"]
tree_cl = np.stack([rf_model.transformer_.inverse_transform(t.predict(X.astype(np.float32)))[:, 0]
                    for t in rf_model.regressor_.estimators_])
rf_sigma = tree_cl.std(axis=0)

exp = load_all(args.experimental)
ladson = exp.get("ladson")
report = {"source": source, "Re": args.re, "alpha_train_max": a_train_max, "experimental_files_found": sorted(exp)}
if ladson is not None:
    ia, icl, icd = args.ladson_cols
    print(f"Ladson file: {len(ladson)} rows, {ladson.shape[1]} columns. First rows read:\n{ladson[:3]}\n"
          f"Using columns alpha={ia}, CL={icl}, CD={icd if ladson.shape[1] > icd else 'n/a'}. "
          f"If this is wrong, pass --ladson-cols.")
    ea, ecl = ladson[:, ia], ladson[:, icl]
    Xe, _ = feature_matrix(frame_from_naca4(0.0, 0.0, 12.0, ea, args.re), bundles["random_forest"]["feature_set"])
    inside, outside = ea <= a_train_max, ea > a_train_max
    for m, bd in bundles.items():
        cl_e, _ = predict_coefficients(bd, Xe)
        report[f"{m}_MAE_Cl_vs_experiment"] = {
            f"alpha<={a_train_max:g}": float(np.abs(cl_e - ecl)[inside].mean()) if inside.any() else None,
            f"alpha>{a_train_max:g}": float(np.abs(cl_e - ecl)[outside].mean()) if outside.any() else None}
else:
    print("No experimental files in", args.experimental, "- run scripts/fetch_experimental.py. Plotting surrogates only.")

# CFD labels of near-symmetric sections at similar Reynolds number (dataset's own RANS "truth", not experiment)
near = df[(df.m_max.abs() < 0.0015) & df.t_max.between(0.11, 0.13) & df.Re.between(0.8 * args.re, 1.1 * args.re)]

ps.apply()
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.5, 1.2, 1.0]})


def shade(ax):
    ax.axvspan(a_train_max, args.alpha_max, color=ps.EXTRAP_SHADE, alpha=0.55, lw=0)


ax = axes[0]
shade(ax)
for m in bundles:
    ax.plot(alphas, pred[m][0], color=ps.MODEL_COLORS[m], label=ps.MODEL_LABELS[m])
ax.fill_between(alphas, pred["random_forest"][0] - 2 * rf_sigma, pred["random_forest"][0] + 2 * rf_sigma,
                color=ps.MODEL_COLORS["random_forest"], alpha=0.25, lw=0)
if len(near):
    ax.scatter(near.alpha_deg, near.Cl, s=14, color=ps.MUTED, edgecolor="none", label="CFD labels, near-symmetric sections")
if ladson is not None:
    ax.scatter(ladson[:, args.ladson_cols[0]], ladson[:, args.ladson_cols[1]], s=22, color=ps.INK, zorder=5,
               label="Ladson experiment (NASA TM 4074)")
if args.all_experiments:
    for key, lab, mk in (("abbott_cl", "Abbott & von Doenhoff", "s"), ("gregory_cl", "Gregory & O'Reilly (Re 3e6)", "^")):
        if key in exp:
            ax.scatter(exp[key][:, 0], exp[key][:, 1], s=14, marker=mk, facecolor="none", edgecolor=ps.INK2, label=lab)
ax.text(args.alpha_max - 0.2, ax.get_ylim()[0] + 0.05, "beyond training data\n(no separated flow)", fontsize=8,
        color=ps.INK2, ha="right", va="bottom")
ax.set_xlabel("angle of attack (deg)")
ax.set_ylabel("$C_l$")
ax.set_title(f"NACA 0012 lift curve, Re = {args.re / 1e6:g} million")
ax.legend(loc="upper left", fontsize=7.5)

ax = axes[1]
shade(ax)
for m in bundles:
    ax.plot(alphas, pred[m][1], color=ps.MODEL_COLORS[m], label=ps.MODEL_LABELS[m])
if ladson is not None and ladson.shape[1] > args.ladson_cols[2]:
    ax.scatter(ladson[:, args.ladson_cols[0]], ladson[:, args.ladson_cols[2]], s=22, color=ps.INK, zorder=5, label="Ladson experiment")
ax.set_xlabel("angle of attack (deg)")
ax.set_ylabel("$C_d$")
ax.set_title("Drag")
lo, hi = ax.get_ylim()
ax.set_ylim(lo, hi + 0.3 * (hi - lo))   # headroom so the legend never sits on data
ax.legend(loc="upper left", fontsize=7.5)

ax = axes[2]
shade(ax)
ax.plot(alphas, 2 * rf_sigma, color=ps.MODEL_COLORS["random_forest"], label="RF's own uncertainty (2$\\sigma$ over trees)")
if ladson is not None:
    err = np.abs(predict_coefficients(bundles["random_forest"], Xe)[0] - ecl)
    ax.scatter(ea, err, s=22, color=ps.INK, zorder=5, label="actual RF error vs experiment")
ax.set_xlabel("angle of attack (deg)")
ax.set_ylabel("$C_l$ uncertainty / error")
ax.set_title("Confident and wrong")
lo, hi = ax.get_ylim()
ax.set_ylim(lo, hi + 0.3 * (hi - lo))
ax.legend(loc="upper left", fontsize=7.5)
if source == "synthetic":
    fig.text(0.5, 0.5, "SYNTHETIC - CODE TEST", fontsize=44, color=ps.INK, alpha=0.07, ha="center", va="center", rotation=12)
fig.savefig(out_dir / "validation_naca0012.png", bbox_inches="tight")
plt.close(fig)

report["rf_sigma_Cl_inside_training_range_mean"] = float(rf_sigma[alphas <= a_train_max].mean())
report["rf_sigma_Cl_beyond_training_range_mean"] = float(rf_sigma[alphas > a_train_max].mean())
for m in bundles:
    report[f"{m}_Cl_at_alpha"] = {f"{a:g}": float(pred[m][0][np.argmin(np.abs(alphas - a))]) for a in (10, 15, 18, 20, 22)
                                  if a <= args.alpha_max}
Path(out_dir / "validation_naca0012.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
