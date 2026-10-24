"""Step 2: train Random Forest, SVR and MLP; nested grouped CV; held-out test; MAE by AoA band.

    python scripts/02_train.py --data data/airfrans.csv --task full
    python scripts/02_train.py --data data/airfrans.csv --task aoa     # extrapolation in angle of attack
    add --fast for a quick pass.

Outputs: models/<source>/<task>/*.joblib, results/<source>/<task>/{metrics.csv,metrics.md,predictions.csv,*.png}
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airfoil_surrogate import plotstyle as ps
from airfoil_surrogate.data import announce, load_dataset, split_masks
from airfoil_surrogate.evaluate import HIGH_ALPHA_DEG, banded_metrics, nested_oof_predictions
from airfoil_surrogate.features import feature_matrix, targets, to_coefficients
from airfoil_surrogate.models import candidates, fit_tuned, save_bundle

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/airfrans.csv")
ap.add_argument("--task", default="full", choices=["full", "scarce", "aoa", "reynolds"])
ap.add_argument("--features", default="physics", choices=["physics", "cst"])
ap.add_argument("--fast", action="store_true")
ap.add_argument("--n-outer", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-jobs", type=int, default=-1)
args = ap.parse_args()

df = load_dataset(args.data)
source = announce(df)
tr, te = split_masks(df, args.task)
X, names = feature_matrix(df, args.features)
Y = targets(df)
groups = df["geometry_id"].to_numpy()
tag = f"{args.task}" + ("" if args.features == "physics" else f"_{args.features}")
models_dir = Path("models") / source / tag
out_dir = Path("results") / source / tag
models_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
print(f"source={source} task={args.task} features={args.features}: train={tr.sum()} test={te.sum()} "
      f"| alpha(train) {df.alpha_deg[tr].min():.1f}..{df.alpha_deg[tr].max():.1f} deg")

rows, preds = [], df.loc[te, ["sim_id", "alpha_deg", "Re", "Cl", "Cd"]].copy()
n_inner = 2 if args.fast else 3
for name, (est, grid) in candidates(fast=args.fast, seed=args.seed, n_jobs=args.n_jobs).items():
    t0 = time.perf_counter()
    oof = nested_oof_predictions(est, grid, X[tr], Y[tr], groups[tr],
                                 n_outer=3 if args.fast else args.n_outer, n_inner=n_inner)
    cv = banded_metrics(df[tr], *to_coefficients(oof))
    gs = fit_tuned(est, grid, X[tr], Y[tr], groups[tr], n_inner=n_inner)
    train_s = time.perf_counter() - t0
    cl_p, cd_p = to_coefficients(gs.predict(X[te]))
    test = banded_metrics(df[te], cl_p, cd_p)
    for split_name, tab in (("cv", cv), ("test", test)):
        for _, r in tab.iterrows():
            rows.append({"model": name, "split": split_name, **r.to_dict()})
    preds[f"Cl_{name}"], preds[f"Cd_{name}"] = cl_p, cd_p
    save_bundle(models_dir / f"{name}.joblib", gs.best_estimator_, args.features, names, source, args.task,
                X[tr], extra={"best_params": gs.best_params_, "train_seconds": train_s,
                              "alpha_train_max": float(df.alpha_deg[tr].max()),
                              "alpha_train_min": float(df.alpha_deg[tr].min())})
    print(f"  {name:14s} done in {train_s:6.1f}s  best={gs.best_params_}")

res = pd.DataFrame(rows)
res.to_csv(out_dir / "metrics.csv", index=False)
preds.to_csv(out_dir / "predictions.csv", index=False)

cols = ["model", "split", "band", "n", "MAE_Cl", "MAE_Cd_counts", "MAPE_Cd_pct", "spearman_Cl", "spearman_Cd", "spearman_LD"]
lines = [f"# Metrics ({source}, task={args.task}, features={args.features})", "",
         f"High angle of attack means alpha >= {HIGH_ALPHA_DEG:g} deg. 1 drag count = 1e-4 in Cd.",
         "cv = nested 5-fold CV on the training set (grouped by geometry); test = held-out test set.", ""]
if source == "synthetic":
    lines.insert(1, "**SYNTHETIC STAND-IN DATA. Code test only. Not a result.**")
lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for _, r in res[cols].iterrows():
    lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
(out_dir / "metrics.md").write_text("\n".join(lines) + "\n")

# ------------------------------ figures ------------------------------
ps.apply()
mnames = list(candidates(fast=True).keys())
fig, axes = plt.subplots(2, 3, figsize=(11.5, 7), constrained_layout=True)
for j, m in enumerate(mnames):
    for i, (coef, lab) in enumerate((("Cl", "$C_l$"), ("Cd", "$C_d$"))):
        ax = axes[i, j]
        yt, yp = preds[coef], preds[f"{coef}_{m}"]
        sc = ax.scatter(yt, yp, c=preds["alpha_deg"], cmap=ps.SEQ_CMAP, s=14, edgecolor="none", alpha=0.9,
                        vmin=df.alpha_deg.min(), vmax=df.alpha_deg.max())
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], color=ps.MUTED, lw=1, ls="--")
        ax.set_xlabel(f"CFD {lab}")
        ax.set_ylabel(f"predicted {lab}")
        if i == 0:
            ax.set_title(ps.MODEL_LABELS[m])
cbar = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02)
cbar.set_label("angle of attack (deg)")
fig.suptitle(f"Held-out test set ({source}, task={args.task})", x=0.01, ha="left", fontsize=11)
if source == "synthetic":
    fig.text(0.5, 0.5, "SYNTHETIC - CODE TEST", fontsize=34, color=ps.INK, alpha=0.07, ha="center", va="center", rotation=20)
fig.savefig(out_dir / "parity_test.png")
plt.close(fig)

# error vs angle of attack: where does each model get worse?
bins = np.arange(np.floor(df.alpha_deg[te].min()), np.ceil(df.alpha_deg[te].max()) + 2.5, 2.5)
mid = 0.5 * (bins[:-1] + bins[1:])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, coef, lab, scale in ((axes[0], "Cl", "MAE of $C_l$", 1.0), (axes[1], "Cd", "MAE of $C_d$ (drag counts)", 1e4)):
    for m in mnames:
        err = (preds[coef] - preds[f"{coef}_{m}"]).abs() * scale
        idx = np.digitize(preds["alpha_deg"], bins) - 1
        vals = [err[idx == k].mean() if (idx == k).sum() >= 3 else np.nan for k in range(len(mid))]
        ax.plot(mid, vals, marker="o", ms=4, color=ps.MODEL_COLORS[m], label=ps.MODEL_LABELS[m])
    ax.axvline(HIGH_ALPHA_DEG, color=ps.MUTED, lw=1, ls=":")
    ax.set_xlabel("angle of attack (deg)")
    ax.set_ylabel(lab)
axes[0].legend()
axes[0].set_title("Error by angle of attack")
fig.savefig(out_dir / "error_vs_alpha.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_dir}/metrics.md, predictions.csv, parity_test.png, error_vs_alpha.png; models in {models_dir}")
