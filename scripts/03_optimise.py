"""Step 3: surrogate-in-the-loop design search. Maximise Cl/Cd with a genetic algorithm.

    python scripts/03_optimise.py --data data/airfrans.csv --task full --model svr

Design variables: NACA 4-digit (M, P, XX) and angle of attack, at a fixed Reynolds number.
Bounds are the AirfRANS sampling ranges, so the search never leaves the region the model has seen.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from airfoil_surrogate import plotstyle as ps
from airfoil_surrogate.data import announce, load_dataset, split_masks
from airfoil_surrogate.features import feature_matrix, frame_from_naca4
from airfoil_surrogate.geometry import naca4_contour
from airfoil_surrogate.models import load_bundle, predict_coefficients
from airfoil_surrogate.optimise import run_ga, run_random_search

# Cost of ONE reference CFD solution: ~25 min on 16 cores of an AMD Threadripper 3960X
# (Bonnet et al., NeurIPS 2022, Table 4: "Simulation ~25 min"; whole dataset ~20 days).
CFD_MIN_PER_SIM, CFD_CORES = 25.0, 16

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/airfrans.csv")
ap.add_argument("--task", default="full")
ap.add_argument("--model", default="svr", choices=["random_forest", "svr", "mlp"])
ap.add_argument("--re", type=float, default=4.0e6, help="fixed Reynolds number of the design point")
ap.add_argument("--pop", type=int, default=100)
ap.add_argument("--gens", type=int, default=60)
ap.add_argument("--seeds", type=int, default=5)
args = ap.parse_args()

df = load_dataset(args.data)
source = announce(df)
bundle = load_bundle(Path("models") / source / args.task / f"{args.model}.joblib")
fs = bundle["feature_set"]
out_dir = Path("results") / source / args.task
out_dir.mkdir(parents=True, exist_ok=True)

# NACA-4 sampling ranges of AirfRANS: M in [0,7], P in {0} U [1.5,7], XX in [5,20]; alpha within training range
bounds = [(0.0, 7.0), (1.5, 7.0), (5.0, 20.0), (bundle["alpha_train_min"], bundle["alpha_train_max"])]
var_names = ["M (% camber)", "P (tenths chord)", "XX (% thickness)", "alpha (deg)"]
timers = {"geometry_features": 0.0, "model": 0.0}


def fitness(pop):
    t0 = time.perf_counter()
    frame = frame_from_naca4(pop[:, 0], pop[:, 1], pop[:, 2], pop[:, 3], args.re)
    X, _ = feature_matrix(frame, fs)
    t1 = time.perf_counter()
    cl, cd = predict_coefficients(bundle, X)
    t2 = time.perf_counter()
    timers["geometry_features"] += t1 - t0
    timers["model"] += t2 - t1
    return cl / cd


runs, t_start = [], time.perf_counter()
for s in range(args.seeds):
    runs.append(run_ga(fitness, bounds, pop_size=args.pop, generations=args.gens, seed=s))
wall_all = time.perf_counter() - t_start
wall_one = wall_all / args.seeds
best = max(runs, key=lambda r: r.best_f)
n_evals = best.n_evals
rand = run_random_search(fitness, bounds, n_evals, seed=0)

# ----------------------- cost accounting (stated assumptions) -----------------------
cfd_search_days = n_evals * CFD_MIN_PER_SIM / 60 / 24
dataset_days = len(df) * CFD_MIN_PER_SIM / 60 / 24
train_s = bundle.get("train_seconds", float("nan"))
report = {
    "source": source, "model": args.model, "Re": args.re,
    "n_evaluations_per_search": int(n_evals), "n_searches": args.seeds,
    "surrogate_wall_clock_s_per_search": wall_one,
    "surrogate_time_us_per_evaluation": wall_one / n_evals * 1e6,
    "time_split_s": dict(timers),
    "best_LD_surrogate": best.best_f,
    "best_design": dict(zip(var_names, map(float, best.best_x))),
    "random_search_best_LD_same_budget": rand.best_f,
    "cfd_assumption": f"{CFD_MIN_PER_SIM:g} min per simulation on {CFD_CORES} cores (AirfRANS paper, Table 4)",
    "cfd_in_the_loop_days_for_same_search": cfd_search_days,
    "cfd_core_hours_for_same_search": n_evals * CFD_MIN_PER_SIM / 60 * CFD_CORES,
    "one_off_dataset_cost_days_same_hardware": dataset_days,
    "one_off_training_seconds": train_s,
    "break_even_evaluations": len(df),
    "speedup_per_search_excluding_dataset": cfd_search_days * 86400 / wall_one,
}

# ----------------------- trust diagnostics -----------------------
tr, _ = split_masks(df, args.task)
Xtr, _ = feature_matrix(df[tr], fs)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
Z = (Xtr - mu) / sd
opt_frame = frame_from_naca4(*[np.array([v]) for v in best.best_x], args.re)
z_opt = (feature_matrix(opt_frame, fs)[0] - mu) / sd
d_opt = float(np.min(np.linalg.norm(Z - z_opt, axis=1)))
sub = Z[np.random.default_rng(0).choice(len(Z), min(300, len(Z)), replace=False)]
loo = [np.sort(np.linalg.norm(Z - z, axis=1))[1] for z in sub]
report["distance_to_nearest_training_point"] = d_opt
report["typical_nearest_neighbour_distance_p95"] = float(np.percentile(loo, 95))
report["bounds_hit"] = [n for n, v, (lo, hi) in zip(var_names, best.best_x, bounds)
                        if min(v - lo, hi - v) < 0.02 * (hi - lo)]
Path(out_dir / f"optimisation_{args.model}.json").write_text(json.dumps(report, indent=2))


def human(seconds):
    for unit, s in (("years", 3.156e7), ("days", 86400), ("hours", 3600), ("min", 60)):
        if seconds >= s:
            return f"{seconds / s:.1f} {unit}"
    return f"{seconds:.1f} s"


# ----------------------- figure -----------------------
ps.apply()
fig = plt.figure(figsize=(12.5, 4.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.8], wspace=0.32)
ax = fig.add_subplot(gs[0])
for r in runs:
    ax.plot(r.history["evals"], r.history["best"], color=ps.MODEL_COLORS["random_forest"], lw=0.9, alpha=0.35)
grid = best.history["evals"]
med = np.median([np.interp(grid, r.history["evals"], r.history["best"]) for r in runs], axis=0)
ax.plot(grid, med, color=ps.MODEL_COLORS["random_forest"], lw=2.4, label=f"genetic algorithm (median of {args.seeds} runs)")
ax.plot(rand.history["evals"], rand.history["best"], color=ps.MUTED, lw=1.8, ls="--", label="random search, same budget")
ax.set_xlabel("surrogate evaluations")
ax.set_ylabel("best predicted $C_l/C_d$ so far")
ax.set_title("Convergence")
ax.legend(loc="lower right")

ax = fig.add_subplot(gs[1])
ref = naca4_contour(0, 0, 12)
opt = naca4_contour(*best.best_x[:3])
ax.fill(ref[:, 0], ref[:, 1], color=ps.GRID, label="NACA 0012")
ax.plot(opt[:, 0], opt[:, 1], color=ps.MODEL_COLORS["random_forest"], lw=2.2,
        label=f"surrogate optimum\nM={best.best_x[0]:.1f} P={best.best_x[1]:.1f} XX={best.best_x[2]:.1f}, "
              f"$\\alpha$={best.best_x[3]:.1f}°")
ax.set_aspect("equal")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.25, 0.3)
ax.grid(False)
ax.set_title("Optimal section")
ax.legend(loc="upper center", fontsize=7.5)

ax = fig.add_subplot(gs[2])
vals = [wall_one, cfd_search_days * 86400]
bars = ax.bar(["surrogate\nsearch", "CFD in\nthe loop"], vals, color=[ps.MODEL_COLORS["random_forest"], ps.MUTED], width=0.55)
ax.set_yscale("log")
ax.set_ylabel("wall-clock time (s, log scale)")
ax.set_title(f"{n_evals:,} evaluations")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.3, human(v), ha="center", fontsize=9, color=ps.INK)
ax.set_ylim(top=max(vals) * 12)
ax.grid(axis="x", visible=False)
fig.text(0.01, -0.04,
         f"CFD figure = {n_evals:,} x {CFD_MIN_PER_SIM:g} min (AirfRANS paper: one RANS solution on 16 cores). "
         f"Dataset generation ({human(dataset_days * 86400)}) is a one-off cost not shown. "
         f"Optimum is a surrogate prediction, not CFD-verified.", fontsize=8, color=ps.INK2)
if source == "synthetic":
    fig.text(0.5, 0.5, "SYNTHETIC - CODE TEST", fontsize=40, color=ps.INK, alpha=0.07, ha="center", va="center", rotation=15)
fig.savefig(out_dir / f"optimisation_{args.model}.png", bbox_inches="tight")
plt.close(fig)

print(json.dumps({k: report[k] for k in ("n_evaluations_per_search", "surrogate_wall_clock_s_per_search",
      "surrogate_time_us_per_evaluation", "best_LD_surrogate", "random_search_best_LD_same_budget",
      "cfd_in_the_loop_days_for_same_search", "one_off_dataset_cost_days_same_hardware",
      "break_even_evaluations", "distance_to_nearest_training_point",
      "typical_nearest_neighbour_distance_p95", "bounds_hit")}, indent=2))
print("best design:", report["best_design"])
