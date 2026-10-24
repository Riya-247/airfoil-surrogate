"""A small real-coded genetic algorithm (tournament selection, SBX crossover, polynomial mutation).

Written out in plain NumPy so every step is visible. The fitness function receives a whole
population at once (an array of shape [pop, n_vars]) and returns one score per individual,
higher is better.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GAResult:
    best_x: np.ndarray
    best_f: float
    n_evals: int
    history: dict = field(default_factory=dict)  # per-generation arrays


def run_ga(fitness, bounds, pop_size=100, generations=60, seed=0, eta_c=15.0, eta_m=20.0,
           p_cross=0.9, tournament=3, elite=2) -> GAResult:
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(bounds, dtype=float).T
    n = len(lo)
    pop = lo + rng.random((pop_size, n)) * (hi - lo)
    f = np.asarray(fitness(pop), dtype=float)
    n_evals = pop_size
    hist = {"evals": [n_evals], "best": [f.max()], "mean": [f.mean()], "x_best": [pop[f.argmax()].copy()]}

    def select():
        idx = rng.integers(0, pop_size, size=(pop_size, tournament))
        return pop[idx[np.arange(pop_size), f[idx].argmax(axis=1)]]

    for _ in range(generations):
        parents = select()
        kids = parents.copy()
        for i in range(0, pop_size - 1, 2):
            if rng.random() < p_cross:
                u = rng.random(n)
                beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
                a, b = parents[i], parents[i + 1]
                kids[i] = 0.5 * ((1 + beta) * a + (1 - beta) * b)
                kids[i + 1] = 0.5 * ((1 - beta) * a + (1 + beta) * b)
        # polynomial mutation, probability 1/n per gene
        mask = rng.random(kids.shape) < 1.0 / n
        u = rng.random(kids.shape)
        delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
        kids = np.where(mask, kids + delta * (hi - lo), kids)
        kids = np.clip(kids, lo, hi)
        # elitism: carry the best individuals over unchanged
        order = np.argsort(f)[::-1][:elite]
        kids[:elite] = pop[order]
        f_kids = np.empty(pop_size)
        f_kids[:elite] = f[order]
        f_kids[elite:] = np.asarray(fitness(kids[elite:]), dtype=float)
        n_evals += pop_size - elite
        pop, f = kids, f_kids
        hist["evals"].append(n_evals)
        hist["best"].append(f.max())
        hist["mean"].append(f.mean())
        hist["x_best"].append(pop[f.argmax()].copy())

    i = int(f.argmax())
    return GAResult(pop[i].copy(), float(f[i]), n_evals, {k: np.asarray(v) for k, v in hist.items()})


def run_random_search(fitness, bounds, n_evals, seed=0, batch=100) -> GAResult:
    """Baseline with the same evaluation budget as the GA: best-so-far of uniform random samples."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(bounds, dtype=float).T
    best_f, best_x, evals, best_hist = -np.inf, None, [], []
    done = 0
    while done < n_evals:
        k = min(batch, n_evals - done)
        x = lo + rng.random((k, len(lo))) * (hi - lo)
        f = np.asarray(fitness(x), dtype=float)
        if f.max() > best_f:
            best_f, best_x = float(f.max()), x[f.argmax()].copy()
        done += k
        evals.append(done)
        best_hist.append(best_f)
    return GAResult(best_x, best_f, done, {"evals": np.asarray(evals), "best": np.asarray(best_hist)})
