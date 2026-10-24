"""Plot style: one validated categorical order, one sequential ramp, recessive chrome."""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

# categorical slots 1-3 (validated for all-pairs use); same entity -> same colour in every figure
MODEL_COLORS = {"random_forest": "#2a78d6", "svr": "#eb6834", "mlp": "#1baf7a"}
MODEL_LABELS = {"random_forest": "Random Forest", "svr": "SVR (RBF)", "mlp": "MLP"}
# steps 250-700 of the blue ramp; the lightest steps would vanish on the light surface
SEQ = ["#86b6ef", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)
EXTRAP_SHADE = "#e1e0d9"


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "font.family": "sans-serif", "lines.linewidth": 2.0, "figure.dpi": 110, "savefig.dpi": 160,
    })
