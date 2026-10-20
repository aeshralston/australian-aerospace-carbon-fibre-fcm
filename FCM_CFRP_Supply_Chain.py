"""
Title:   FCM CF Supply Chain
Author:  Aeshlein Ralston
Version: 8 (final)

Fuzzy Cognitive Map (FCM) of the Australian aerospace PAN / CF / CFRP
supply chain. Target concept: C6 (Australian Aerospace Component Production).

Model
-----
Sigmoid FCM on [0, 1] using the rescaled inference rule
(Papageorgiou, 2011, Applied Soft Computing 11(1), 500-513):

    A(t+1) = f( (2A(t) - 1) + (2A(t) - 1) W ),   f = logistic sigmoid

Activation 0.5 represents normal operating conditions and is a stable
state: with every driver at 0.5 the whole map stays at 0.5. Above 0.5 means
more than normal, below 0.5 means less. A capability shock clamps a node at
0 (total loss); a risk shock clamps a driver at 1 (maximum disruption).
Near normal conditions a weight w passes on roughly w of a change, so the
weights keep their intended meaning.

Weights
-------
Each causal edge is assigned a linguistic strength on a five-term scale
(very weak 0.1 to very strong 0.9), a sign, and the evidence supporting it.
Each term carries an uncertainty band (+/-0.1) used by the Monte Carlo
analysis.

Exogenous drivers (C7 to C11) have no incoming edges; they are model inputs
and are clamped in every run.

Robustness checks (reported in Chapter 10, not Chapter 7)
---------------------------------------------------------
1. Update rule: the analysis is repeated with the standard modified Kosko rule,
   A(t+1) = f(A(t) + A(t) W), to show how the choice affects results.
2. Weight uncertainty: Monte Carlo resampling of every edge weight within
   its uncertainty band. Signs and map structure are preserved, so this
   tests parameter uncertainty, not structural uncertainty.
Chapter 7 figures show results at the reference weights only.

Convergence
-----------
The current map has no feedback loops, so every run converges quickly.
steady_states() raises an error if a run fails to converge, which matters
once loops are added (e.g. the domestic rCF node for RQ2).

Outputs
-------
figures/ : each figure as vector PDF (for the thesis) and 300 dpi PNG
results/ : fcm_summary.txt (all tables), fcm_checks.txt (model checks and
           structural robustness) and mc_samples.csv (raw MC output)
"""

import itertools
import os
import warnings
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ----------------------------------------------------------------------------
# 1. CONCEPT NODES
# ----------------------------------------------------------------------------

NODES = [
    "C1: Propylene Supply",
    "C2: AN Production",
    "C3: PAN Precursor Production",
    "C4: Carbon Fibre Production",
    "C5: Aerospace Composite Intermediate Material Supply",
    "C6: Australian Aerospace Component Production",
    "C7: Global Oil Prices",
    "C8: Industrial Energy Costs",
    "C9: Technical IP and Qualification Constraints",
    "C10: Geopolitical Tension and Export-Control Intensity",
    "C11: Cold-Chain Transport Disruption",
]

N = len(NODES)
CODES = [name.split(": ")[0] for name in NODES]
NAMES = [name.split(": ")[1] for name in NODES]
LABELS = [f"{c} {n}" for c, n in zip(CODES, NAMES)]

CHAIN_NODES = [0, 1, 2, 3, 4, 5]      # capability nodes (higher = better)
RISK_NODES = [6, 7, 8, 9, 10]         # exogenous drivers (higher = worse)
TARGET = 5                            # C6
SHOCK_NODES = [i for i in range(N) if i != TARGET]

TIMESTEPS = 21                        # iterations shown in trajectory plots
MAX_STEPS = 500                       # iteration cap for steady-state runs
CONVERGENCE_TOL = 1e-6

# Figures with a colour bar or second axis trigger a harmless tight_layout
# warning; the saved figures are unaffected.
warnings.filterwarnings("ignore", message="This figure includes Axes that are "
                        "not compatible with tight_layout")

SHOW_FIGURES = False                   # False for batch export only
SHOW_TITLES = False                   # True for slides; False for the thesis,
                                      # where figure captions carry the title
FIG_DIR = "figures"
RESULTS_DIR = "results"


# ----------------------------------------------------------------------------
# 2. LINGUISTIC WEIGHT SCALE AND CAUSAL EDGES
# ----------------------------------------------------------------------------

WEIGHT_SCALE = {             # term: (weight magnitude, uncertainty band +/-)
    "very weak": (0.1, 0.1),
    "weak": (0.3, 0.1),
    "moderate": (0.5, 0.1),
    "strong": (0.7, 0.1),
    "very strong": (0.9, 0.1),
}

# (source, target, sign, strength, evidence)
# PROVISIONAL marks edges whose previous numeric weight sat between two terms; the term
# shown is a proposed reading and must be confirmed against the evidence.
# Complete every evidence field before submission.

EDGES = [
    # Core supply-chain relationships
    (0, 1, +1, "very strong",
     "PROVISIONAL (previous weight 0.8). AN is produced from propylene by ammoxidation; "
     "no practical substitute feedstock. Source: TO ADD"),
    (1, 2, +1, "very strong", "Source: TO ADD"),
    (2, 3, +1, "very strong", "Source: TO ADD"),
    (3, 4, +1, "strong",
     "PROVISIONAL (previous weight 0.8). Material converters can partly buffer "
     "via alternative fibre suppliers and inventory. Source: TO ADD"),
    (4, 5, +1, "strong",
     "PROVISIONAL (previous weight 0.6). Australian manufacturing depends on "
     "imported qualified intermediate materials. Source: TO ADD"),

    # External disruption relationships
    (6, 0, -1, "moderate", "Source: TO ADD"),
    (6, 1, -1, "strong", "Source: TO ADD"),
    (7, 3, -1, "moderate",
     "PROVISIONAL (previous weight 0.6). Energy costs raise carbonisation cost more "
     "than they cap throughput. Source: TO ADD"),
    (8, 2, -1, "strong", "Source: TO ADD"),
    (8, 4, -1, "moderate",
     "NEW. Aerospace qualification limits substitution of intermediate "
     "material suppliers. Source: TO ADD"),
    (9, 2, -1, "very strong",
     "PROVISIONAL (previous weight 0.8). PAN precursor supply is concentrated in few "
     "countries. Source: TO ADD"),
    (9, 4, -1, "moderate",
     "NEW (replaces C10->C6). Export controls restrict access to imported "
     "CF and intermediate materials. Source: TO ADD"),
    (10, 4, -1, "moderate", "Source: TO ADD"),
]

# Alternative ratings for edges whose evidence sits between two adjacent
# terms. Every combination is tested (rating_alternatives_test).
ALT_TERMS = {
    (0, 1): "strong",
    (3, 4): "very strong",
    (4, 5): "moderate",
    (7, 3): "strong",
    (9, 2): "strong",
}

MIN_WEIGHT = 0.05          # MC samples never flip sign or delete an edge


def edge_weight(edge):
    _, _, sign, term, _ = edge
    return sign * WEIGHT_SCALE[term][0]


def edge_band(edge):
    return WEIGHT_SCALE[edge[3]][1]


def build_weights(weights=None):
    """Adjacency matrix from EDGES, optionally with replacement weights."""
    W = np.zeros((N, N))
    for k, edge in enumerate(EDGES):
        i, j = edge[0], edge[1]
        W[i, j] = edge_weight(edge) if weights is None else weights[k]
    return W


W = build_weights()


# ----------------------------------------------------------------------------
# 3. UPDATE RULES
# ----------------------------------------------------------------------------
# normal  : activation of exogenous drivers under normal conditions
# initial : activation of capability nodes at t = 0 (does not affect the
#           steady state, which depends only on weights and clamps)
# loss    : activation representing total loss of a capability node

MODELS = {
    "rescaled": {
        "label": r"Rescaled rule, $\lambda$ = 1 (primary)",
        "rescaled": True, "lam": 1.0,
        "normal": 0.5, "initial": 0.5, "loss": 0.0,
    },
    "kosko": {
        "label": r"Modified Kosko rule, $\lambda$ = 1 (comparison)",
        "rescaled": False, "lam": 1.0,
        "normal": 0.1, "initial": 0.5, "loss": 0.0,
    },
}

PRIMARY = "rescaled"
RISK_MAX = 1.0

SCENARIOS = {
    "baseline": {"title": "Baseline (normal conditions)", "clamp": {}},
    "energy": {"title": "Severe global energy shock",
               "clamp": {6: RISK_MAX, 7: RISK_MAX}},
    "geopol": {"title": "Severe geopolitical shock",
               "clamp": {9: RISK_MAX}},
    "compound": {"title": "Compound conflict shock",
                 "clamp": {6: RISK_MAX, 7: RISK_MAX, 9: RISK_MAX}},
}
SCEN_KEYS = [k for k in SCENARIOS if k != "baseline"]

GROUPS = {                  # rankings are reported within each node class
    "chokepoint": {"title": "Supply chain stages (C1 to C5)",
                   "nodes": [0, 1, 2, 3, 4]},
    "threat": {"title": "External pressures (C7 to C11)",
               "nodes": [6, 7, 8, 9, 10]},
}


# ----------------------------------------------------------------------------
# 4. FCM ENGINE
# ----------------------------------------------------------------------------

def sigmoid(x, lam=1.0):
    """Logistic squashing function, constrains activation to [0, 1]."""
    return 1.0 / (1.0 + np.exp(-lam * x))


def fcm_step(A, W, model):
    """One update. A may be a single state (N,) or a batch (R, N)."""
    X = 2.0 * A - 1.0 if model["rescaled"] else A
    return sigmoid(X + X @ W, model["lam"])


def clamp_set(model, overrides=None):
    """Exogenous drivers at normal level, with scenario overrides applied."""
    clamps = {i: model["normal"] for i in RISK_NODES}
    clamps.update(overrides or {})
    return clamps


def shock_override(i, model):
    """Severe single-node shock: risk maximised or capability lost."""
    return {i: RISK_MAX if i in RISK_NODES else model["loss"]}


def initial_state(model, n=N):
    A = np.full(n, model["initial"])
    A[RISK_NODES] = model["normal"]
    return A


def simulate(W, model, overrides=None, steps=TIMESTEPS):
    """Activation history for plotting, shape (steps, N)."""
    clamps = clamp_set(model, overrides)
    idx = np.array(list(clamps.keys()))
    val = np.array(list(clamps.values()))

    n = W.shape[0]
    H = np.zeros((steps, n))
    H[0] = initial_state(model, n)
    H[0, idx] = val
    for t in range(1, steps):
        H[t] = fcm_step(H[t - 1], W, model)
        H[t, idx] = val
    return H


def steady_states(W, model, override_list):
    """
    Run a batch of clamp configurations to convergence.

    Returns
    -------
    ndarray, shape (len(override_list), N)
    """
    R, n = len(override_list), W.shape[0]
    mask = np.zeros((R, n), dtype=bool)
    vals = np.zeros((R, n))
    for r, overrides in enumerate(override_list):
        for i, v in clamp_set(model, overrides).items():
            mask[r, i] = True
            vals[r, i] = v

    A = np.where(mask, vals, np.tile(initial_state(model, n), (R, 1)))
    for _ in range(MAX_STEPS):
        A_next = np.where(mask, vals, fcm_step(A, W, model))
        if np.abs(A_next - A).max() < CONVERGENCE_TOL:
            return A_next
        A = A_next

    raise RuntimeError(f"FCM did not converge within {MAX_STEPS} steps "
                       f"({model['label']}). Check for oscillation.")


def sample_weights(rng):
    """Uniform resample of each |w| within its band, sign preserved."""
    w = np.array([edge_weight(e) for e in EDGES])
    band = np.array([edge_band(e) for e in EDGES])
    magnitude = np.clip(np.abs(w) + rng.uniform(-band, band),
                        MIN_WEIGHT, 1.0)
    return build_weights(np.sign(w) * magnitude)


def iterations_to_converge(H, tol=1e-4):
    change = np.abs(np.diff(H, axis=0)).max(axis=1)
    below = np.where(change < tol)[0]
    return int(below[0] + 1) if below.size else None


# ----------------------------------------------------------------------------
# 8. FIGURE STYLE
# ----------------------------------------------------------------------------

FIG_WIDTH = 6.3   # inches, approx. A4 text width, so figures are not rescaled

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})

# Okabe-Ito palette (colour-blind safe); markers keep greyscale legible
NODE_STYLE = {
    0: {"color": "#0072B2", "marker": "o"},
    1: {"color": "#56B4E9", "marker": "s"},
    2: {"color": "#009E73", "marker": "^"},
    3: {"color": "#D55E00", "marker": "v"},
    4: {"color": "#E69F00", "marker": "D"},
    5: {"color": "#000000", "marker": "o"},
}

SCENARIO_STYLE = {
    "baseline": {"color": "#000000", "linestyle": "-", "marker": "o"},
    "energy": {"color": "#0072B2", "linestyle": "--", "marker": "s"},
    "geopol": {"color": "#D55E00", "linestyle": "-.", "marker": "^"},
    "compound": {"color": "#CC79A7", "linestyle": ":", "marker": "D"},
}

MODEL_STYLE = {
    "rescaled": {"color": "#000000", "marker": "o", "mfc": "#000000"},
    "kosko": {"color": "#D55E00", "marker": "^", "mfc": "white"},
}

COL_CAPABILITY = "#0072B2"
COL_RISK = "#D55E00"
COL_GRID = "0.85"


# ----------------------------------------------------------------------------
# 9. PLOTTING HELPERS
# ----------------------------------------------------------------------------

def wrap_label(text, width=30):
    """Break a long node label over two lines so it fits beside the plot."""
    import textwrap
    return "\n".join(textwrap.wrap(text, width, break_on_hyphens=False))


def signed(value, spec="+.3f"):
    """Signed number with a typographic minus sign."""
    return f"{value:{spec}}".replace("-", "\u2212")


def shock_label(i):
    value = RISK_MAX if i in RISK_NODES else PM["loss"]
    return f"{LABELS[i]} = {signed(value, '.0f')}"


def plot_node(ax, t, y, idx):
    style = NODE_STYLE[idx]
    is_target = idx == TARGET
    ax.plot(t, y, color=style["color"], marker=style["marker"],
            markersize=4.0 if is_target else 3.2,
            linewidth=1.8 if is_target else 1.0,
            markerfacecolor=style["color"] if is_target else "white",
            markeredgewidth=0.9, zorder=4 if is_target else 3,
            label=LABELS[idx])


def format_time_axis(ax):
    ax.set_xlim(-0.4, TIMESTEPS - 0.6)
    ax.set_xticks(np.arange(0, TIMESTEPS, 2))
    ax.set_xlabel("Iteration, $t$")
    ax.grid(axis="y", color=COL_GRID, linewidth=0.5)
    ax.set_axisbelow(True)


FIGURE_TITLES = {
    # Chapter 7
    "fig_fcm_network":
        "FCM of the Australian aerospace carbon fibre supply chain",
    "fig_fcm_propagation_geopol":
        "How a severe geopolitical shock spreads through the supply chain",
    "fig_fcm_propagation_energy":
        "How a severe global energy shock spreads through the supply chain",
    "fig_fcm_propagation_compound":
        "How a compound conflict shock spreads through the supply chain",
    "fig_fcm_c6_comparison":
        "Australian aerospace component production (C6) under each scenario",
    "fig_fcm_propagation_matrix":
        "How far each single disruption spreads along the supply chain",
    "fig_fcm_sensitivity_c6":
        "How much each single disruption reduces Australian aerospace "
        "component production (C6)",
    "fig_fcm_sensitivity_slide":
        "What does the most damage to Australian aerospace component production?",
    "fig_mc_rank_stability":
        "How stable the rankings are across 2,000 weight samples",
    "fig_mc_scenarios":
        "Effect of each scenario on C6 across 2,000 weight samples",
    "fig_fcm_model_robustness":
        "Sensitivity rankings under two update rules",
    # Chapter 8
    "fig8_network_rcf":
        "Supply chain model with domestic recovered carbon fibre supply (C12)",
    "fig8_scenario_comparison":
        "Effect of domestic rCF on each disruption scenario",
    "fig8_share_sweep":
        "Share of the C6 loss avoided as the domestic share grows",
    "fig8_chokepoint_mitigation":
        "Each single disruption with and without domestic rCF",
    "fig8_dependency_tests":
        "How the domestic supply's own dependencies reduce its benefit",
    # Chapter 10
    "fig9_systematic_bias":
        "Results when whole groups of weights are over- or understated",
    "fig9_critical_weights":
        "Which individual weights the results depend on most",
}


def save_figure(fig, name):
    if SHOW_TITLES and name in FIGURE_TITLES:
        # Placed just above the existing layout; the tight bounding box used
        # when saving expands to include it, so no subplot positions change.
        size = 15 if fig.get_figwidth() > 8 else 10.5    # slide vs thesis
        fig.suptitle(FIGURE_TITLES[name], y=1.0, va="bottom", fontsize=size,
                     fontweight="bold")
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    if SHOW_FIGURES:
        # Display the saved PNG rather than the live figure, so the window
        # shows exactly what was saved: full axis labels and the title.
        img = plt.imread(os.path.join(FIG_DIR, f"{name}.png"))
        h, w = img.shape[:2]
        view = plt.figure(figsize=(10, 10 * h / w))
        view.add_axes((0, 0, 1, 1)).imshow(img)
        view.axes[0].axis("off")
        view.canvas.manager.set_window_title(name) \
            if view.canvas.manager is not None else None
        plt.show()


# ----------------------------------------------------------------------------
# 9b. FIGURE: FCM NETWORK DIAGRAM
# ----------------------------------------------------------------------------
# Drawn from EDGES, so any change to the edge list updates the figure.

NETWORK_POS = {                     # (x, y) layout: chain in the middle row
    0: (0.0, 0.0), 1: (2.0, 0.0), 2: (4.0, 0.0),
    3: (6.0, 0.0), 4: (8.0, 0.0), 5: (10.0, 0.0),
    6: (1.0, 1.55), 8: (3.6, 1.55), 7: (6.0, 1.55), 10: (8.0, 1.55),
    9: (7.0, -1.55),
}
LABEL_T = {(8, 4): 0.3, (7, 3): 0.66}            # label position along crowded edges
AIM_OFFSET = {(8, 4): (0.3, 0.2)}   # aim point offset so the edge enters
                                    # its target through the top face


COL_NEW = "#009E73"


def plot_network_diagram(edges=None, pos=None, codes=None, names=None,
                         new_nodes=(), label_t=None, aim_offset=None,
                         curve=None, ylim=(-2.0, 2.0), label_fmt="+.1f",
                         name="fig_fcm_network"):
    """
    Draw an FCM network. Defaults draw the Chapter 6 map from EDGES.

    edges      : list of (source, target, weight)
    new_nodes  : nodes added to the Chapter 6 map, drawn in green
    curve      : {(source, target): rad} for edges routed as arcs
    """
    import textwrap
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    edges = edges or [(e[0], e[1], edge_weight(e)) for e in EDGES]
    pos = pos or NETWORK_POS
    codes, names = codes or CODES, names or NAMES
    label_t = LABEL_T if label_t is None else label_t
    aim_offset = AIM_OFFSET if aim_offset is None else aim_offset
    curve = curve or {}

    bw, bh = 1.42, 0.74
    height = 3.7 * (ylim[1] - ylim[0]) / 4.0
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, height))
    ax.set_xlim(-0.8, 10.8)
    ax.set_ylim(*ylim)
    ax.axis("off")

    handles = [
        Patch(fc="#DDEBF6", ec=COL_CAPABILITY, label="Supply chain node"),
        Patch(fc="0.88", ec="black", lw=1.6, label="Target node"),
        Patch(fc="#FBE6D6", ec=COL_RISK, label="Exogenous driver"),
        Line2D([], [], color="0.15", lw=1.6,
               label="Positive (dependency) edge"),
        Line2D([], [], color=COL_RISK, lw=1.6,
               label="Negative (disruption) edge"),
        Line2D([], [], color="none",
               label="Line width proportional to |weight|"),
    ]
    if new_nodes:
        handles += [Patch(fc="#D5F0E5", ec=COL_NEW, label="Domestic rCF node"),
                    Line2D([], [], color=COL_NEW, lw=1.6,
                           label="rCF supply edge")]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, 0.02), ncol=3, columnspacing=1.4,
              handlelength=1.8)
    fig.tight_layout()            # fix the layout before placing arc labels

    boxes = {}
    for i, (x, y) in pos.items():
        if i == TARGET:
            fc, ec, lw = "0.88", "black", 1.6
        elif i in new_nodes:
            fc, ec, lw = "#D5F0E5", COL_NEW, 1.2
        elif i in RISK_NODES:
            fc, ec, lw = "#FBE6D6", COL_RISK, 0.9
        else:
            fc, ec, lw = "#DDEBF6", COL_CAPABILITY, 0.9
        box = FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh,
                             boxstyle="round,pad=0,rounding_size=0.08",
                             fc=fc, ec=ec, lw=lw, zorder=3)
        ax.add_patch(box)
        boxes[i] = box
        ax.text(x, y + bh / 2 - 0.07, codes[i], ha="center", va="top",
                fontsize=7, fontweight="bold", zorder=4)
        ax.text(x, y - 0.06,
                "\n".join(textwrap.wrap(names[i], 17, break_on_hyphens=False)),
                ha="center", va="center", fontsize=6.2, linespacing=1.1,
                zorder=4)

    to_disp, to_data = ax.transData, ax.transData.inverted()
    for i, j, w in edges:
        if w > 0 and (i in new_nodes or j in new_nodes):
            colour = COL_NEW
        else:
            colour = "0.15" if w > 0 else COL_RISK
        ox, oy = aim_offset.get((i, j), (0.0, 0.0))
        aim = (pos[j][0] + ox, pos[j][1] + oy)
        rad = curve.get((i, j), 0.0)
        ax.add_patch(FancyArrowPatch(
            pos[i], aim, patchA=boxes[i], patchB=boxes[j],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=8, shrinkA=1, shrinkB=1,
            color=colour, lw=0.5 + 1.6 * abs(w), zorder=2))

        (x0, y0), (x1, y1) = pos[i], aim
        if rad:                                        # label at arc apex
            p0, p2 = to_disp.transform(pos[i]), to_disp.transform(aim)
            d = p2 - p0
            c = (p0 + p2) / 2 + rad * np.array([d[1], -d[0]])
            xm, ym = to_data.transform(0.25 * p0 + 0.5 * c + 0.25 * p2)
            kw = {"xy": (xm, ym), "ha": "center"}
        else:
            t = label_t.get((i, j), 0.5)
            xm, ym = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            if y0 == y1:                               # level link
                kw = {"xy": (xm, ym + 0.24), "ha": "center"}
            elif x0 == x1:                             # upright link
                kw = {"xy": (xm + 0.1, ym), "ha": "left"}
            else:                                      # diagonal link
                kw = {"xy": (xm, ym), "ha": "center"}
        ax.annotate(signed(w, label_fmt), xycoords="data", va="center",
                    fontsize=6.5, color=colour, zorder=5,
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white",
                          "ec": "none"}, **kw)

    save_figure(fig, name)


# ----------------------------------------------------------------------------
# CHAPTER 6 ANALYSIS (runs only when this file is executed directly;
# importing it gives the model engine without re-running the Chapter 7 analysis)
# ----------------------------------------------------------------------------

if __name__ == "__main__":

    # ----------------------------------------------------------------------------
    # 5. ANALYSIS
    # ----------------------------------------------------------------------------

    RUN_KEYS = list(SCENARIOS) + [f"shock_{i}" for i in SHOCK_NODES]


    def analyse(W, model):
        """
        Steady state of every scenario and single-node shock.

        Returns dict of key -> deviation from baseline, shape (N,).
        """
        overrides = ([SCENARIOS[k]["clamp"] for k in SCENARIOS]
                     + [shock_override(i, model) for i in SHOCK_NODES])
        states = dict(zip(RUN_KEYS, steady_states(W, model, overrides)))
        base = states["baseline"]
        return {"state": states,
                "delta": {k: v - base for k, v in states.items()}}


    def target_deltas(result):
        """Delta C6 for each shock node, in SHOCK_NODES order."""
        return np.array([result["delta"][f"shock_{i}"][TARGET]
                         for i in SHOCK_NODES])


    def rank_of(values):
        """Rank by magnitude, 1 = largest."""
        order = np.argsort(-np.abs(values), kind="stable")
        ranks = np.empty(len(values), dtype=int)
        ranks[order] = np.arange(1, len(values) + 1)
        return ranks


    def spearman(a, b):
        return np.corrcoef(rank_of(a), rank_of(b))[0, 1]


    # Deterministic results for every model form
    results = {key: analyse(W, m) for key, m in MODELS.items()}
    P = results[PRIMARY]
    PM = MODELS[PRIMARY]

    det_deltas = {key: target_deltas(r) for key, r in results.items()}
    det_rank = rank_of(det_deltas[PRIMARY])
    RANK_ORDER = [SHOCK_NODES[k] for k in np.argsort(det_rank)]   # node indices


    def group_order(deltas, nodes):
        """Nodes of one group, most damaging first."""
        return sorted(nodes, key=lambda i: deltas[SHOCK_NODES.index(i)])


    GROUP_ORDER = {g: group_order(det_deltas[PRIMARY], v["nodes"])
                   for g, v in GROUPS.items()}


    def rating_alternatives_test():
        """
        Re-run the analysis at every combination of the alternative ratings in
        ALT_TERMS. Returns the number of combinations and, for each, the top
        chokepoint, top threat and scenario Delta C6 values.
        """
        idx = [k for k, e in enumerate(EDGES) if (e[0], e[1]) in ALT_TERMS]
        base = [edge_weight(e) for e in EDGES]
        outcomes = []
        for combo in itertools.product([False, True], repeat=len(idx)):
            w = list(base)
            for k, use_alt in zip(idx, combo):
                if use_alt:
                    e = EDGES[k]
                    w[k] = e[2] * WEIGHT_SCALE[ALT_TERMS[(e[0], e[1])]][0]
            r = analyse(build_weights(w), PM)
            d = target_deltas(r)
            outcomes.append({
                "chokepoint": group_order(d, GROUPS["chokepoint"]["nodes"])[0],
                "threat": group_order(d, GROUPS["threat"]["nodes"])[0],
                "scen": {k: r["delta"][k][TARGET] for k in SCEN_KEYS},
            })
        return outcomes


    alt_outcomes = rating_alternatives_test()


    # ----------------------------------------------------------------------------
    # 6. MONTE CARLO WEIGHT UNCERTAINTY
    # ----------------------------------------------------------------------------

    MC_SAMPLES = 2000
    MC_SEED = 42
    MC_KEYS = SCEN_KEYS + [f"shock_{i}" for i in SHOCK_NODES]


    def run_monte_carlo(model, n=MC_SAMPLES, seed=MC_SEED):
        """Delta C6 for every MC key, shape (n, len(MC_KEYS))."""
        rng = np.random.default_rng(seed)
        out = np.zeros((n, len(MC_KEYS)))
        for s in range(n):
            delta = analyse(sample_weights(rng), model)["delta"]
            out[s] = [delta[k][TARGET] for k in MC_KEYS]
        return out


    mc = run_monte_carlo(PM)
    n_scen = len(SCEN_KEYS)
    mc_scen = mc[:, :n_scen]                               # SCEN_KEYS order
    mc_shocks = mc[:, n_scen:]                             # SHOCK_NODES order

    mc_p05 = np.percentile(mc_shocks, 5, axis=0)
    mc_p95 = np.percentile(mc_shocks, 95, axis=0)

    # Within-group rank probabilities, rows in GROUP_ORDER:
    # group_rank_prob[g][row, r] = P(node has rank r + 1 within its group)
    group_rank_prob = {}
    for g, nodes in GROUP_ORDER.items():
        cols = [SHOCK_NODES.index(i) for i in nodes]
        ranks = np.array([rank_of(row) for row in mc_shocks[:, cols]])
        group_rank_prob[g] = np.array(
            [np.bincount(ranks[:, k] - 1, minlength=len(nodes)) / MC_SAMPLES
             for k in range(len(nodes))])


    def p_worse(key_a, key_b):
        """Share of paired MC samples where scenario a reduces C6 more than b."""
        a, b = SCEN_KEYS.index(key_a), SCEN_KEYS.index(key_b)
        return np.mean(mc_scen[:, a] < mc_scen[:, b])


    # ----------------------------------------------------------------------------
    # 6b. MODEL CHECKS AND STRUCTURAL ROBUSTNESS
    # ----------------------------------------------------------------------------

    def run_states(W, model, overrides):
        states = steady_states(W, model, [{}] + list(overrides))
        return states[0], states[1:]

    # Clamp persistence: every clamped node holds its clamp value at steady state
    check_sets = [clamp_set(PM, SCENARIOS[k]["clamp"]) for k in SCEN_KEYS]
    _, check_states = run_states(W, PM, [SCENARIOS[k]["clamp"] for k in SCEN_KEYS])
    clamps_hold = all(np.allclose(st[list(c)], list(c.values()))
                      for st, c in zip(check_states, check_sets))

    # Shutdown transmission: total loss of each supply stage
    shutdown = []
    for i in range(5):
        _, st = run_states(W, PM, [{i: PM["loss"]}])
        shutdown.append((i, st[0][i + 1], st[0][TARGET]))

    # Structural robustness: do the Chapter 7 conclusions survive changes to
    # the steepness and to contested structural choices?
    def structural_case(W_case, model):
        ovs = ([SCENARIOS[k]["clamp"] for k in SCEN_KEYS] +
               [shock_override(i, model) for i in SHOCK_NODES])
        b0, st = run_states(W_case, model, ovs)
        d = st[:, TARGET] - b0[TARGET]
        scen = dict(zip(SCEN_KEYS, d[:len(SCEN_KEYS)]))
        shock = dict(zip(SHOCK_NODES, d[len(SCEN_KEYS):]))
        chain = sorted(range(5), key=lambda i: shock[i])
        risk = sorted(RISK_NODES, key=lambda i: shock[i])
        return {"chokepoints": [CODES[i] for i in chain[:3]],
                "threats": [CODES[i] for i in risk[:2]],
                "scen": scen,
                "upstream_share": shock[2] / shock[4]}

    def with_lambda(lam):
        m = dict(PM)
        m["lam"] = lam
        return m

    def modified(changes):
        W_case = W.copy()
        for (i, j), w in changes.items():
            W_case[i, j] = w
        return W_case

    STRUCTURAL_CASES = [
        ("Reference (lambda = 1)", W, PM),
        ("lambda = 0.5", W, with_lambda(0.5)),
        ("lambda = 2", W, with_lambda(2.0)),
        ("lambda = 3", W, with_lambda(3.0)),
        ("C10 acts on C6 instead of C5", modified({(9, 4): 0.0, (9, 5): -0.5}), PM),
        ("No C9 -> C5 link", modified({(8, 4): 0.0}), PM),
        ("Equal chain weights (all +0.7)",
         modified({(0, 1): 0.7, (1, 2): 0.7, (2, 3): 0.7, (3, 4): 0.7, (4, 5): 0.7}), PM),
        ("C5 -> C6 very strong (+0.9)", modified({(4, 5): 0.9}), PM),
    ]
    structural = [(label, structural_case(Wc, m)) for label, Wc, m in STRUCTURAL_CASES]

    # Independence from initial state: 200 random starts, drivers clamped at
    # normal, each iterated to convergence and compared with the baseline.
    def random_start_test(W, model, n_starts=200, seed=42):
        rng = np.random.default_rng(seed)
        clamps = clamp_set(model)
        idx, val = np.array(list(clamps)), np.array(list(clamps.values()))
        base = steady_states(W, model, [{}])[0]
        worst, max_it = 0.0, 0
        for _ in range(n_starts):
            A = rng.uniform(0.0, 1.0, W.shape[0])
            A[idx] = val
            for t in range(1, MAX_STEPS + 1):
                A_next = fcm_step(A, W, model)
                A_next[idx] = val
                done = np.abs(A_next - A).max() < CONVERGENCE_TOL
                A = A_next
                if done:
                    break
            else:
                raise RuntimeError("Random-start run did not converge.")
            worst, max_it = max(worst, np.abs(A - base).max()), max(max_it, t)
        return n_starts, max_it, worst

    random_starts = random_start_test(W, PM)

    # Geopolitical shock split by route: each route acting alone, with the
    # other C10 link removed (C10 -> C3 precursor route; C10 -> C5 export route).
    def c6_change(W_case, overrides):
        b0, st = run_states(W_case, PM, [overrides])
        return st[0][TARGET] - b0[TARGET]

    geo_routes = {
        "Both routes (reference)": c6_change(W, {9: RISK_MAX}),
        "Precursor route only (C10 -> C3)": c6_change(modified({(9, 4): 0.0}), {9: RISK_MAX}),
        "Export-control route only (C10 -> C5)": c6_change(modified({(9, 2): 0.0}), {9: RISK_MAX}),
    }


    def build_checks():
        L, rule = [], "-" * 104
        L.append("MODEL CHECKS")
        L.append(rule)
        L.append(f"Clamped nodes hold their clamp values in every scenario: {clamps_hold}")
        L.append("Total loss of a supply stage (clamped at 0): activation of the next stage and of C6")
        for i, nxt, c6 in shutdown:
            L.append(f"  {CODES[i]} lost -> {CODES[i + 1]} {nxt:.3f}, C6 {c6:.3f}  (normal = 0.5, total loss = 0)")
        n, it, worst = random_starts
        L.append(f"Random initial states (drivers at normal): {n} of {n} converged within {it} "
                 f"iterations; largest difference from baseline {worst:.1e}")
        L.append("Geopolitical shock (C10 = 1) by route: change in C6")
        for label, d in geo_routes.items():
            L.append(f"  {label:<40}{d:+.3f}")
        L.append("")
        L.append("STRUCTURAL ROBUSTNESS")
        L.append(rule)
        L.append(f"{'Case':<34}{'Top chokepoints':<20}{'Top threats':<14}"
                 f"{'Energy':>8}{'Geopol':>8}{'Compound':>10}{'C3/C5 loss':>12}")
        for label, r in structural:
            L.append(f"{label:<34}{', '.join(r['chokepoints']):<20}"
                     f"{', '.join(r['threats']):<14}"
                     f"{r['scen']['energy']:>+8.3f}{r['scen']['geopol']:>+8.3f}"
                     f"{r['scen']['compound']:>+10.3f}{r['upstream_share']:>12.2f}")
        return "\n".join(L)


    checks = build_checks()
    print(checks)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "fcm_checks.txt"), "w") as f:
        f.write(checks + "\n")


    # ----------------------------------------------------------------------------
    # 7. TEXT OUTPUT
    # ----------------------------------------------------------------------------

    def fmt(x, spec="+.4f"):
        return f"{x:{spec}}"


    def build_summary():
        L = []
        rule = "-" * 104

        plain = PM["label"].replace("$", "").replace("\\lambda", "lambda")
        L.append(f"PRIMARY MODEL: {plain.replace(' (primary)', '')}")
        L.append("0.5 = normal operating conditions; Delta values are "
                 "deviations from the baseline.")
        L.append("")
        L.append("SCENARIO RESULTS (steady-state Delta C6)")
        L.append(rule)
        L.append(f"{'Scenario':<34}{'Nominal':>10}{'MC median':>12}"
                 f"{'MC 5%':>10}{'MC 95%':>10}")
        for col, key in enumerate(SCEN_KEYS):
            d = P["delta"][key][TARGET]
            L.append(f"{SCENARIOS[key]['title']:<34}{fmt(d):>10}"
                     f"{fmt(np.median(mc_scen[:, col])):>12}"
                     f"{fmt(np.percentile(mc_scen[:, col], 5)):>10}"
                     f"{fmt(np.percentile(mc_scen[:, col], 95)):>10}")
        L.append(f"P(geopolitical more severe than energy) = "
                 f"{p_worse('geopol', 'energy'):.1%}")
        L.append(f"P(compound more severe than geopolitical) = "
                 f"{p_worse('compound', 'geopol'):.1%}")

        L.append("")
        for g, spec in GROUPS.items():
            L.append(f"SENSITIVITY OF C6: {spec['title'].upper()}")
            L.append(rule)
            L.append(f"{'Rank':<6}{'Concept':<56}{'Shock':>7}{'Nominal':>10}"
                     f"{'MC 5%':>9}{'MC 95%':>9}{'P(rank 1)':>11}")
            for row, i in enumerate(GROUP_ORDER[g]):
                k = SHOCK_NODES.index(i)
                shock = RISK_MAX if i in RISK_NODES else PM["loss"]
                L.append(f"{row + 1:<6}{NODES[i]:<56}{shock:>7.1f}"
                         f"{fmt(det_deltas[PRIMARY][k]):>10}"
                         f"{fmt(mc_p05[k]):>9}{fmt(mc_p95[k]):>9}"
                         f"{group_rank_prob[g][row, 0]:>11.1%}")
            L.append("")

        n_alt = len(alt_outcomes)
        L.append(f"ALTERNATIVE RATINGS TEST ({n_alt} combinations of "
                 f"{len(ALT_TERMS)} between-term edges)")
        L.append(rule)
        for g in GROUPS:
            counts = {}
            for o in alt_outcomes:
                counts[o[g]] = counts.get(o[g], 0) + 1
            L.append(f"Top {g}: " + ", ".join(
                f"{CODES[i]} in {c}/{n_alt}" for i, c in counts.items()))
        geo_e = sum(o["scen"]["geopol"] < o["scen"]["energy"] for o in alt_outcomes)
        cmp_g = sum(o["scen"]["compound"] < o["scen"]["geopol"]
                    for o in alt_outcomes)
        L.append(f"Geopolitical more severe than energy in {geo_e}/{n_alt}; "
                 f"compound more severe than geopolitical in {cmp_g}/{n_alt}")

        L.append("")
        L.append("UPDATE-RULE ROBUSTNESS (ranking of |Delta C6|)")
        L.append(rule)
        L.append(f"{'Concept':<56}"
                 + "".join(f"{k:>14}" for k in MODELS))
        for i in RANK_ORDER:
            k = SHOCK_NODES.index(i)
            L.append(f"{NODES[i]:<56}"
                     + "".join(f"{rank_of(det_deltas[m])[k]:>14d}"
                               for m in MODELS))
        for m in MODELS:
            if m != PRIMARY:
                L.append(f"Spearman rank correlation, {PRIMARY} vs {m}: "
                         f"{spearman(det_deltas[PRIMARY], det_deltas[m]):.3f}")

        L.append("")
        L.append("MONTE CARLO SETTINGS")
        L.append(rule)
        L.append(f"Samples: {MC_SAMPLES}   Seed: {MC_SEED}   "
                 f"Bands: per linguistic term (see edge table)   "
                 f"Sampling: uniform, sign preserved, |w| in "
                 f"[{MIN_WEIGHT}, 1]")

        L.append("")
        L.append("EDGE TABLE (weight provenance)")
        L.append(rule)
        L.append(f"{'Edge':<10}{'Strength':<14}{'Weight':>8}{'Band':>7}   "
                 f"Evidence")
        for e in EDGES:
            edge = f"{CODES[e[0]]}->{CODES[e[1]]}"
            L.append(f"{edge:<10}{e[3]:<14}{edge_weight(e):>+8.1f}"
                     f"{edge_band(e):>7.1f}   {e[4]}")

        return "\n".join(L)


    summary = build_summary()
    print(summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "fcm_summary.txt"), "w") as f:
        f.write(summary + "\n")

    header = ",".join(SCEN_KEYS + [CODES[i] for i in SHOCK_NODES])
    np.savetxt(os.path.join(RESULTS_DIR, "mc_samples.csv"), mc,
               delimiter=",", header=header, comments="", fmt="%.6f")


    # ----------------------------------------------------------------------------
    # 10. FIGURE: SHOCK PROPAGATION CURVE (one per scenario)
    # ----------------------------------------------------------------------------

    def plot_shock_propagation(key):
        """Deviation from baseline of C1-C6 at each iteration for one scenario."""
        s = SCENARIOS[key]
        dev = simulate(W, PM, s["clamp"]) - simulate(W, PM)
        t = np.arange(TIMESTEPS)

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.0))
        ax.axhline(0, color="0.3", linewidth=0.6, zorder=1)
        for idx in CHAIN_NODES:
            plot_node(ax, t, dev[:, idx], idx)

        ax.set_ylabel(r"Deviation from baseline, $\Delta A_i$")
        format_time_axis(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
                  columnspacing=1.2)
        fig.tight_layout()
        save_figure(fig, f"fig_fcm_propagation_{key}")


    # ----------------------------------------------------------------------------
    # 11. FIGURE: C6 SCENARIO COMPARISON
    # ----------------------------------------------------------------------------

    def plot_target_comparison():
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.9))
        t = np.arange(TIMESTEPS)

        for key, s in SCENARIOS.items():
            y = simulate(W, PM, s["clamp"])[:, TARGET]
            d = P["delta"][key][TARGET]
            label = s["title"]
            if key != "baseline":
                label += f" ($\\Delta A_6$ = {signed(d)})"
            st = SCENARIO_STYLE[key]
            ax.plot(t, y, color=st["color"], linestyle=st["linestyle"],
                    marker=st["marker"], markersize=3.5, markerfacecolor="white",
                    markeredgewidth=0.9, linewidth=1.3, label=label)

        ax.set_ylabel(f"{CODES[TARGET]} activation, $A_6$")
        format_time_axis(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
        fig.tight_layout()
        save_figure(fig, "fig_fcm_c6_comparison")


    # ----------------------------------------------------------------------------
    # 12. FIGURE: SENSITIVITY OF C6 WITH MONTE CARLO INTERVALS
    # ----------------------------------------------------------------------------

    def plot_sensitivity_ranking():
        """Two panels: (a) chokepoints C1-C5, (b) exogenous threats C7-C11."""
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(FIG_WIDTH, 4.6),
                                 gridspec_kw={"hspace": 0.35})
        span = np.abs(det_deltas[PRIMARY]).max()

        for panel, (ax, (g, spec)) in enumerate(zip(axes, GROUPS.items())):
            order = GROUP_ORDER[g][::-1]                  # largest at top
            k_idx = [SHOCK_NODES.index(i) for i in order]
            d = det_deltas[PRIMARY][k_idx]
            lo, hi = mc_p05[k_idx], mc_p95[k_idx]
            colour = COL_RISK if g == "threat" else COL_CAPABILITY
            y = np.arange(len(order))

            ax.barh(y, d, color=colour, height=0.62, edgecolor="none",
                    alpha=0.9)
            ax.axvline(0, color="0.3", linewidth=0.6)
            for yi, di in zip(y, d):
                ax.text(di - 0.02 * span, yi, signed(di), va="center",
                        ha="right", fontsize=7)

            ax.set_yticks(y)
            ax.set_yticklabels([wrap_label(LABELS[i]) for i in order])
            ax.set_title(f"({chr(97 + panel)}) {spec['title']}", loc="left")
            ax.grid(axis="x", color=COL_GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)

        axes[-1].set_xlim(-1.22 * span, 0.04 * span)
        axes[-1].set_xlabel(r"Change in steady-state C6 activation, "
                            r"$\Delta A_6$")
        axes[-1].legend(handles=[
            Patch(color=COL_CAPABILITY,
                  label=f"Supply lost (clamped at {PM['loss']:.0f})"),
            Patch(color=COL_RISK,
                  label=f"Driver maximised (clamped at {RISK_MAX:.0f})"),
        ], loc="upper center", bbox_to_anchor=(0.35, -0.3), ncol=2,
            columnspacing=1.5)

        fig.tight_layout()
        save_figure(fig, "fig_fcm_sensitivity_c6")


    # Presentation version: plain-language labels, positive values, larger
    # text, and only the bars discussed on the slide shown in colour.
    SLIDE_NAMES = {
        0: "Propylene supply",
        1: "AN production",
        2: "PAN precursor production",
        3: "Carbon fibre production",
        4: "Aerospace intermediate material supply",
        6: "Global oil prices",
        7: "Industrial energy costs",
        8: "Qualification and IP constraints",
        9: "Geopolitical tension and export controls",
        10: "Cold-chain transport disruption",
    }
    SLIDE_HIGHLIGHT = {4, 8, 9}
    SLIDE_PANELS = {"chokepoint": "Losing a supply chain stage",
                    "threat": "An external threat at its worst"}

    def plot_sensitivity_slide():
        style = {"font.size": 13, "axes.titlesize": 13.5,
                 "axes.labelsize": 12.5, "xtick.labelsize": 11.5,
                 "ytick.labelsize": 12.5}
        with plt.rc_context(style):
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6.4),
                                     gridspec_kw={"hspace": 0.45})
            top = -det_deltas[PRIMARY].min()

            for ax, g in zip(axes, GROUPS):
                order = GROUP_ORDER[g][::-1]              # largest at top
                k_idx = [SHOCK_NODES.index(i) for i in order]
                vals = -det_deltas[PRIMARY][k_idx]
                main = COL_RISK if g == "threat" else COL_CAPABILITY
                key = [i in SLIDE_HIGHLIGHT for i in order]
                y = np.arange(len(order))

                ax.barh(y, vals, height=0.62, edgecolor="none",
                        color=[main if k else "#CDD1D7" for k in key])
                for yi, v, k in zip(y, vals, key):
                    ax.text(v + 0.004, yi, f"{v:.2f}", va="center",
                            ha="left", fontsize=12,
                            fontweight="bold" if k else "normal",
                            color="black" if k else "0.5")

                ax.set_yticks(y)
                ax.set_yticklabels([SLIDE_NAMES[i] for i in order])
                for label, k in zip(ax.get_yticklabels(), key):
                    label.set_fontweight("bold" if k else "normal")
                    label.set_color("black" if k else "0.45")
                ax.set_title(SLIDE_PANELS[g], loc="left", fontweight="bold")
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", length=0)
                ax.grid(axis="x", color=COL_GRID, linewidth=0.6)
                ax.set_axisbelow(True)

            # Upstream losses fade but do not vanish
            c1 = -det_deltas[PRIMARY][SHOCK_NODES.index(0)]
            axes[0].text(c1 + 0.03, 0, f"still {c1 / top:.0%} of the "
                         "largest effect", va="center", ha="left",
                         fontsize=11.5, style="italic", color="0.35")

            axes[-1].set_xlim(0, top * 1.15)
            axes[-1].set_xlabel("Reduction in Australian aerospace component "
                                "production (relative scale)")
            fig.tight_layout()
            save_figure(fig, "fig_fcm_sensitivity_slide")


    # ----------------------------------------------------------------------------
    # 13. FIGURE: SHOCK PROPAGATION MATRIX
    # ----------------------------------------------------------------------------

    def plot_propagation_matrix():
        """
        Steady-state deviation of each supply-chain concept (columns) under a
        persistent single-node shock (rows). The shocked node's own cell is
        masked, since it is fixed by the clamp rather than by the model.
        """
        order = [i for i in SHOCK_NODES if i in CHAIN_NODES] + RISK_NODES
        data = np.array([P["delta"][f"shock_{i}"][CHAIN_NODES] for i in order])

        mask = np.zeros_like(data, dtype=bool)
        for row, i in enumerate(order):
            if i in CHAIN_NODES:
                mask[row, CHAIN_NODES.index(i)] = True

        limit = np.abs(data[~mask]).max()
        if data[~mask].max() > 1e-9:
            cmap, norm = "RdBu", TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        else:
            cmap, norm = "Reds_r", Normalize(vmin=-limit, vmax=0)

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.6))
        im = ax.imshow(np.ma.masked_array(data, mask), cmap=cmap, norm=norm,
                       aspect="auto")
        ax.set_facecolor("0.88")

        for (row, col), value in np.ndenumerate(data):
            if mask[row, col]:
                text, colour = "clamped", "0.35"
            elif abs(value) < 1e-12:
                text, colour = "0", "0.55"               # no causal path
            elif abs(value) < 5e-4:
                text, colour = "\u22480", "black"
            else:
                text = signed(value)
                colour = "white" if abs(value) > 0.6 * limit else "black"
            ax.text(col, row, text, ha="center", va="center", fontsize=7,
                    color=colour)

        ax.set_xticks(range(len(CHAIN_NODES)))
        ax.set_xticklabels([CODES[i] for i in CHAIN_NODES])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([shock_label(i) for i in order])
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Response concept (upstream to downstream)")
        ax.set_ylabel("Persistent shock")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        n_chain = sum(i in CHAIN_NODES for i in order)
        ax.axhline(n_chain - 0.5, color="black", linewidth=0.8)
        col = CHAIN_NODES.index(TARGET)
        ax.add_patch(plt.Rectangle((col - 0.5, -0.5), 1, len(order),
                                   fill=False, edgecolor="black", linewidth=1.0))

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(r"Steady-state deviation, $\Delta A_i$")
        cbar.outline.set_linewidth(0.5)

        fig.tight_layout()
        save_figure(fig, "fig_fcm_propagation_matrix")


    # ----------------------------------------------------------------------------
    # 14. FIGURE: MODEL-FORM ROBUSTNESS
    # ----------------------------------------------------------------------------

    def plot_model_robustness():
        """
        |Delta C6| for each shock, normalised by the largest effect within each
        update rule so the two are directly comparable. Log axis shows how
        strongly each rule attenuates upstream shocks.
        """
        order = RANK_ORDER[::-1]
        k_idx = [SHOCK_NODES.index(i) for i in order]
        y = np.arange(len(order))

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.4))
        for m, model in MODELS.items():
            v = np.abs(det_deltas[m])
            v = v / v.max()
            st = MODEL_STYLE[m]
            rho = "" if m == PRIMARY else \
                f", $\\rho_s$ = {spearman(det_deltas[PRIMARY], det_deltas[m]):.2f}"
            is_primary = m == PRIMARY
            ax.plot(v[k_idx], y, linestyle="none", marker=st["marker"],
                    color=st["color"], markerfacecolor=st["mfc"],
                    markeredgewidth=0.9, markersize=3.8 if is_primary else 5.5,
                    zorder=5 if is_primary else 3, label=model["label"] + rho)

        ax.set_xscale("log")
        ax.set_xlim(left=max(1e-5, ax.get_xlim()[0]), right=1.6)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[i] for i in order])
        ax.set_xlabel(r"Normalised sensitivity, $|\Delta A_6| \,/\, "
                      r"\max|\Delta A_6|$ (log scale)")
        ax.grid(axis="x", which="major", color=COL_GRID, linewidth=0.5)
        ax.grid(axis="y", color="0.94", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.16), ncol=1)

        fig.tight_layout()
        save_figure(fig, "fig_fcm_model_robustness")


    # ----------------------------------------------------------------------------
    # 15. FIGURE: MONTE CARLO RANK STABILITY
    # ----------------------------------------------------------------------------

    def plot_rank_stability():
        """Within-group rank probabilities: (a) chokepoints, (b) threats."""
        fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.7),
                                 gridspec_kw={"wspace": 0.9})

        for panel, (ax, (g, spec)) in enumerate(zip(axes, GROUPS.items())):
            data = group_rank_prob[g] * 100
            n = len(GROUP_ORDER[g])
            ax.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="auto")
            for (row, col), p in np.ndenumerate(data):
                if p >= 0.5:
                    ax.text(col, row, f"{p:.0f}", ha="center", va="center",
                            fontsize=7, color="white" if p > 60 else "black")
            ax.set_xticks(range(n))
            ax.set_xticklabels(range(1, n + 1))
            ax.set_yticks(range(n))
            ax.set_yticklabels(["\n".join(textwrap.wrap(LABELS[i], 24))
                                for i in GROUP_ORDER[g]], fontsize=7)
            ax.set_xlabel("Rank within group")
            ax.set_title(f"({chr(97 + panel)}) {spec['title']}", loc="left",
                         fontsize=8)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.tight_layout()
        save_figure(fig, "fig_mc_rank_stability")


    # ----------------------------------------------------------------------------
    # 16. FIGURE: MONTE CARLO SCENARIO DISTRIBUTIONS
    # ----------------------------------------------------------------------------

    def plot_mc_scenarios():
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.8))
        bins = np.linspace(mc_scen.min(), mc_scen.max(), 55)
        weights = np.full(MC_SAMPLES, 100 / MC_SAMPLES)

        for col, key in enumerate(SCEN_KEYS):
            st = SCENARIO_STYLE[key]
            x = mc_scen[:, col]
            ax.hist(x, bins=bins, weights=weights, color=st["color"], alpha=0.3,
                    edgecolor="none")
            ax.hist(x, bins=bins, weights=weights, color=st["color"],
                    histtype="step", linewidth=1.0,
                    label=f"{SCENARIOS[key]['title']} (median "
                          f"{signed(np.median(x))}, 90% interval "
                          f"{signed(np.percentile(x, 5))} to "
                          f"{signed(np.percentile(x, 95))})")
            ax.axvline(np.median(x), color=st["color"], linestyle="--",
                       linewidth=0.9)

        ax.set_xlabel(r"Change in steady-state C6 activation, $\Delta A_6$")
        ax.set_ylabel("Share of samples (%)")
        ax.grid(axis="y", color=COL_GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24))
        fig.tight_layout()
        save_figure(fig, "fig_mc_scenarios")


    # ----------------------------------------------------------------------------
    # 17. GENERATE FIGURES
    # ----------------------------------------------------------------------------
    # Results chapter: sensitivity, propagation matrix, MC figures, robustness
    # Scenario section: shock propagation curves, C6 comparison

    plot_network_diagram()
    plot_sensitivity_ranking()
    plot_sensitivity_slide()
    plot_propagation_matrix()
    plot_rank_stability()
    plot_mc_scenarios()
    plot_model_robustness()
    plot_target_comparison()
    plot_shock_propagation("geopol")
    plot_shock_propagation("energy")
    plot_shock_propagation("compound")
