"""
Title:   FCM CF Supply Chain: Domestic rCF Scenarios
Author:  Aeshlein Ralston
Version: 2

Chapter 8 analysis. Adds a domestic recovered carbon fibre node (C12) to the
11-node FCM of Chapter 6 and tests how it changes the response of Australian
aerospace composite manufacturing (C6) to supply disruptions.

The Chapter 6 model is imported unchanged from FCM_CFRP_Supply_Chain.py:
same nodes, weights, update rule, clamping and disruption scenarios.

Representation: supply diversification
--------------------------------------
An FCM has no built-in substitution. Adding C12 as an extra positive input to
C5 only adds a route by which disruption reaches C6 (version 1 of this script
found that this made C6 more vulnerable). Domestic supply is therefore modelled
as diversification of C5's supply, with domestic share sigma:

    C4  -> C5   w * (1 - sigma)    imported fibre
    C10 -> C5   w * (1 - sigma)    export controls act on imports only
    C11 -> C5   w * (1 - sigma)    international cold chain acts on imports only
    C12 -> C5   w(C4 -> C5) * sigma   domestic rCF; total supply dependence kept

Sigma is the share of qualified intermediate material that domestic rCF can
replace, so it carries capability, qualification and volume limits. The
domestic supply is not immune to disruption:

    C6  -> C12  feedstock depends on Australian manufacturing scrap (feedback)
    C8  -> C12  energy costs act on pyrolysis
    C10 -> C12  residual import dependence (resin, equipment); tested only

Benefit measure
---------------
    mitigation = 1 - dA6(with rCF) / dA6(existing network)

0% = no benefit; 100% = the disruption no longer affects C6; negative = worse.

Outputs: figures/fig8_*.pdf|png, results/rcf_summary.txt
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import FCM_CFRP_Supply_Chain as base


# ----------------------------------------------------------------------------
# 1. EXTENDED NETWORK AND SCENARIOS
# ----------------------------------------------------------------------------

RCF = 11                                        # index of C12
NODES = base.NODES + ["C12: Domestic rCF Supply"]
N = len(NODES)
CODES = [name.split(": ")[0] for name in NODES]
NAMES = [name.split(": ")[1] for name in NODES]
LABELS = [f"{c} {n}" for c, n in zip(CODES, NAMES)]

TARGET = base.TARGET
PM = base.MODELS[base.PRIMARY]
C4, C5, C6, C8, C9, C10, C11 = 3, 4, 5, 7, 8, 9, 10
W_TERM = {t: v[0] for t, v in base.WEIGHT_SCALE.items()}

# Section 6.6
RCF_SCENARIOS = {
    "limited": {"title": "Limited domestic rCF", "sigma": 0.10,
                "feedstock": W_TERM["strong"], "energy": -W_TERM["weak"]},
    "developed": {"title": "Developed domestic rCF", "sigma": 0.30,
                  "feedstock": W_TERM["moderate"], "energy": -W_TERM["weak"]},
}
RCF_KEYS = list(RCF_SCENARIOS)

DISRUPTIONS = base.SCEN_KEYS                    # energy, geopol, compound
SHORT = {"energy": "Energy", "geopol": "Geopolitical", "compound": "Compound"}


def rcf_weights(sigma, feedstock, energy, residual=0.0, W11=None):
    """12 x 12 matrix with domestic share sigma of C5's supply."""
    W11 = base.W if W11 is None else W11
    W = np.zeros((N, N))
    W[:base.N, :base.N] = W11
    for i in (C4, C10, C11):                    # import-dependent links into C5
        W[i, C5] = W11[i, C5] * (1.0 - sigma)
    W[RCF, C5] = W11[C4, C5] * sigma
    W[C6, RCF] = feedstock
    W[C8, RCF] = energy
    W[C10, RCF] = residual
    return W


def scenario_weights(key, W11=None, **changes):
    params = {k: v for k, v in RCF_SCENARIOS[key].items() if k != "title"}
    params.update(changes)
    return rcf_weights(W11=W11, **params)


W_EXISTING = rcf_weights(0.0, 0.0, 0.0)          # C12 present, no role
W_SCEN = {k: scenario_weights(k) for k in RCF_KEYS}


# ----------------------------------------------------------------------------
# 2. ANALYSIS FUNCTIONS
# ----------------------------------------------------------------------------

def c6_deltas(W, overrides, model=PM):
    states = base.steady_states(W, model, [{}] + list(overrides))
    return states[1:, TARGET] - states[0, TARGET]


def disruption_deltas(W, model=PM):
    return c6_deltas(W, [base.SCENARIOS[k]["clamp"] for k in DISRUPTIONS],
                     model)


def mitigation(d_rcf, d_existing):
    return 1.0 - np.asarray(d_rcf) / np.asarray(d_existing)


def max_iterations(W, overrides, tol=base.CONVERGENCE_TOL):
    worst = 0
    for ov in [{}] + list(overrides):
        H = base.simulate(W, PM, ov, steps=base.MAX_STEPS)
        change = np.abs(np.diff(H, axis=0)).max(axis=1)
        worst = max(worst, int(np.where(change < tol)[0][0] + 1))
    return worst


# ----------------------------------------------------------------------------
# 3. SCENARIO COMPARISON AND CHOKEPOINT MITIGATION
# ----------------------------------------------------------------------------

d_existing = disruption_deltas(W_EXISTING)
assert np.allclose(d_existing, disruption_deltas(base.W))   # matches Chapter 7

d_scen = {k: disruption_deltas(W_SCEN[k]) for k in RCF_KEYS}
mit_scen = {k: mitigation(d_scen[k], d_existing) for k in RCF_KEYS}

shock_list = [base.shock_override(i, PM) for i in base.SHOCK_NODES]
shock_existing = c6_deltas(W_EXISTING, shock_list)
shock_dev = c6_deltas(W_SCEN["developed"], shock_list)
shock_mit = mitigation(shock_dev, shock_existing)

all_overrides = [base.SCENARIOS[k]["clamp"] for k in DISRUPTIONS] + shock_list
iters = {k: max_iterations(W_SCEN[k], all_overrides) for k in RCF_KEYS}


# ----------------------------------------------------------------------------
# 4. DOMESTIC SHARE SWEEP
# ----------------------------------------------------------------------------
# Mitigation as sigma rises from 0 to 50%, for three strengths of the
# feedstock feedback loop (other links at developed values).

SIGMAS = np.round(np.arange(0.0, 0.501, 0.05), 2)
FEEDBACK_CASES = [("No feedback loop", 0.0),
                  ("Moderate feedback (developed)", W_TERM["moderate"]),
                  ("Strong feedback (limited)", W_TERM["strong"])]

sweep = np.zeros((len(FEEDBACK_CASES), len(SIGMAS), len(DISRUPTIONS)))
for f, (_, fb) in enumerate(FEEDBACK_CASES):
    for s, sig in enumerate(SIGMAS):
        W = scenario_weights("developed", sigma=sig, feedstock=fb)
        sweep[f, s] = mitigation(disruption_deltas(W), d_existing)


def sigma_for(target, f_case=1):
    """Smallest domestic share giving at least `target` mitigation."""
    out = {}
    for d, key in enumerate(DISRUPTIONS):
        fine = np.linspace(0, 0.5, 501)
        vals = np.interp(fine, SIGMAS, sweep[f_case, :, d])
        hit = np.where(vals >= target)[0]
        out[key] = fine[hit[0]] if hit.size else None
    return out


threshold_10 = sigma_for(0.10)


# ----------------------------------------------------------------------------
# 5. DEPENDENCY TESTS (developed scenario)
# ----------------------------------------------------------------------------

DEPENDENCY_TESTS = [
    ("Developed scenario (reference)", {}),
    ("Residual import dependence: C10 \u2192 C12 weak",
     {"residual": -W_TERM["weak"]}),
    ("Residual import dependence: C10 \u2192 C12 moderate",
     {"residual": -W_TERM["moderate"]}),
    ("Feedstock independent of manufacturing: no C6 \u2192 C12",
     {"feedstock": 0.0}),
    ("Feedstock tied to manufacturing: C6 \u2192 C12 very strong",
     {"feedstock": W_TERM["very strong"]}),
    ("Energy exposure moderate: C8 \u2192 C12 moderate",
     {"energy": -W_TERM["moderate"]}),
]
dependency = np.array([
    mitigation(disruption_deltas(scenario_weights("developed", **ch)),
               d_existing) for _, ch in DEPENDENCY_TESTS])


# ----------------------------------------------------------------------------
# 6. STRUCTURAL ROBUSTNESS OF THE rCF RESULT (reported in Chapter 10)
# ----------------------------------------------------------------------------

def with_lambda(lam):
    m = dict(PM)
    m["lam"] = lam
    return m


def modified_W11(changes):
    W11 = base.W.copy()
    for (i, j), w in changes.items():
        W11[i, j] = w
    return W11


STRUCTURAL_CASES = [
    ("Reference (lambda = 1)", base.W, PM),
    ("lambda = 0.5", base.W, with_lambda(0.5)),
    ("lambda = 2", base.W, with_lambda(2.0)),
    ("C10 acts on C6 instead of C5",
     modified_W11({(C10, C5): 0.0, (C10, C6): -0.5}), PM),
    ("No C9 -> C5 link", modified_W11({(C9, C5): 0.0}), PM),
    ("Equal chain weights (all +0.7)",
     modified_W11({(0, 1): 0.7, (1, 2): 0.7, (2, 3): 0.7, (3, 4): 0.7,
                   (4, 5): 0.7}), PM),
]
structural = []
for label, W11, model in STRUCTURAL_CASES:
    d_ex = disruption_deltas(rcf_weights(0.0, 0.0, 0.0, W11=W11), model)
    d_dev = disruption_deltas(scenario_weights("developed", W11=W11), model)
    structural.append((label, mitigation(d_dev, d_ex)))


# ----------------------------------------------------------------------------
# 7. MONTE CARLO WEIGHT UNCERTAINTY (reported in Chapter 10)
# ----------------------------------------------------------------------------
# Chapter 6 edges resampled within their strength bands; the feedstock and energy links
# resampled within their strength bands. Sigma is the scenario definition and
# is held fixed (its effect is shown by the sweep).

MC_SAMPLES = 1000
MC_SEED = 42


def run_monte_carlo(n=MC_SAMPLES, seed=MC_SEED):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, len(RCF_KEYS), len(DISRUPTIONS)))
    band = base.WEIGHT_SCALE["moderate"][1]
    for s in range(n):
        W11 = base.sample_weights(rng)
        d_ex = disruption_deltas(rcf_weights(0.0, 0.0, 0.0, W11=W11))
        for k, key in enumerate(RCF_KEYS):
            sc = RCF_SCENARIOS[key]
            fb = np.clip(sc["feedstock"] + rng.uniform(-band, band),
                         base.MIN_WEIGHT, 1.0)
            en = -np.clip(-sc["energy"] + rng.uniform(-band, band),
                          base.MIN_WEIGHT, 1.0)
            W = rcf_weights(sc["sigma"], fb, en, W11=W11)
            out[s, k] = mitigation(disruption_deltas(W), d_ex)
    return out


mc = run_monte_carlo()
mc_p05, mc_p50, mc_p95 = (np.percentile(mc, q, axis=0) for q in (5, 50, 95))
mc_all_positive = bool((mc > 0).all())


# ----------------------------------------------------------------------------
# 8. TEXT OUTPUT
# ----------------------------------------------------------------------------

def pct(x):
    return f"{100 * x:5.1f}%"


def build_summary():
    L, rule = [], "-" * 100

    L.append("rCF SCENARIOS (Section 6.6)")
    L.append(rule)
    for key, sc in RCF_SCENARIOS.items():
        L.append(f"{sc['title']:<26}domestic share {sc['sigma']:.0%}, "
                 f"C6->C12 {sc['feedstock']:+.1f}, C8->C12 {sc['energy']:+.1f}")
    L.append(f"Convergence: max iterations {iters}; tolerance "
             f"{base.CONVERGENCE_TOL}. Unlinked C12 reproduces Chapter 7: True")

    L.append("")
    L.append("SCENARIO COMPARISON (steady-state Delta C6; share of C6 loss avoided)")
    L.append(rule)
    L.append(f"{'Disruption':<16}{'Existing':>10}" + "".join(
        f"{RCF_SCENARIOS[k]['title']:>32}" for k in RCF_KEYS))
    for d, key in enumerate(DISRUPTIONS):
        row = f"{SHORT[key]:<16}{d_existing[d]:>+10.3f}"
        for k in RCF_KEYS:
            row += f"{d_scen[k][d]:>+22.3f} ({pct(mit_scen[k][d])})"
        L.append(row)

    L.append("")
    L.append("MONTE CARLO: share of C6 loss avoided, median [5th, 95th percentile]")
    L.append(rule)
    for k, key in enumerate(RCF_KEYS):
        cells = "   ".join(f"{SHORT[dk]} {pct(mc_p50[k, d])} "
                           f"[{pct(mc_p05[k, d])}, {pct(mc_p95[k, d])}]"
                           for d, dk in enumerate(DISRUPTIONS))
        L.append(f"{RCF_SCENARIOS[key]['title']:<26}{cells}")
    L.append(f"Samples {MC_SAMPLES}, seed {MC_SEED}. Mitigation positive in "
             f"every sample: {mc_all_positive}")

    L.append("")
    L.append("DOMESTIC SHARE SWEEP (share of C6 loss avoided)")
    L.append(rule)
    for f, (label, _) in enumerate(FEEDBACK_CASES):
        L.append(label)
        L.append(f"{'sigma':>8}" + "".join(f"{SHORT[k]:>14}" for k in DISRUPTIONS))
        for s, sig in enumerate(SIGMAS):
            L.append(f"{sig:>8.0%}" + "".join(f"{pct(v):>14}" for v in sweep[f, s]))
    L.append("Domestic share needed to avoid 10% of the C6 loss "
             "(moderate feedback): " + ", ".join(
                 f"{SHORT[k]} {v:.0%}" if v is not None else f"{SHORT[k]} >50%"
                 for k, v in threshold_10.items()))

    L.append("")
    L.append("CHOKEPOINT MITIGATION (developed scenario, single-node shocks)")
    L.append(rule)
    L.append(f"{'Node':<56}{'Existing':>10}{'Developed':>11}{'Avoided':>10}")
    for k, i in enumerate(base.SHOCK_NODES):
        L.append(f"{NODES[i]:<56}{shock_existing[k]:>+10.3f}"
                 f"{shock_dev[k]:>+11.3f}{pct(shock_mit[k]):>10}")

    L.append("")
    L.append("DEPENDENCY TESTS (developed scenario, share of C6 loss avoided)")
    L.append(rule)
    L.append(f"{'Test':<60}" + "".join(f"{SHORT[k]:>14}" for k in DISRUPTIONS))
    for (label, _), row in zip(DEPENDENCY_TESTS, dependency):
        L.append(f"{label:<60}" + "".join(f"{pct(v):>14}" for v in row))

    L.append("")
    L.append("STRUCTURAL ROBUSTNESS (developed scenario, share of C6 loss avoided)")
    L.append(rule)
    L.append(f"{'Case':<60}" + "".join(f"{SHORT[k]:>14}" for k in DISRUPTIONS))
    for label, row in structural:
        L.append(f"{label:<60}" + "".join(f"{pct(v):>14}" for v in row))

    return "\n".join(L)




# ----------------------------------------------------------------------------
# 9. FIGURES
# ----------------------------------------------------------------------------

SCEN_COLOURS = {"existing": "0.55", "limited": "#56B4E9",
                "developed": base.COL_NEW}
SCEN_TITLES = {"existing": "Existing network",
               **{k: v["title"] for k, v in RCF_SCENARIOS.items()}}


def style_barh(ax):
    ax.axvline(0, color="0.3", linewidth=0.6)
    ax.grid(axis="x", color=base.COL_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_network():
    pos = dict(base.NETWORK_POS)
    pos[RCF] = (10.0, 1.55)
    W = W_SCEN["developed"]
    edges = [(i, j, W[i, j]) for i in range(N) for j in range(N)
             if W[i, j] != 0]
    aim = dict(base.AIM_OFFSET)
    aim[(RCF, C5)] = (0.3, 0.2)
    base.plot_network_diagram(
        edges=edges, pos=pos, codes=CODES, names=NAMES, new_nodes=(RCF,),
        aim_offset=aim, curve={(C8, RCF): -0.42}, ylim=(-2.0, 2.9),
        label_fmt="+.2f", name="fig8_network_rcf")


def plot_scenario_comparison():
    keys = ["existing"] + RCF_KEYS
    values = {"existing": d_existing, **d_scen}
    y = np.arange(len(DISRUPTIONS))[::-1]
    h = 0.26
    fig, ax = plt.subplots(figsize=(base.FIG_WIDTH, 2.9))
    for n, k in enumerate(keys):
        yy = y + (1 - n) * h
        ax.barh(yy, values[k], height=h * 0.9, color=SCEN_COLOURS[k],
                label=SCEN_TITLES[k])
        for d, (yi, v) in enumerate(zip(yy, values[k])):
            text = base.signed(v)
            if k != "existing":
                text += f"  ({100 * mit_scen[k][d]:.0f}% avoided)"
            ax.text(v - 0.004, yi, text, va="center", ha="right", fontsize=6.5)
    ax.set_xlim(d_existing.min() * 1.42, 0.005)
    ax.set_yticks(y)
    ax.set_yticklabels([base.SCENARIOS[k]["title"] for k in DISRUPTIONS])
    ax.set_xlabel(r"Change in steady-state C6 activation, $\Delta A_6$")
    style_barh(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=3)
    fig.tight_layout()
    base.save_figure(fig, "fig8_scenario_comparison")


def plot_share_sweep():
    """Share of C6 loss avoided against domestic share, per disruption."""
    fig, axes = plt.subplots(1, len(DISRUPTIONS), sharey=True,
                             figsize=(base.FIG_WIDTH, 2.8))
    styles = [("0.55", ":"), (base.COL_NEW, "-"), ("#56B4E9", "--")]
    for d, (ax, key) in enumerate(zip(axes, DISRUPTIONS)):
        for f, (label, _) in enumerate(FEEDBACK_CASES):
            c, ls = styles[f]
            ax.plot(SIGMAS * 100, sweep[f, :, d] * 100, color=c, ls=ls,
                    lw=1.4 if f == 1 else 1.0, label=label)
        for k in RCF_KEYS:
            sig = RCF_SCENARIOS[k]["sigma"]
            ax.axvline(sig * 100, color="0.8", lw=0.6, zorder=0)
        ax.axhline(10, color="0.3", lw=0.6, ls="--")
        ax.set_title(SHORT[key], fontsize=8.5)
        ax.set_xlim(0, 50)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.grid(axis="y", color=base.COL_GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("C6 loss avoided (%)")
    axes[0].text(10.8, 1, "limited", fontsize=6.5, color="0.45")
    axes[0].text(30.8, 1, "developed", fontsize=6.5, color="0.45")
    fig.supxlabel("Domestic share of intermediate material supply, \u03c3 (%)",
                  fontsize=9, y=0.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.34,
                        wspace=0.1)
    base.save_figure(fig, "fig8_share_sweep")


def plot_chokepoint_mitigation():
    """
    RQ1 vs RQ2: the Chapter 7 single-node ranking (existing network) beside
    the same disruptions with domestic rCF at the limited (10%) and developed
    (30%) shares. Same layout, colours and scale as fig_fcm_sensitivity_c6.
    """
    from matplotlib.patches import Patch

    shock_lim = c6_deltas(W_SCEN["limited"], shock_list)
    groups = [("(a) Supply chain stages (C1 to C5)", [0, 1, 2, 3, 4],
               base.COL_CAPABILITY),
              ("(b) External pressures (C7 to C11)", base.RISK_NODES,
               base.COL_RISK)]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(base.FIG_WIDTH, 6.2),
                             gridspec_kw={"hspace": 0.32})
    h = 0.27
    lo = shock_existing.min()

    for ax, (title, nodes, colour) in zip(axes, groups):
        idx = [base.SHOCK_NODES.index(i) for i in nodes]
        order = sorted(range(len(nodes)), key=lambda n: shock_existing[idx[n]])
        nodes = [nodes[n] for n in order][::-1]          # largest at top
        idx = [base.SHOCK_NODES.index(i) for i in nodes]
        y = np.arange(len(nodes))

        ax.barh(y + h, shock_existing[idx], height=h * 0.92, color=colour)
        ax.barh(y, shock_lim[idx], height=h * 0.92, color="#A8DCC6")
        ax.barh(y - h, shock_dev[idx], height=h * 0.92,
                color=SCEN_COLOURS["developed"])
        for yi, k in zip(y, idx):
            ax.text(shock_existing[k] - 0.004, yi + h,
                    base.signed(shock_existing[k]), va="center", ha="right",
                    fontsize=6.5)
            ax.text(shock_dev[k] - 0.004, yi - h,
                    f"{base.signed(shock_dev[k])}  "
                    f"({abs(round(100 * shock_mit[k])):.0f}% avoided)",
                    va="center", ha="right", fontsize=6.5)

        ax.set_yticks(y)
        ax.set_yticklabels([base.wrap_label(LABELS[i]) for i in nodes])
        ax.set_title(title, loc="left")
        ax.set_xlim(lo * 1.45, 0.005)
        style_barh(ax)

    axes[-1].set_xlabel(r"Change in steady-state C6 activation, $\Delta A_6$")
    axes[-1].legend(handles=[
        Patch(color=base.COL_CAPABILITY, label="Existing network: supply lost"),
        Patch(color=base.COL_RISK, label="Existing network: driver maximised"),
        Patch(color="#A8DCC6", label="With domestic rCF, 10% share"),
        Patch(color=SCEN_COLOURS["developed"],
              label="With domestic rCF, 30% share"),
    ], loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2, fontsize=7)
    fig.tight_layout()
    base.save_figure(fig, "fig8_chokepoint_mitigation")


def plot_dependency_tests():
    labels = [t for t, _ in DEPENDENCY_TESTS]
    y = np.arange(len(labels))[::-1]
    h = 0.26
    fig, ax = plt.subplots(figsize=(base.FIG_WIDTH, 3.5))
    for d, key in enumerate(DISRUPTIONS):
        ax.barh(y + (1 - d) * h, dependency[:, d] * 100, height=h * 0.9,
                color=base.SCENARIO_STYLE[key]["color"], label=SHORT[key])
    ax.set_yticks(y)
    ax.set_yticklabels([base.wrap_label(t, 34) for t in labels])
    ax.set_xlabel("C6 loss avoided by developed rCF (%)")
    style_barh(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.16), ncol=3,
              fontsize=7)
    fig.tight_layout()
    base.save_figure(fig, "fig8_dependency_tests")


if __name__ == "__main__":
    summary = build_summary()
    print(summary)
    os.makedirs(base.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(base.RESULTS_DIR, "rcf_summary.txt"), "w") as f:
        f.write(summary + "\n")

    plot_network()
    plot_scenario_comparison()
    plot_share_sweep()
    plot_chokepoint_mitigation()
    plot_dependency_tests()
