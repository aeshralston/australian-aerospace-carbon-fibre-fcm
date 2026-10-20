"""
Title:   FCM CF Supply Chain: Weight Sensitivity
Author:  Aeshlein Ralston
Version: 1

Chapter 10 analysis (Section 10.2). Every edge weight in the model is the author's judgement
against the five-term strength scale. This script tests whether the thesis
conclusions depend on those judgements, for three kinds of error:

1. Random error      each weight independently off by up to one band
                     (Monte Carlo)
2. Ambiguous ratings each weight whose evidence sat between two terms,
                     rated at the other term (every combination)
3. Systematic bias   whole classes of weights one term too high or too low
                     (a single assessor's errors are likely to be correlated)
4. Critical weights  each weight shifted one term up and down, one at a time;
                     influence measured on mean exposure across the ten
                     single-node shocks and on the geopolitical rCF benefit

Conclusions tested
------------------
RQ1:       chokepoint order (C5, C4, C3); top two threats (C10, C9);
           scenario severity order (compound >= geopolitical >= energy)
RQ2:       domestic rCF benefit positive under every disruption; larger for
           the geopolitical shock than the energy shock; domestic share needed
           to avoid 10% of the geopolitical loss

The rCF links (feedstock, energy) are held at their Section 6.6 values; their
effect is tested in Section 10.3.4.

Outputs: figures/fig9_*.pdf|png, results/weight_sensitivity.txt
"""

import itertools
import os
import re

import numpy as np
import matplotlib.pyplot as plt

import FCM_CFRP_Supply_Chain as base
import FCM_rCF_Scenarios as rcf


PM = base.MODELS[base.PRIMARY]
TERMS = list(base.WEIGHT_SCALE)
CHAIN = [0, 1, 2, 3, 4]
SCEN = base.SCEN_KEYS                            # energy, geopol, compound
SHORT = rcf.SHORT


# ----------------------------------------------------------------------------
# 1. EVALUATING THE CONCLUSIONS FOR ANY WEIGHT SET
# ----------------------------------------------------------------------------

def edge_terms():
    """Current strength-term index of each edge, in EDGES order."""
    return [TERMS.index(e[3]) for e in base.EDGES]


def weights_from_terms(term_idx):
    ws = [e[2] * base.WEIGHT_SCALE[TERMS[k]][0]
          for e, k in zip(base.EDGES, term_idx)]
    return base.build_weights(ws)


def share_for(W11, target=0.10, key="geopol"):
    """Smallest domestic share avoiding `target` of the loss (developed links)."""
    d = SCEN.index(key)
    ex = rcf.disruption_deltas(rcf.rcf_weights(0.0, 0.0, 0.0, W11=W11))[d]
    sigmas = np.round(np.arange(0.0, 0.501, 0.05), 2)     # as in Chapter 8
    mit = [1 - rcf.disruption_deltas(rcf.scenario_weights(
        "developed", W11=W11, sigma=sig))[d] / ex for sig in sigmas]
    fine = np.linspace(0, 0.5, 501)
    hit = np.where(np.interp(fine, sigmas, mit) >= target)[0]
    return fine[hit[0]] if hit.size else None


def evaluate(W11, with_share=True):
    """Every tested conclusion for one 11-node weight matrix."""
    ovs = ([base.SCENARIOS[k]["clamp"] for k in SCEN] +
           [base.shock_override(i, PM) for i in base.SHOCK_NODES])
    st = base.steady_states(W11, PM, [{}] + ovs)
    d = st[1:, base.TARGET] - st[0, base.TARGET]
    scen = d[:len(SCEN)]
    shock = dict(zip(base.SHOCK_NODES, d[len(SCEN):]))

    chain = [base.CODES[i] for i in sorted(CHAIN, key=lambda i: shock[i])]
    risk = [base.CODES[i] for i in sorted(base.RISK_NODES,
                                          key=lambda i: shock[i])]

    ex = rcf.disruption_deltas(rcf.rcf_weights(0.0, 0.0, 0.0, W11=W11))
    dev = rcf.disruption_deltas(rcf.scenario_weights("developed", W11=W11))
    mit = rcf.mitigation(dev, ex)

    return {
        "scen": scen,
        "mean_shock": float(np.mean(list(shock.values()))),
        "chokepoints": chain[:3],
        "threats": risk[:2],
        "order_ok": scen[2] <= scen[1] + 1e-9 and scen[1] <= scen[0] + 1e-9,
        "chain_ok": chain[:3] == ["C5", "C4", "C3"],
        "threats_ok": set(risk[:2]) == {"C10", "C9"},
        "mit": mit,
        "rcf_positive": bool((mit > 0).all()),
        "geo_gt_energy": mit[1] > mit[0],
        "share10": share_for(W11) if with_share else None,
    }


REF_TERMS = edge_terms()
REF = evaluate(base.W)


# ----------------------------------------------------------------------------
# 2. RANDOM ERROR (Monte Carlo)
# ----------------------------------------------------------------------------

MC_SAMPLES = 2000
MC_SEED = 42

rng = np.random.default_rng(MC_SEED)
mc_rows = [evaluate(base.sample_weights(rng), with_share=False)
           for _ in range(MC_SAMPLES)]
mc_rate = {k: np.mean([r[k] for r in mc_rows])
           for k in ("order_ok", "chain_ok", "threats_ok",
                     "rcf_positive", "geo_gt_energy")}
mc_mit = np.array([r["mit"] for r in mc_rows])


# ----------------------------------------------------------------------------
# 3. AMBIGUOUS RATINGS
# ----------------------------------------------------------------------------
# Edges marked PROVISIONAL had a previous numeric weight midway between two
# terms. Each is tested at both terms, in every combination.

ambiguous = []
for k, e in enumerate(base.EDGES):
    m = re.search(r"previous weight ([0-9.]+)", e[4])
    if m:
        prev = float(m.group(1))
        pair = [t for t in TERMS
                if abs(base.WEIGHT_SCALE[t][0] - prev) < 0.11]
        ambiguous.append((k, [TERMS.index(t) for t in pair]))

amb_rows = []
for combo in itertools.product(*[opts for _, opts in ambiguous]):
    terms = list(REF_TERMS)
    for (k, _), t in zip(ambiguous, combo):
        terms[k] = t
    amb_rows.append(evaluate(weights_from_terms(terms), with_share=False))
amb_rate = {k: np.mean([r[k] for r in amb_rows])
            for k in ("order_ok", "chain_ok", "threats_ok",
                      "rcf_positive", "geo_gt_energy")}
amb_mit = np.array([r["mit"] for r in amb_rows])


# ----------------------------------------------------------------------------
# 4. SYSTEMATIC BIAS
# ----------------------------------------------------------------------------
# Supply chain edges start at C1 to C5; driver edges start at C7 to C11.

def shifted_terms(chain_shift, driver_shift):
    out = []
    for e, k in zip(base.EDGES, REF_TERMS):
        s = chain_shift if e[0] in CHAIN else driver_shift
        out.append(int(np.clip(k + s, 0, len(TERMS) - 1)))
    return out


BIAS_CASES = [
    ("Reference", 0, 0),
    ("Drivers one term lower", 0, -1),
    ("Drivers one term higher", 0, +1),
    ("Chain one term lower", -1, 0),
    ("Chain one term higher", +1, 0),
    ("All one term lower", -1, -1),
    ("All one term higher", +1, +1),
    ("Drivers higher, chain lower", -1, +1),
    ("Drivers lower, chain higher", +1, -1),
]
bias = [(label, evaluate(weights_from_terms(shifted_terms(c, d))))
        for label, c, d in BIAS_CASES]


# ----------------------------------------------------------------------------
# 5. CRITICAL INDIVIDUAL WEIGHTS
# ----------------------------------------------------------------------------

critical = []
for k, e in enumerate(base.EDGES):
    row = {"edge": f"{base.CODES[e[0]]} \u2192 {base.CODES[e[1]]}",
           "term": e[3]}
    for s in (-1, +1):
        t = REF_TERMS[k] + s
        if 0 <= t < len(TERMS):
            terms = list(REF_TERMS)
            terms[k] = t
            row[s] = evaluate(weights_from_terms(terms), with_share=False)
        else:
            row[s] = None                          # already at scale limit
    critical.append(row)


def effect(row, s, metric):
    r = row[s]
    if r is None:
        return 0.0
    if metric == "exposure":
        return r["mean_shock"] - REF["mean_shock"]
    return (r["mit"][1] - REF["mit"][1]) * 100     # geopolitical benefit, pts


for row in critical:
    row["span_c"] = max(abs(effect(row, s, "exposure")) for s in (-1, 1))
    row["span_m"] = max(abs(effect(row, s, "mit")) for s in (-1, 1))
    row["flips"] = [name for name, key in (("chokepoints", "chain_ok"),
                                           ("threats", "threats_ok"),
                                           ("scenario order", "order_ok"),
                                           ("rCF benefit sign", "rcf_positive"))
                    if any(row[s] is not None and not row[s][key]
                           for s in (-1, 1))]


# ----------------------------------------------------------------------------
# 6. TEXT OUTPUT
# ----------------------------------------------------------------------------

def pct(x):
    return f"{100 * x:5.1f}%"


def ok(x):
    return "yes" if x else "NO"


def build_summary():
    L, rule = [], "-" * 110
    names = {"order_ok": "Scenario order (compound >= geopolitical >= energy)",
             "chain_ok": "Chokepoint order C5, C4, C3",
             "threats_ok": "Top two threats C10 and C9",
             "rcf_positive": "rCF benefit positive under every disruption",
             "geo_gt_energy": "rCF benefit larger for geopolitical than energy"}

    L.append("REFERENCE")
    L.append(rule)
    L.append(f"Delta C6: energy {REF['scen'][0]:+.3f}, geopolitical "
             f"{REF['scen'][1]:+.3f}, compound {REF['scen'][2]:+.3f}; "
             f"developed rCF benefit {pct(REF['mit'][0])} / "
             f"{pct(REF['mit'][1])} / {pct(REF['mit'][2])}; share for 10% "
             f"of geopolitical loss {REF['share10']:.0%}")

    L.append("")
    L.append(f"RANDOM ERROR ({MC_SAMPLES} samples) and AMBIGUOUS RATINGS "
             f"({len(amb_rows)} combinations): share of cases in which each "
             f"conclusion holds")
    L.append(rule)
    for k, n in names.items():
        L.append(f"{n:<56}{pct(mc_rate[k]):>10}{pct(amb_rate[k]):>12}")
    L.append("Developed rCF benefit, 5th to 95th percentile (random error): "
             + ", ".join(f"{SHORT[s]} {pct(np.percentile(mc_mit[:, d], 5))}"
                         f" to {pct(np.percentile(mc_mit[:, d], 95))}"
                         for d, s in enumerate(SCEN)))
    L.append("Developed rCF benefit, range (ambiguous ratings): "
             + ", ".join(f"{SHORT[s]} {pct(amb_mit[:, d].min())} to "
                         f"{pct(amb_mit[:, d].max())}"
                         for d, s in enumerate(SCEN)))

    L.append("")
    L.append("SYSTEMATIC BIAS")
    L.append(rule)
    L.append(f"{'Case':<30}{'Energy':>8}{'Geopol':>8}{'Compound':>10}"
             f"{'Chokepoints':>14}{'Threats':>10}{'Order':>7}"
             f"{'rCF E':>8}{'rCF G':>8}{'rCF C':>8}{'Share':>7}")
    for label, r in bias:
        share = f"{r['share10']:.0%}" if r["share10"] is not None else ">50%"
        L.append(f"{label:<30}{r['scen'][0]:>+8.3f}{r['scen'][1]:>+8.3f}"
                 f"{r['scen'][2]:>+10.3f}{', '.join(r['chokepoints']):>14}"
                 f"{', '.join(r['threats']):>10}{ok(r['order_ok']):>7}"
                 f"{pct(r['mit'][0]):>8}{pct(r['mit'][1]):>8}"
                 f"{pct(r['mit'][2]):>8}{share:>7}")

    L.append("")
    L.append("CRITICAL INDIVIDUAL WEIGHTS (one term down / up)")
    L.append(rule)
    L.append("Exposure = mean Delta C6 across the ten single-node shocks "
             f"(reference {REF['mean_shock']:+.3f})")
    L.append(f"{'Edge':<12}{'Term':<13}{'Exposure (down, up)':>26}"
             f"{'Geopol rCF benefit, pts':>26}   Conclusions changed")
    for row in sorted(critical, key=lambda r: -r["span_c"]):
        dc = [effect(row, s, "exposure") if row[s] else None for s in (-1, 1)]
        dm = [effect(row, s, "mit") if row[s] else None for s in (-1, 1)]
        fc = ", ".join("limit" if v is None else f"{v:+.3f}" for v in dc)
        fm = ", ".join("limit" if v is None else f"{v:+.1f}" for v in dm)
        L.append(f"{row['edge']:<12}{row['term']:<13}{fc:>26}{fm:>26}   "
                 f"{', '.join(row['flips']) or 'none'}")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# 7. FIGURES
# ----------------------------------------------------------------------------

def plot_bias():
    """Scenario severity and rCF benefit under each bias case."""
    labels = [b[0] for b in bias]
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(base.FIG_WIDTH, 3.6),
                             gridspec_kw={"wspace": 0.08})
    for d, key in enumerate(SCEN):
        st = base.SCENARIO_STYLE[key]
        kw = dict(color=st["color"], marker=st["marker"], ls="none",
                  markersize=5, markerfacecolor="white", markeredgewidth=1.1,
                  label=SHORT[key])
        axes[0].plot([r["scen"][d] for _, r in bias], y, **kw)
        axes[1].plot([100 * r["mit"][d] for _, r in bias], y, **kw)
    for ax in axes:
        ax.axhline(y[0] - 0.5, color="0.7", lw=0.6)
        ax.grid(axis="x", color=base.COL_GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel(r"(a) $\Delta A_6$ under each disruption")
    axes[1].set_xlabel("(b) C6 loss avoided by developed rCF (%)")
    axes[1].set_xlim(left=0)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.9, -0.18), ncol=3,
                   fontsize=7)
    fig.tight_layout()
    base.save_figure(fig, "fig9_systematic_bias")


def plot_critical():
    """Tornado charts of single-weight influence."""
    order = sorted(critical, key=lambda r: r["span_c"])
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(base.FIG_WIDTH, 3.8),
                             gridspec_kw={"wspace": 0.08})
    metrics = [("exposure", r"(a) Change in mean single-shock $\Delta A_6$"),
               ("mit", "(b) Change in geopolitical rCF benefit (pts)")]
    for ax, (metric, xlabel) in zip(axes, metrics):
        lo = [effect(r, -1, metric) for r in order]
        hi = [effect(r, +1, metric) for r in order]
        ax.barh(y, lo, height=0.6, color="#56B4E9", label="One term lower")
        ax.barh(y, hi, height=0.6, color="#0072B2", label="One term higher")
        ax.axvline(0, color="0.3", lw=0.6)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=base.COL_GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{r['edge']} ({r['term']})" for r in order])
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.17), ncol=2,
                   fontsize=7)
    fig.tight_layout()
    base.save_figure(fig, "fig9_critical_weights")


if __name__ == "__main__":
    summary = build_summary()
    print(summary)
    os.makedirs(base.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(base.RESULTS_DIR, "weight_sensitivity.txt"),
              "w") as f:
        f.write(summary + "\n")
    plot_bias()
    plot_critical()
