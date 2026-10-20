"""
Title:   FCM CF Supply Chain: Sensitivity to Key Assumptions
Author:  Aeshlein Ralston
Version: 1

Chapter 10 analysis (Section 10.3 and the summary in Section 10.4).
Each key modelling assumption is changed one at a time and the full
analysis is re-run. The conclusions checked are those discussed in Chapter 9:

RQ1  chokepoint order across the five supply stages; top two threats
     (C10, C9); scenario order (compound >= geopolitical >= energy)
RQ2  domestic rCF benefit positive under every disruption; larger for the
     geopolitical than the energy shock; domestic share needed to avoid 10%
     of the geopolitical loss

Assumptions changed
-------------------
1. Update rule      modified Kosko rule; steepness lambda = 0.5, 2, 3
2. Shock severity   half (0.25 / 0.75) and mild (0.4 / 0.6) clamps
3. Contested links  C10 acts on C6 instead of C5; no C9 -> C5 link;
                    equal chain weights (+0.7); C5 -> C6 very strong (+0.9)
4. Single weights   each weight changed by +/-10% of its value, one at a time

The weight tests (random error, ambiguous ratings, systematic bias, one-term
shifts) are run by FCM_Weight_Sensitivity.py and imported here only for the
summary figure.

Outputs: figures/fig10_*.pdf|png, results/assumption_sensitivity.txt
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import FCM_CFRP_Supply_Chain as base
import FCM_rCF_Scenarios as rcf
import FCM_Weight_Sensitivity as ws


PM = base.MODELS[base.PRIMARY]
SCEN = base.SCEN_KEYS                            # energy, geopol, compound
CHAIN = [0, 1, 2, 3, 4]
GEO = SCEN.index("geopol")


# ----------------------------------------------------------------------------
# 1. EVALUATION AT ANY MODEL FORM, STRUCTURE AND SEVERITY
# ----------------------------------------------------------------------------

def model_with(**changes):
    m = dict(PM)
    m.update(changes)
    return m


def clamps(severity, model):
    """Clamp values for a lost stage and a maximised driver."""
    lo = model["normal"] - (model["normal"] - model["loss"]) * severity
    hi = model["normal"] + (base.RISK_MAX - model["normal"]) * severity
    return lo, hi


def overrides(severity, model):
    lo, hi = clamps(severity, model)
    scen = [{i: hi for i in base.SCENARIOS[k]["clamp"]} for k in SCEN]
    shocks = [{i: hi if i in base.RISK_NODES else lo}
              for i in base.SHOCK_NODES]
    return scen, shocks


def c6_changes(W, model, ovs):
    st = base.steady_states(W, model, [{}] + list(ovs))
    return st[1:, base.TARGET] - st[0, base.TARGET]


def evaluate(W11=None, model=PM, severity=1.0, with_share=True):
    W11 = base.W if W11 is None else W11
    scen_ov, shock_ov = overrides(severity, model)
    d = c6_changes(W11, model, scen_ov + shock_ov)
    scen = d[:len(SCEN)]
    shock = dict(zip(base.SHOCK_NODES, d[len(SCEN):]))
    chain = [base.CODES[i] for i in sorted(CHAIN, key=lambda i: shock[i])]
    risk = [base.CODES[i] for i in sorted(base.RISK_NODES,
                                          key=lambda i: shock[i])]

    ex = c6_changes(rcf.rcf_weights(0.0, 0.0, 0.0, W11=W11), model, scen_ov)
    dev = c6_changes(rcf.scenario_weights("developed", W11=W11), model,
                     scen_ov)
    mit = rcf.mitigation(dev, ex)

    share = None
    if with_share:
        sig = np.round(np.arange(0.0, 0.501, 0.05), 2)
        m = [1 - c6_changes(rcf.scenario_weights("developed", W11=W11,
                                                 sigma=s), model,
                            [scen_ov[GEO]])[0] / ex[GEO] for s in sig]
        fine = np.linspace(0, 0.5, 501)
        hit = np.where(np.interp(fine, sig, m) >= 0.10)[0]
        share = fine[hit[0]] if hit.size else None

    return {
        "scen": scen, "shock": shock, "chain": chain, "risk": risk,
        "chain_ok": chain == ["C5", "C4", "C3", "C2", "C1"],
        "threats_ok": set(risk[:2]) == {"C10", "C9"},
        "order_ok": scen[2] <= scen[1] + 1e-9 and scen[1] <= scen[0] + 1e-9,
        "mit": mit,
        "rcf_positive": bool((mit > 1e-6).all()),
        "geo_gt_energy": mit[1] > mit[0],
        "share10": share,
        "upstream": shock[0] / shock[4],          # C1 relative to C5
    }


def modified(changes):
    W11 = base.W.copy()
    for (i, j), w in changes.items():
        W11[i, j] = w
    return W11


# ----------------------------------------------------------------------------
# 2. ASSUMPTION CASES
# ----------------------------------------------------------------------------

CASES = [
    ("Reference", "", {}),
    ("Update rule", "Modified Kosko rule", {"model": base.MODELS["kosko"]}),
    ("Update rule", "λ = 0.5", {"model": model_with(lam=0.5)}),
    ("Update rule", "λ = 2", {"model": model_with(lam=2.0)}),
    ("Update rule", "λ = 3", {"model": model_with(lam=3.0)}),
    ("Shock severity", "Half (0.25 or 0.75)", {"severity": 0.5}),
    ("Shock severity", "Mild (0.4 or 0.6)", {"severity": 0.2}),
    ("Contested links", "C10 acts on C6 instead of C5",
     {"W11": modified({(9, 4): 0.0, (9, 5): -0.5})}),
    ("Contested links", "No C9 → C5 link",
     {"W11": modified({(8, 4): 0.0})}),
    ("Contested links", "Equal chain weights (+0.7)",
     {"W11": modified({(0, 1): .7, (1, 2): .7, (2, 3): .7, (3, 4): .7,
                       (4, 5): .7})}),
    ("Contested links", "C5 → C6 very strong (+0.9)",
     {"W11": modified({(4, 5): 0.9})}),
]
results = [(g, lab, evaluate(**kw)) for g, lab, kw in CASES]
REF = results[0][2]


# Kosko rank agreement with the primary rule (all ten nodes)
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


kosko = results[1][2]
rho_kosko = spearman([REF["shock"][i] for i in base.SHOCK_NODES],
                     [kosko["shock"][i] for i in base.SHOCK_NODES])


# ----------------------------------------------------------------------------
# 3. EACH WEIGHT +/- 10%, ONE AT A TIME
# ----------------------------------------------------------------------------

pm10 = []
for e in base.EDGES:
    i, j = e[0], e[1]
    row = {"edge": f"{base.CODES[i]} → {base.CODES[j]}"}
    for f in (0.9, 1.1):
        W11 = base.W.copy()
        W11[i, j] = np.sign(W11[i, j]) * min(abs(W11[i, j]) * f, 1.0)
        row[f] = evaluate(W11)
    pm10.append(row)
pm10_all = [r[f] for r in pm10 for f in (0.9, 1.1)]
pm10_held = {k: all(r[k] for r in pm10_all)
             for k in ("chain_ok", "threats_ok", "order_ok", "rcf_positive",
                       "geo_gt_energy")}


# ----------------------------------------------------------------------------
# 4. SUMMARY RANGES FOR THE TORNADO FIGURE
# ----------------------------------------------------------------------------
# Outcomes: (a) change in C6 under the geopolitical shock; (b) share of the
# geopolitical loss avoided by the developed rCF scenario.

def span(vals):
    vals = np.asarray(vals, dtype=float)
    return float(vals.min()), float(vals.max())


def case(label):
    return next(r for g, lab, r in results if lab == label)


crit_rows = [r[s] for r in ws.critical for s in (-1, +1) if r.get(s)]
TORNADO = [
    ("Steepness λ (0.5 to 3)",
     span([case(l)["scen"][GEO] for l in ("λ = 0.5", "λ = 2",
                                         "λ = 3")] + [REF["scen"][GEO]]),
     span([case(l)["mit"][GEO] for l in ("λ = 0.5", "λ = 2",
                                        "λ = 3")] + [REF["mit"][GEO]])),
    ("Shock severity (half, mild)", None,
     span([case(l)["mit"][GEO] for l in ("Half (0.25 or 0.75)",
                                        "Mild (0.4 or 0.6)")]
          + [REF["mit"][GEO]])),
    ("C10 acts on C6 instead of C5",
     span([case("C10 acts on C6 instead of C5")["scen"][GEO],
           REF["scen"][GEO]]),
     span([case("C10 acts on C6 instead of C5")["mit"][GEO],
           REF["mit"][GEO]])),
    ("Systematic bias (8 cases)",
     span([r["scen"][GEO] for _, r in ws.bias]),
     span([r["mit"][GEO] for _, r in ws.bias])),
    ("Each weight one term up or down",
     span([r["scen"][GEO] for r in crit_rows] + [REF["scen"][GEO]]),
     span([r["mit"][GEO] for r in crit_rows] + [REF["mit"][GEO]])),
    ("Random error (90% range)",
     tuple(np.percentile([r["scen"][GEO] for r in ws.mc_rows], [5, 95])),
     tuple(np.percentile(ws.mc_mit[:, GEO], [5, 95]))),
    ("Ambiguous ratings (32 cases)",
     span([r["scen"][GEO] for r in ws.amb_rows]),
     span(ws.amb_mit[:, GEO])),
    ("C5 → C6 very strong",
     span([case("C5 → C6 very strong (+0.9)")["scen"][GEO],
           REF["scen"][GEO]]),
     span([case("C5 → C6 very strong (+0.9)")["mit"][GEO],
           REF["mit"][GEO]])),
    ("Each weight ±10%",
     span([r["scen"][GEO] for r in pm10_all] + [REF["scen"][GEO]]),
     span([r["mit"][GEO] for r in pm10_all] + [REF["mit"][GEO]])),
    ("Equal chain weights",
     span([case("Equal chain weights (+0.7)")["scen"][GEO],
           REF["scen"][GEO]]),
     span([case("Equal chain weights (+0.7)")["mit"][GEO],
           REF["mit"][GEO]])),
    ("No C9 → C5 link",
     span([case("No C9 → C5 link")["scen"][GEO], REF["scen"][GEO]]),
     span([case("No C9 → C5 link")["mit"][GEO], REF["mit"][GEO]])),
]


# ----------------------------------------------------------------------------
# 5. OUTPUT
# ----------------------------------------------------------------------------

def yn(b):
    return "held" if b else "CHANGED"


def build_summary():
    L = ["SENSITIVITY TO KEY ASSUMPTIONS", "=" * 30, ""]
    L.append("Case | chain order | top threats | scenario order | "
             "dE dG dC | rCF benefit E/G/C | share10 | C1/C5")
    for g, lab, r in results:
        share = "none" if r["share10"] is None else f"{100 * r['share10']:.0f}%"
        L.append(
            f"{g:15s} {lab:32s} {yn(r['chain_ok']):7s} "
            f"{','.join(r['risk'][:2]):8s} {yn(r['order_ok']):7s} "
            + " ".join(f"{x:+.3f}" for x in r["scen"]) + "  "
            + " ".join(f"{100 * x:5.1f}%" for x in r["mit"])
            + f"  {share}"
            + f"  {r['upstream']:.2f}"
            + f"  chain={'>'.join(r['chain'])} risk={'>'.join(r['risk'])}")
    L += ["", f"Spearman rho, Kosko vs rescaled (10 nodes): {rho_kosko:.2f}",
          "", "EACH WEIGHT +/-10%"]
    L.append("All conclusions held: " + ", ".join(
        f"{k}={v}" for k, v in pm10_held.items()))
    for r in pm10:
        a, b = r[0.9], r[1.1]
        L.append(f"  {r['edge']:10s} mean shock {np.mean(list(a['shock'].values())):+.4f}"
                 f"/{np.mean(list(b['shock'].values())):+.4f}  "
                 f"geo {a['scen'][GEO]:+.3f}/{b['scen'][GEO]:+.3f}  "
                 f"geo benefit {100*a['mit'][GEO]:.1f}/{100*b['mit'][GEO]:.1f}%  "
                 f"share10 {a['share10']}/{b['share10']}  "
                 f"top threats {','.join(a['risk'][:2])}/{','.join(b['risk'][:2])}")
    L += ["", "TORNADO RANGES (geo dC6 ; geo benefit %)"]
    for lab, a, b in TORNADO:
        sa = "n/a" if a is None else f"{a[0]:+.3f} to {a[1]:+.3f}"
        L.append(f"  {lab:34s} {sa:22s} {100*b[0]:5.1f} to {100*b[1]:5.1f}%")
    return "\n".join(L)


WEIGHT_ROWS = {"Systematic bias (8 cases)", "Each weight one term up or down",
               "Random error (90% range)", "Ambiguous ratings (32 cases)",
               "Each weight \u00b110%"}


def plot_tornado():
    from matplotlib.patches import Patch
    rows = sorted(TORNADO, key=lambda t: t[2][1] - t[2][0])
    fig, axes = plt.subplots(1, 2, figsize=(base.FIG_WIDTH, 4.5),
                             sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(rows))
    ref = (REF["scen"][GEO], 100 * REF["mit"][GEO])
    col = {True: "#9ecae1", False: "#08519c"}
    for p, ax in enumerate(axes):
        for yi, (lab, a, b) in zip(y, rows):
            rng = a if p == 0 else (100 * b[0], 100 * b[1])
            if rng is None:
                ax.text(-0.48, yi, "not shown: the shock itself is smaller",
                        va="center", ha="left", fontsize=6.3, color="0.4",
                        style="italic")
                continue
            lo, hi = rng
            width = max(hi - lo, 0.004 if p == 0 else 0.25)
            ax.barh(yi, width, left=lo if hi - lo > 0 else lo - width / 2,
                    height=0.6, color=col[lab in WEIGHT_ROWS],
                    edgecolor="none")
        ax.axvline(ref[p], color="0.2", linewidth=0.8, linestyle="--")
        ax.grid(axis="x", color=base.COL_GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
    axes[0].set_xlim(-0.5, 0.0)
    axes[1].set_xlim(0, 31)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=7.5)
    axes[0].set_title("(a) Change in C6 under the\ngeopolitical shock",
                      loc="left", fontsize=8.5)
    axes[1].set_title("(b) Share of the geopolitical loss\navoided by the "
                      "developed rCF scenario", loc="left", fontsize=8.5)
    axes[0].set_xlabel(r"$\Delta A_6$ (reference $-$0.197, dashed)",
                       fontsize=7.5)
    axes[1].set_xlabel("Loss avoided, % (reference 15.0%, dashed)",
                       fontsize=7.5)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7)
    axes[1].legend(handles=[Patch(color=col[False], label="Changed assumption"),
                            Patch(color=col[True], label="Weight uncertainty")],
                   loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    base.save_figure(fig, "fig10_tornado")


if __name__ == "__main__":
    os.makedirs(base.RESULTS_DIR, exist_ok=True)
    text = build_summary()
    with open(os.path.join(base.RESULTS_DIR, "assumption_sensitivity.txt"),
              "w") as f:
        f.write(text + "\n")
    print(text)
    plot_tornado()
