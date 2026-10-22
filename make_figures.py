#!/usr/bin/env python3
"""
Histogram / summary figures from run_irb_sim.py outputs.
Needs matplotlib:  uv run --no-project --with matplotlib --with numpy python make_figures.py [--data DIR]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# reference palette (light mode): categorical slots 1-3 validate all-pairs
C_H, C_P00, C_MIN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EST_COLOR = {"P00": C_P00, "H": C_H, "min": C_MIN}
EST_LABEL = {"P00": "joint P00", "H": "H = (q1+q2)/2", "min": "min(q1,q2) pooled"}
MODE_LABEL = {"depol": "depolarizing", "coh_aligned": "coherent ZZ, aligned", "coh_opposing": "coherent ZZ, opposing",
              "mixed_aligned": "mixed, aligned", "mixed_opposing": "mixed, opposing"}
MODES = list(MODE_LABEL)
PAIRS = [(0.99, 0.99), (0.95, 0.99), (0.95, 0.95), (0.99, 0.95)]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5,
})


class Store:
    def __init__(self, d):
        self.d = d
        self.meta = {}
        for f in sorted(os.listdir(d)):
            if f.startswith("scen_") and f.endswith(".json"):
                m = json.load(open(os.path.join(d, f)))
                self.meta[(m["F_R"], m["F_G"], m["mode"], m["K"])] = m

    def get(self, FR, FG, mode, K):
        m = self.meta.get((FR, FG, mode, K))
        if m is None:
            return None, None
        return m, np.load(os.path.join(self.d, f"scen_{m['id']:02d}.npz"))


OUT_OF_RANGE = {"checked": 0, "outside": 0}


def check_cover(vals, bins):
    """Record whether every finite sample falls inside the histogram bins (no silent tail clipping)."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    OUT_OF_RANGE["checked"] += v.size
    OUT_OF_RANGE["outside"] += int(np.sum((v < bins[0]) | (v > bins[-1])))


def full_range(*arrs):
    v = np.concatenate([np.atleast_1d(np.asarray(a, float)) for a in arrs])
    v = v[np.isfinite(v)]
    return float(v.min()), float(v.max())


def grid_bins(lo, hi, g, target=60):
    """Bins whose width is a multiple of the estimator's lattice spacing g (avoids aliasing)."""
    w = max(1, round((hi - lo) / target / g)) * g
    start = np.floor(lo / g) * g - g / 2
    return np.arange(start, hi + w, w)


def ckey(c):
    return f"a{c['a']:.12f}_th{c['theta']:+.12f}"


def save(fig, name, out):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------- fig 1
def fig_single_circuit(st, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, mode in zip(axes, ["depol", "coh_aligned", "mixed_aligned"]):
        m, _ = st.get(0.99, 0.99, mode, 50)
        if m is None:
            ax.set_visible(False); continue
        c = m["ref"]
        b = np.load(os.path.join(st.d, f"bank_t1_ref_{ckey(c)}.npz"))
        md = int(b["m"])
        hc, hs = b["H_circ"].astype(float), b["H_shot"].astype(float)
        theory = (1 + c["p"] ** md) / 2
        lo, hi = full_range(hc, hs, [theory])          # full finite sample range
        pad = 0.03 * (hi - lo + 1e-3)
        bins = grid_bins(lo - pad, hi + pad, 1 / (2 * 500), 70)
        check_cover(hs, bins); check_cover(hc, bins)
        ax.hist(hs, bins=bins, density=True, color=C_P00, alpha=0.35)
        if hc.std() < 1e-6:
            ax.axvline(hc.mean(), color=C_H, lw=2.2)
            note = "exact E[H|C] identical for all circuits"
        else:
            ax.hist(hc, bins=bins, density=True, histtype="step", lw=2, color=C_H)
            note = f"circuit SD of E[H|C] = {hc.std():.4f}"
        ax.axvline(theory, color=INK, lw=1.2, ls="--")
        ax.set_title(f"{MODE_LABEL[mode].split(',')[0]} (reference, F_R = 0.99)\n"
                     f"m = {md}, lambda*m = {c['lam']*md:.2f}; {note}\n"
                     f"shot-H SD = {hs.std():.4f}", fontsize=9)
        ax.set_xlabel("single-circuit H")
        ax.set_ylabel("density")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_P00, alpha=0.35, label="per-circuit H estimate, S = 500 shots"),
               plt.Line2D([], [], color=C_H, lw=2, label="per-circuit exact (infinite-shot) E[H|C]"),
               plt.Line2D([], [], color=INK, lw=1.2, ls="--", label="ensemble mean (1+p^m)/2")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Single-circuit H at m ~ 1/lambda, 20000 independent circuits "
                 "(min(q1,q2) is only formed after pooling, so it is not shown per circuit)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig1_single_circuit_H_ref_F99.png", out)


# ---------------------------------------------------------------- fig 2
def fig_pooled(st, out, F):
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 8.2))
    m = None
    for i, mode in enumerate(["depol", "coh_aligned", "mixed_aligned"]):
        data = {}
        for K in (10, 50):
            m, z = st.get(F, F, mode, K)
            if m is None:
                continue
            j = m["theory_checks"]["int"]["t1_index"]
            data[K] = (m, z, j)
            mref = m
        if not data:
            continue
        allv = np.concatenate([np.concatenate([z["int_H_Y"][:, j], z["int_min_Y"][:, j]]) for m, z, j in data.values()])
        lines = [d[0]["oracle"]["int"][d[2]]["theory_marg"] for d in data.values()] + \
                [d[0]["oracle"]["int"][d[2]]["mean_min"][str(K)] for K, d in data.items()]
        lo, hi = full_range(allv, lines)                 # full finite sample range
        pad = 0.02 * (hi - lo)
        bins = grid_bins(lo - pad, hi + pad, 1 / (2 * 500 * 10), 70)
        for k_i, K in enumerate((10, 50)):
            ax = axes[i, k_i]
            if K not in data:
                ax.set_visible(False); continue
            m, z, j = data[K]
            md = m["depths"]["int"][j]
            orc = m["oracle"]["int"][j]
            check_cover(z["int_H_Y"][:, j], bins); check_cover(z["int_min_Y"][:, j], bins)
            ax.set_xlim(bins[0], bins[-1])
            ax.hist(z["int_H_Y"][:, j], bins=bins, color=C_H, alpha=0.35, label="pooled H")
            ax.hist(z["int_min_Y"][:, j], bins=bins, histtype="step", lw=1.8, color=C_MIN, label="pooled min(q1,q2)")
            ax.axvline(orc["theory_marg"], color=C_H, lw=1.6, ls="--", label=f"theory E[H] = {orc['theory_marg']:.4f}")
            ax.axvline(orc["mean_min"][str(K)], color=C_MIN, lw=1.6, ls=":",
                       label=f"E[min] (oracle MC) = {orc['mean_min'][str(K)]:.4f}")
            ax.set_title(f"{MODE_LABEL[mode]}, K={K}  (interleaved, m={md}, lambda*m={m['int']['lam']*md:.2f})")
            ax.set_xlabel("pooled estimate (K x 500 shots)")
            ax.set_ylabel("runs")
            ax.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor=SURF, edgecolor=GRID, framealpha=0.9)
    if m is None:
        plt.close(fig); return
    fig.suptitle(f"Pooled H and min across {mref['R']} repeated circuit sets, F_R = F_G = {F}", y=1.01, fontsize=11)
    fig.tight_layout()
    save(fig, f"fig2_pooled_H_min_int_aligned_F{int(F*100)}.png", out)


# ---------------------------------------------------------------- fig 3
def fig_r_hist(st, out, FR, FG, ests, tag):
    fig, axes = plt.subplots(5, 2, figsize=(10.5, 13))
    for i, mode in enumerate(MODES):
        rows = {K: st.get(FR, FG, mode, K) for K in (10, 50)}
        m0 = next((mm for mm, _ in rows.values() if mm is not None), None)
        if m0 is None:
            continue
        rt, rn = m0["r_true"] * 100, m0["r_nominal"] * 100
        vals = [z[f"r_{e}"][np.isfinite(z[f"r_{e}"])] * 100 for mm, z in rows.values() if mm is not None for e in ests]
        vals = [v for v in vals if v.size]
        if vals:
            lo, hi = full_range(np.concatenate(vals), [rt, rn])   # full finite sample range incl. truth/nominal
        else:
            lo, hi = min(rt, rn), max(rt, rn)
        pad = 0.08 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 80)
        for k_i, K in enumerate((10, 50)):
            ax = axes[i, k_i]
            mm, z = rows[K]
            if mm is None:
                ax.set_visible(False); continue
            if mm["degenerate_int"]:
                ax.axvline(rt, color=INK, lw=1.6)
                ax.axvline(rn, color=INK2, lw=1.6, ls="--")
                ax.text(0.5, 0.5, "EXACT CANCELLATION\nthetaR - thetaG = 0, a_eff = 1  ->  p_int = 1\n"
                        "interleaved curve flat (P00 = 1); lambda_int unidentifiable\n"
                        "3-parameter fits skipped; nominal r is negative",
                        transform=ax.transAxes, ha="center", va="center", fontsize=8.5,
                        bbox=dict(boxstyle="round", fc="#fff4ec", ec=C_P00))
                ax.set_yticks([])
            else:
                for e in ests:
                    r = z[f"r_{e}"] * 100
                    r = r[np.isfinite(r)]
                    check_cover(r, bins)
                    s = mm["summary"][e]
                    ax.hist(r, bins=bins, histtype="stepfilled" if e == "H" else "step",
                            alpha=0.35 if e == "H" else 1.0, lw=1.8, color=EST_COLOR[e],
                            label=f"{EST_LABEL[e]}: median {s['median']*100:.3f}, SD {s['std']*100:.3f}"
                                  f" (valid {s['n_valid']}/{mm['R']})")
                ax.axvline(rt, color=INK, lw=1.6)
                ax.axvline(rn, color=INK2, lw=1.6, ls="--")
                if "min" in ests and mm.get("r_pseudo_true_min") is not None:
                    ax.axvline(mm["r_pseudo_true_min"] * 100, color=C_MIN, lw=1.2, ls=":")
                ax.set_ylabel("runs")
            ax.set_xlim(bins[0], bins[-1])
            ax.set_title(f"{MODE_LABEL[mode]}, K={K}   [true {rt:.3f}, nominal {rn:.3f}]")
            ax.set_xlabel("inferred target error r (x1e-2)")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="best", fontsize=6.8, frameon=True, facecolor=SURF, edgecolor=GRID, framealpha=0.92)
    hl = [plt.Line2D([], [], color=INK, lw=1.6, label="true r_G = 1 - F_G"),
          plt.Line2D([], [], color=INK2, lw=1.6, ls="--", label="nominal asymptotic r_IRB = 0.75(1 - p_int/p_ref)")]
    if "min" in ests:
        hl.append(plt.Line2D([], [], color=C_MIN, lw=1.2, ls=":", label="pooled-min pseudo-true r (fit to oracle E[min] curves)"))
    fig.legend(handles=hl, loc="lower center", ncol=len(hl), bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(f"Inferred CZ error, F_R={FR}, F_G={FG} (true r_G={100*(1-FG):.0f}e-2): {' vs '.join(EST_LABEL[e] for e in ests)}",
                 y=1.005, fontsize=11)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    save(fig, f"fig3_r_hist_FR{int(FR*100)}_FG{int(FG*100)}_{tag}.png", out)


# ---------------------------------------------------------------- fig 4
def fig_summary(st, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=False)
    ests = ["P00", "H", "min"]
    offs = {(e, K): (ei - 1) * 0.22 + (-0.055 if K == 10 else 0.055) for ei, e in enumerate(ests) for K in (10, 50)}
    for ax, (FR, FG) in zip(axes.ravel(), PAIRS):
        for xi, mode in enumerate(MODES):
            m0 = st.get(FR, FG, mode, 10)[0] or st.get(FR, FG, mode, 50)[0]
            if m0 is None:
                continue
            rt = m0["r_true"]
            ax.plot([xi - 0.4, xi + 0.4], [m0["r_nominal"] / rt] * 2, color=INK, lw=2.2, solid_capstyle="butt")
            if m0["degenerate_int"]:
                ax.annotate("p_int = 1\nno fit", (xi, m0["r_nominal"] / rt), xytext=(0, 12), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK2)
                continue
            for K in (10, 50):
                mm, _ = st.get(FR, FG, mode, K)
                if mm is None:
                    continue
                for e in ests:
                    s = mm["summary"][e]
                    if "median" not in s:
                        continue
                    x = xi + offs[(e, K)]
                    med, lo, hi = s["median"] / rt, s["p2_5"] / rt, s["p97_5"] / rt
                    ax.plot([x, x], [lo, hi], color=EST_COLOR[e], lw=2 if K == 50 else 1.2)
                    ax.plot(x, med, marker="o" if K == 50 else "o", ms=6, color=EST_COLOR[e],
                            mfc=EST_COLOR[e] if K == 50 else SURF, mew=1.6)
        ax.axhline(1.0, color=INK2, lw=0.9, ls="--")
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels([MODE_LABEL[mm].replace(", ", "\n") for mm in MODES], fontsize=8)
        ax.set_title(f"F_R = {FR}, F_G = {FG}  (true r_G = {100*(1-FG):.0f}e-2)")
        ax.set_ylabel("r_est / r_true")
    handles = [plt.Line2D([], [], color=EST_COLOR[e], marker="o", lw=1.6, label=EST_LABEL[e]) for e in ests]
    handles += [plt.Line2D([], [], color=INK2, marker="o", mfc=SURF, lw=1.2, label="K=10 (hollow, thin)"),
                plt.Line2D([], [], color=INK2, marker="o", lw=2, label="K=50 (filled, thick)"),
                plt.Line2D([], [], color=INK, lw=2.2, label="nominal asymptotic r_IRB"),
                plt.Line2D([], [], color=INK2, lw=0.9, ls="--", label="truth (ratio = 1)")]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.03), fontsize=8.5)
    fig.suptitle("IRB bias and precision: MC median and 2.5-97.5% interval of r_est / r_true (all valid runs)",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig4_summary_bias_precision_grid.png", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st = Store(a.data)
    fig_single_circuit(st, a.out)
    for F in (0.95, 0.99):
        fig_pooled(st, a.out, F)
    for FR, FG in ((0.99, 0.99), (0.95, 0.99)):
        fig_r_hist(st, a.out, FR, FG, ["H", "P00"], "H_vs_P00")
        fig_r_hist(st, a.out, FR, FG, ["H", "min"], "H_vs_min")
    fig_summary(st, a.out)
    print(f"histogram coverage: {OUT_OF_RANGE['outside']} of {OUT_OF_RANGE['checked']} finite samples outside bins")
    if OUT_OF_RANGE["outside"]:
        raise SystemExit("ERROR: histogram bins clip samples")


if __name__ == "__main__":
    main()
