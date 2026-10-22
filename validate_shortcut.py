#!/usr/bin/env python3
"""
Independent validation of the toggling-frame shortcut against direct Clifford-matrix simulation.
Writes validation/validation_results.json and validation/validation_report.txt.
"""
import json
import math
import os
import time

import numpy as np

import irb_core as ic

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation")


def ks_2samp(a, b):
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    cdfa = np.searchsorted(a, allv, side="right") / a.size
    cdfb = np.searchsorted(b, allv, side="right") / b.size
    D = float(np.max(np.abs(cdfa - cdfb)))
    ne = a.size * b.size / (a.size + b.size)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * D
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return D, float(min(max(p, 0.0), 1.0))


def moments(pr):
    q1 = pr[:, 0] + pr[:, 1]; q2 = pr[:, 0] + pr[:, 2]
    return {"P00": pr[:, 0], "q1": q1, "H": (q1 + q2) / 2, "min_q": np.minimum(q1, q2)}


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(424242)
    res = {}

    # 1. group
    G, CZ = ic.clifford_group_2q()
    res["clifford_group_order"] = int(len(G))
    unit_err = float(np.max(np.abs(np.einsum("gij,gkj->gik", G, G.conj()) - np.eye(4))))
    res["clifford_unitarity_max_err"] = unit_err

    # 2. exhaustive conjugation of ZZ: C^+ ZZ C over all 11520 Cliffords
    counts = np.zeros((16, 2), dtype=int)
    for C in G:
        j, s = ic.signed_pauli_of(C.conj().T @ ic.PAULI[ic.ZZ_IDX] @ C)
        counts[j, (s + 1) // 2] += 1
    res["zz_conjugation_counts"] = {ic.PAULI_LABEL[j]: {"-": int(counts[j, 0]), "+": int(counts[j, 1])}
                                    for j in range(16)}
    nz = counts[1:].ravel()
    res["zz_conjugation_uniform_signed_nonidentity"] = bool(counts[0].sum() == 0 and np.all(nz == len(G) // 30))
    res["zz_conjugation_count_each"] = int(len(G) // 30)

    # 3. per-sequence exact equality direct vs toggling frame + distribution checks
    cases = [
        {"name": "ref_coherent_F95", "aR": 1.0, "thR": ic.noise_params(0.95, "coherent")[1], "inter": None},
        {"name": "ref_mixed_F99", "aR": ic.noise_params(0.99, "mixed")[0],
         "thR": ic.noise_params(0.99, "mixed")[1], "inter": None},
        {"name": "int_coherent_aligned_95_95", "aR": 1.0, "thR": ic.noise_params(0.95, "coherent")[1],
         "inter": (1.0, +ic.noise_params(0.95, "coherent")[1])},
        {"name": "int_coherent_opposing_95_99", "aR": 1.0, "thR": ic.noise_params(0.95, "coherent")[1],
         "inter": (1.0, -ic.noise_params(0.99, "coherent")[1])},
        {"name": "int_mixed_opposing_95_95", "aR": ic.noise_params(0.95, "mixed")[0],
         "thR": ic.noise_params(0.95, "mixed")[1],
         "inter": (ic.noise_params(0.95, "mixed")[0], -ic.noise_params(0.95, "mixed")[1])},
    ]
    n_dir, m_dir = 3000, 12
    seq_res = []
    pair_counts = np.zeros((30, 30), dtype=np.int64)
    for cs in cases:
        seq = rng.integers(0, len(G), size=(n_dir, m_dir))
        inter = None
        aeff, theff = cs["aR"], cs["thR"]
        if cs["inter"] is not None:
            aG, thG = cs["inter"]
            inter = (CZ, aG, thG)
            aeff, theff = cs["aR"] * aG, cs["thR"] + thG
        pr_dir, idx, sgn = ic.simulate_direct(G, seq, cs["aR"], cs["thR"], inter)
        # (a) deterministic: same Clifford sequence -> derived signed Paulis -> toggling-frame state
        am = aeff ** m_dir
        pr_tog = am * ic.apply_signed_paulis(idx, sgn, theff) + (1 - am) / 4
        exact_err = float(np.max(np.abs(pr_dir - pr_tog)))
        # (b) derived Paulis uniform? (30 signed classes)
        cls = (idx - 1) * 2 + (sgn > 0)
        hist = np.bincount(cls.ravel(), minlength=30)
        exp_ = cls.size / 30
        chi2 = float(np.sum((hist - exp_) ** 2 / exp_))
        for k in range(m_dir - 1):
            np.add.at(pair_counts, (cls[:, k], cls[:, k + 1]), 1)
        # (c) distribution: direct vs independent shortcut sample
        pr_sc = ic.simulate_shortcut(rng, 100000, m_dir, theff, aeff)
        md, ms = moments(pr_dir), moments(pr_sc)
        p = aeff * ic.pu(theff)
        theory = {"P00": (1 + 3 * p ** m_dir) / 4, "q1": (1 + p ** m_dir) / 2, "H": (1 + p ** m_dir) / 2}
        comp = {}
        for k in md:
            a, b = md[k], ms[k]
            se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
            if max(a.std(), b.std()) < 1e-12:   # degenerate (deterministic) distribution
                th_ = theory.get(k, b.mean())
                comp[k] = {"degenerate_zero_variance": True, "mean_direct": float(a.mean()),
                           "mean_shortcut": float(b.mean()), "theory_mean": float(th_),
                           "max_abs_dev_from_theory": float(max(np.abs(a - th_).max(), np.abs(b - th_).max()))}
                continue
            D, pks = ks_2samp(a, b)
            row = {"mean_direct": float(a.mean()), "mean_shortcut": float(b.mean()),
                   "z_direct_vs_shortcut": float((a.mean() - b.mean()) / se) if se > 0 else 0.0,
                   "sd_direct": float(a.std(ddof=1)), "sd_shortcut": float(b.std(ddof=1)),
                   "ks_D": D, "ks_p": pks}
            if k in theory and k != "min_q":
                row["theory_mean"] = float(theory[k])
                row["z_shortcut_vs_theory"] = float((b.mean() - theory[k]) / (b.std(ddof=1) / math.sqrt(b.size))) \
                    if b.std() > 0 else 0.0
                row["z_direct_vs_theory"] = float((a.mean() - theory[k]) / (a.std(ddof=1) / math.sqrt(a.size))) \
                    if a.std() > 0 else 0.0
            comp[k] = row
        seq_res.append({"case": cs["name"], "m": m_dir, "n_direct": n_dir, "n_shortcut": 100000,
                        "a_eff": aeff, "theta_eff": theff, "p_eff": p,
                        "max_abs_diff_direct_vs_toggling_same_sequence": exact_err,
                        "derived_pauli_chi2_29dof": chi2, "comparison": comp})
    res["sequence_checks"] = seq_res
    npairs = pair_counts.sum()
    e = npairs / 900
    chi2p = float(np.sum((pair_counts - e) ** 2 / e))
    res["consecutive_pair_independence"] = {"pairs": int(npairs), "chi2": chi2p, "dof": 899,
                                            "z": (chi2p - 899) / math.sqrt(2 * 899)}

    # 4. degenerate exact cancellation: coherent opposing equal F -> theta_eff = 0, a_eff = 1 -> P00 = 1
    thc = ic.noise_params(0.99, "coherent")[1]
    seq = rng.integers(0, len(G), size=(200, 30))
    pr_deg, _, _ = ic.simulate_direct(G, seq, 1.0, thc, (CZ, 1.0, -thc))
    res["exact_cancellation_direct_P00_min"] = float(pr_deg[:, 0].min())

    # 5. noise parametrisation: Favg exact
    fr = []
    for F in (0.95, 0.99):
        for kind in ("depol", "coherent", "mixed"):
            a, th = ic.noise_params(F, kind)
            # independent Favg via entanglement fidelity of D_a o U: Fe = a cos^2(th/2) + (1-a)/16
            Fe = a * math.cos(th / 2) ** 2 + (1 - a) / 16
            fr.append({"F": F, "kind": kind, "a": a, "theta": th, "Favg_formula": ic.favg(a, th),
                       "Favg_from_Fe": (4 * Fe + 1) / 5})
    res["noise_parametrisation"] = fr

    # 6. fitter self-test: noiseless recovery + finite-difference jacobian
    x = np.array([1, 3, 7, 15, 26, 37, 56, 75, 112, 150, 224], float)
    th_true = np.array([[0.74, 0.26, 0.0268], [0.5, 0.5, 0.0134], [0.3, 0.6, 0.1]])
    Y = ic._model(th_true, x)
    th, conv, _ = ic.fit_wls(Y, x, np.full(x.size, 1e-4), np.array([0.6, 0.3, 0.02]))
    Jn = np.zeros((3, x.size, 3))
    for i in range(3):
        d = np.zeros(3); d[i] = 1e-7
        Jn[:, :, i] = (ic._model(th_true + d, x) - ic._model(th_true - d, x)) / 2e-7
    res["fitter_selftest"] = {"max_param_err_noiseless": float(np.max(np.abs(th - th_true))),
                              "all_converged": bool(conv.all()),
                              "jacobian_max_abs_err": float(np.max(np.abs(Jn - ic._jac(th_true, x))))}
    res["runtime_s"] = time.time() - t0

    with open(os.path.join(OUT, "validation_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    L = ["Toggling-frame shortcut validation", ""]
    L.append(f"Clifford group order: {res['clifford_group_order']} (expected 11520); unitarity err {unit_err:.1e}")
    L.append(f"C^+ ZZ C over all Cliffords: uniform over 30 signed non-identity Paulis, "
             f"{res['zz_conjugation_count_each']} each: {res['zz_conjugation_uniform_signed_nonidentity']}")
    pc = res["consecutive_pair_independence"]
    L.append(f"Consecutive derived-Pauli pair independence: chi2={pc['chi2']:.1f} / 899 dof (z={pc['z']:+.2f}), "
             f"{pc['pairs']} pairs")
    L.append(f"Exact cancellation (coherent opposing F=.99/.99, direct sim): min P00 = {res['exact_cancellation_direct_P00_min']:.15f}")
    L.append("")
    for s in seq_res:
        L.append(f"[{s['case']}] m={s['m']} a_eff={s['a_eff']:.5f} theta_eff={s['theta_eff']:+.5f} p_eff={s['p_eff']:.5f}")
        L.append(f"   same-sequence |direct - toggling| max = {s['max_abs_diff_direct_vs_toggling_same_sequence']:.2e};"
                 f" derived Pauli chi2 = {s['derived_pauli_chi2_29dof']:.1f} (29 dof)")
        for k, v in s["comparison"].items():
            if v.get("degenerate_zero_variance"):
                L.append(f"   {k:<6} deterministic: direct {v['mean_direct']:.6f} shortcut {v['mean_shortcut']:.6f}"
                         f" theory {v['theory_mean']:.6f} max|dev| {v['max_abs_dev_from_theory']:.1e}")
                continue
            th = (f" theory {v['theory_mean']:.5f} z_sc={v['z_shortcut_vs_theory']:+.2f} z_dir={v['z_direct_vs_theory']:+.2f}"
                  if "theory_mean" in v else "")
            L.append(f"   {k:<6} mean dir {v['mean_direct']:.5f} sc {v['mean_shortcut']:.5f} (z {v['z_direct_vs_shortcut']:+.2f})"
                     f"  sd dir {v['sd_direct']:.5f} sc {v['sd_shortcut']:.5f}  KS D={v['ks_D']:.4f} p={v['ks_p']:.3f}{th}")
    L.append("")
    for r in fr:
        L.append(f"F={r['F']} {r['kind']:<8} a={r['a']:.6f} theta={r['theta']:.6f}  Favg={r['Favg_formula']:.10f}"
                 f" (via Fe: {r['Favg_from_Fe']:.10f})")
    ft = res["fitter_selftest"]
    L.append(f"\nFitter self-test: noiseless max param err {ft['max_param_err_noiseless']:.1e}, converged {ft['all_converged']},"
             f" jacobian FD err {ft['jacobian_max_abs_err']:.1e}")
    L.append(f"runtime {res['runtime_s']:.1f}s")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "validation_report.txt"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
