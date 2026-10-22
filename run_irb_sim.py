#!/usr/bin/env python3
"""
Circuit-level 2q Clifford RB + CZ interleaved RB Monte Carlo (toggling-frame exact shortcut, see irb_core).

Scenarios: (F_R, F_G) in {.95,.99}^2  x  mode in {depol, coh_aligned, coh_opposing, mixed_aligned, mixed_opposing}
           x K in {10, 50};  S = 500 shots per circuit; R independent experiments per scenario.
Each experiment: independent reference + interleaved data sets; every depth uses K freshly sampled circuits;
one multinomial bitstring draw per circuit feeds all estimators (P00, q1, H=(q1+q2)/2, min(q1,q2) after pooling).
Each curve fitted with A exp(-lam m) + B, all 3 free, WLS with ORACLE variances from an independent
20000-circuit ensemble per (curve, depth) (shot + circuit variance; pooled-min variance by resampling).

Resumable: per-scenario outputs in data/; progress in progress.log.
Usage: python3 run_irb_sim.py [--reps 1000] [--aggregate-only] [--only ID,ID]
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np

import irb_core as ic

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
LOG = os.path.join(HERE, "progress.log")
BASE_SEED = 20261022
S = 500
T_GRID = np.array([0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
PAIRS = [(0.99, 0.99), (0.95, 0.99), (0.95, 0.95), (0.99, 0.95)]
MODES = {"depol": ("depol", +1), "coh_aligned": ("coherent", +1), "coh_opposing": ("coherent", -1),
         "mixed_aligned": ("mixed", +1), "mixed_opposing": ("mixed", -1)}
KS = (10, 50)
BANK_N = 20000
ORACLE_RESAMPLE = 10000


def scenarios():
    out = []
    for FR, FG in PAIRS:
        for mode in MODES:
            for K in KS:
                out.append({"id": len(out), "F_R": FR, "F_G": FG, "mode": mode, "K": K})
    return out


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def schedule(lam):
    return np.unique(np.maximum(1, np.round(T_GRID / lam).astype(int)))


def curve_params(sc):
    kind, sgn = MODES[sc["mode"]]
    aR, thR = ic.noise_params(sc["F_R"], kind)
    aG, thG = ic.noise_params(sc["F_G"], kind)
    ref = {"a": aR, "theta": thR}
    inte = {"a": aR * aG, "theta": thR + sgn * thG}
    for c in (ref, inte):
        c["p"] = c["a"] * ic.pu(c["theta"])
        c["lam"] = -math.log(c["p"]) if c["p"] < 1 else 0.0
    return ref, inte, {"aR": aR, "thR": thR, "aG": aG, "thG_signed": sgn * thG}


def curve_key(c):
    return f"a{c['a']:.12f}_th{c['theta']:+.12f}"


# ---------------------------------------------------------------- oracle ensemble (cached across scenarios)
_ORACLE = {}


def oracle(c, m, want_bank=False):
    key = (curve_key(c), int(m))
    if key in _ORACLE and not want_bank:
        return _ORACLE[key][0]
    ka = int(round(c["a"] * 1e9)) % (2**31)
    kt = int(round((c["theta"] + 10) * 1e9)) % (2**31)
    rng = np.random.default_rng(np.random.SeedSequence(BASE_SEED, spawn_key=(10**6, ka, kt, int(m))))
    pr = ic.simulate_shortcut(rng, BANK_N, int(m), c["theta"], c["a"])
    st = {"m": int(m), "theory_P00": (1 + 3 * c["p"] ** m) / 4, "theory_marg": (1 + c["p"] ** m) / 2,
          "bank_mean_P00": float(pr[:, 0].mean()), "bank_se_P00": float(pr[:, 0].std(ddof=1) / math.sqrt(BANK_N)),
          "var": {}, "mean_min": {}, "se_mean_min": {}, "circ_sd": {}}
    for e, w in ic.EST_W.items():
        mu_c = pr @ w
        vshot_c = pr @ (w * w) - mu_c ** 2
        st["circ_sd"][e] = float(mu_c.std(ddof=1))
        for K in KS:
            st["var"].setdefault(str(K), {})[e] = float((vshot_c.mean() / S + mu_c.var(ddof=1)) / K)
    for K in KS:
        idx = rng.integers(0, BANK_N, size=(ORACLE_RESAMPLE, K))
        cnt = rng.multinomial(S, pr[idx]).sum(1)
        mn = ic.pooled_estimators(cnt)["min"]
        st["var"][str(K)]["min"] = float(mn.var(ddof=1))
        st["mean_min"][str(K)] = float(mn.mean())
        st["se_mean_min"][str(K)] = float(mn.std(ddof=1) / math.sqrt(ORACLE_RESAMPLE))
    _ORACLE[key] = (st,)
    if want_bank:
        q1 = pr[:, 0] + pr[:, 1]; q2 = pr[:, 0] + pr[:, 2]
        cnt1 = rng.multinomial(S, pr)
        shot = ic.pooled_estimators(cnt1)
        return st, {"H_circ": ((q1 + q2) / 2).astype(np.float32), "H_shot": shot["H"].astype(np.float32),
                    "P00_circ": pr[:, 0].astype(np.float32)}
    return st


# ---------------------------------------------------------------- fitting
def fit_curve(Y, m, V, A0, B0, lam0):
    th, conv, _ = ic.fit_wls(Y, m, V, np.array([0.9 * A0, B0 + 0.02, 0.8 * lam0]))
    retry = ~conv
    if retry.any():
        th2, conv2, _ = ic.fit_wls(Y[retry], m, V, np.array([A0, B0, 1.25 * lam0]))
        th[retry] = np.where(conv2[:, None], th2, th[retry])
        conv[retry] = conv2
    return th, conv, retry


def simulate_curve(rng, c, ms, R, K):
    """Return dict est -> (R, nd) pooled estimator values."""
    out = {e: np.empty((R, len(ms))) for e in ic.EST_NAMES}
    for j, m in enumerate(ms):
        pr = ic.simulate_shortcut(rng, R * K, int(m), c["theta"], c["a"])
        cnt = rng.multinomial(S, pr).reshape(R, K, 4).sum(1)
        est = ic.pooled_estimators(cnt)
        for e in ic.EST_NAMES:
            out[e][:, j] = est[e]
    return out


def run_scenario(sc, R):
    t0 = time.time()
    sid = sc["id"]
    K = sc["K"]
    ref, inte, raw = curve_params(sc)
    rG = 1 - sc["F_G"]
    degenerate = abs(inte["p"] - 1) < 1e-12
    r_nom = 0.75 * (1 - inte["p"] / ref["p"])
    rng = np.random.default_rng(np.random.SeedSequence(BASE_SEED, spawn_key=(sid,)))
    meta = {**sc, "R": R, "S": S, "ref": ref, "int": inte, "raw_noise": raw, "r_true": rG,
            "r_nominal": r_nom, "bias_nominal_abs": r_nom - rG, "bias_nominal_rel": (r_nom - rG) / rG,
            "degenerate_int": bool(degenerate)}
    curves = {"ref": ref} if degenerate else {"ref": ref, "int": inte}
    sched = {k: schedule(c["lam"]) for k, c in curves.items()}
    if degenerate:
        sched["int"] = sched["ref"]
        meta["degenerate_reason"] = ("coherent opposing with F_R = F_G: theta_eff = 0 and a_eff = 1 -> "
                                     "p_int = 1 exactly, interleaved signal is constant (P00 = 1 at every depth); "
                                     "lambda_int unidentifiable under free A,B; 3-param fits skipped, nominal reported.")
        chk = ic.simulate_shortcut(rng, 1000, int(sched["ref"][-1]), inte["theta"], inte["a"])
        meta["degenerate_check_int_P00_min_at_max_depth"] = float(chk[:, 0].min())
    meta["depths"] = {k: v.tolist() for k, v in sched.items()}

    npz = {}
    fits = {}
    oracle_stats = {}
    theory_checks = {}
    for cname, c in curves.items():
        ms = sched[cname]
        it1 = int(np.argmin(np.abs(ms * c["lam"] - 1.0)))
        stats = []
        for j, m in enumerate(ms):
            if j == it1:
                st, bank = oracle(c, m, want_bank=True)
                bpath = os.path.join(DATA, f"bank_t1_{cname}_{curve_key(c)}.npz")
                if not os.path.exists(bpath):
                    np.savez_compressed(bpath, m=m, **bank)
            else:
                st = oracle(c, m)
            stats.append(st)
        oracle_stats[cname] = stats
        Ys = simulate_curve(rng, c, ms, R, K)
        V = {e: np.array([s["var"][str(K)][e] for s in stats]) for e in ic.EST_NAMES}
        tc = {}
        for e in ("P00", "H"):
            th_ = np.array([s["theory_P00"] if e == "P00" else s["theory_marg"] for s in stats])
            tc[e] = float(np.max(np.abs((Ys[e].mean(0) - th_) / np.sqrt(V[e] / R))))
        mm = np.array([s["mean_min"][str(K)] for s in stats])
        # SE: MC runs + oracle resampling + finite-bank error (approx. circuit SD of E[q1|C]/sqrt(BANK_N))
        se = np.sqrt(V["min"] / R + np.array([s["se_mean_min"][str(K)] for s in stats]) ** 2
                     + np.array([s["circ_sd"]["q1"] for s in stats]) ** 2 / BANK_N)
        tc["min_vs_oracle"] = float(np.max(np.abs((Ys["min"].mean(0) - mm) / se)))
        theory_checks[cname] = {"max_abs_z_per_depth": tc, "n_depths": len(ms), "t1_index": it1}
        for e in ic.EST_NAMES:
            A0, B0 = ic.EST_AB[e]
            th, conv, retry = fit_curve(Ys[e], ms.astype(float), V[e], A0, B0, c["lam"])
            fits[(cname, e)] = (th, conv, retry)
            npz[f"{cname}_{e}_Y"] = Ys[e].astype(np.float32)
            npz[f"{cname}_{e}_theta"] = th
            npz[f"{cname}_{e}_conv"] = conv
            npz[f"{cname}_{e}_retry"] = retry
            npz[f"{cname}_{e}_V"] = V[e]
        npz[f"{cname}_m"] = ms
        # pseudo-true lambda of the naive pooled-min curve (fit to oracle-mean min curve)
        thm, convm, _ = ic.fit_wls(mm[None, :], ms.astype(float), V["min"], np.array([0.45, 0.52, 0.8 * c["lam"]]))
        meta.setdefault("min_pseudo_true", {})[cname] = {"lam": float(thm[0, 2]), "converged": bool(convm[0]),
                                                          "lam_nominal": c["lam"]}
    meta["oracle"] = oracle_stats
    meta["theory_checks"] = theory_checks

    # IRB estimates
    summ = {}
    if not degenerate:
        lr, li = meta["min_pseudo_true"]["ref"]["lam"], meta["min_pseudo_true"]["int"]["lam"]
        meta["r_pseudo_true_min"] = 0.75 * (1 - math.exp(lr - li))
    for e in ic.EST_NAMES:
        thr, cr, rr = fits[("ref", e)]
        row = {"estimator": e, "r_true": rG, "r_nominal": r_nom,
               "r_pseudo_true": meta.get("r_pseudo_true_min") if e == "min" else (None if degenerate else r_nom),
               "fail_ref": int((~cr).sum()), "retry_ref": int(rr.sum()),
               "nonphys_ref": int((cr & (thr[:, 2] <= 0)).sum()),
               "extreme_ref": int((cr & (thr[:, 2] > 20 * ref["lam"])).sum())}
        if degenerate:
            r = np.full(R, np.nan)
            row.update({"status": "SKIPPED_DEGENERATE", "n_valid": 0})
        else:
            thi, ci, ri = fits[("int", e)]
            with np.errstate(over="ignore", invalid="ignore"):
                r = 0.75 * (1 - np.exp(thr[:, 2] - thi[:, 2]))
            valid = cr & ci & np.isfinite(r)
            rv = r[valid]
            row.update({"status": "OK", "n_valid": int(valid.sum()),
                        "fail_int": int((~ci).sum()), "retry_int": int(ri.sum()),
                        "nonphys_int": int((ci & (thi[:, 2] <= 0)).sum()),
                        "extreme_int": int((ci & (thi[:, 2] > 20 * inte["lam"])).sum()),
                        "neg_r": int((rv < 0).sum())})
            if rv.size:
                row.update({"mean": float(rv.mean()), "median": float(np.median(rv)), "std": float(rv.std(ddof=1)),
                            "p2_5": float(np.percentile(rv, 2.5)), "p97_5": float(np.percentile(rv, 97.5)),
                            "mc_bias_vs_true_rel": float(rv.mean() / rG - 1),
                            "median_bias_vs_true_rel": float(np.median(rv) / rG - 1),
                            "rel_std_vs_true": float(rv.std(ddof=1) / rG)})
            # fractions over ALL R runs (failed fits count as misses)
            for tol in (0.1, 0.2):
                row[f"frac_within_{int(tol*100)}pct_true"] = float(np.sum(valid & (np.abs(r - rG) <= tol * rG)) / R)
                row[f"frac_within_{int(tol*100)}pct_nominal"] = (
                    float(np.sum(valid & (np.abs(r - r_nom) <= tol * abs(r_nom))) / R) if abs(r_nom) > 1e-12 else None)
        npz[f"r_{e}"] = r
        summ[e] = row
    meta["summary"] = summ
    meta["runtime_s"] = time.time() - t0
    np.savez_compressed(os.path.join(DATA, f"scen_{sid:02d}.npz"), **npz)
    with open(os.path.join(DATA, f"scen_{sid:02d}.json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    return meta


# ---------------------------------------------------------------- aggregation
SUM_COLS = ["id", "F_R", "F_G", "mode", "K", "estimator", "status", "r_true", "r_nominal", "bias_nominal_rel",
            "r_pseudo_true", "n_valid", "mean", "median", "std", "p2_5", "p97_5", "mc_bias_vs_true_rel",
            "median_bias_vs_true_rel", "rel_std_vs_true", "frac_within_10pct_true", "frac_within_20pct_true",
            "frac_within_10pct_nominal", "frac_within_20pct_nominal", "fail_ref", "fail_int", "retry_ref",
            "retry_int", "nonphys_ref", "nonphys_int", "extreme_ref", "extreme_int", "neg_r"]


def aggregate():
    os.makedirs(RES, exist_ok=True)
    rows, metas = [], []
    for sc in scenarios():
        p = os.path.join(DATA, f"scen_{sc['id']:02d}.json")
        if not os.path.exists(p):
            continue
        meta = json.load(open(p))
        metas.append(meta)
        for e, s in meta["summary"].items():
            rows.append({**{k: meta[k] for k in ("id", "F_R", "F_G", "mode", "K")},
                         "bias_nominal_rel": meta["bias_nominal_rel"], **s})
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUM_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in SUM_COLS})
    # per-run estimates
    with open(os.path.join(RES, "per_run_estimates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "F_R", "F_G", "mode", "K", "run", "estimator", "A_ref", "B_ref", "lam_ref",
                    "conv_ref", "retry_ref", "A_int", "B_int", "lam_int", "conv_int", "retry_int", "r_est", "status"])
        for meta in metas:
            z = np.load(os.path.join(DATA, f"scen_{meta['id']:02d}.npz"))
            R = meta["R"]
            for e in ic.EST_NAMES:
                tr, cr, rr = z[f"ref_{e}_theta"], z[f"ref_{e}_conv"], z[f"ref_{e}_retry"]
                deg = meta["degenerate_int"]
                if not deg:
                    ti, ci, ri = z[f"int_{e}_theta"], z[f"int_{e}_conv"], z[f"int_{e}_retry"]
                r = z[f"r_{e}"]
                for i in range(R):
                    if deg:
                        st = "SKIPPED_DEGENERATE_int"
                        w.writerow([meta["id"], meta["F_R"], meta["F_G"], meta["mode"], meta["K"], i, e,
                                    f"{tr[i,0]:.6g}", f"{tr[i,1]:.6g}", f"{tr[i,2]:.6g}", int(cr[i]), int(rr[i]),
                                    "", "", "", "", "", "", st])
                        continue
                    flags = []
                    if not cr[i]: flags.append("ref_fail")
                    if not ci[i]: flags.append("int_fail")
                    if cr[i] and tr[i, 2] <= 0: flags.append("ref_lam_nonpos")
                    if ci[i] and ti[i, 2] <= 0: flags.append("int_lam_nonpos")
                    w.writerow([meta["id"], meta["F_R"], meta["F_G"], meta["mode"], meta["K"], i, e,
                                f"{tr[i,0]:.6g}", f"{tr[i,1]:.6g}", f"{tr[i,2]:.6g}", int(cr[i]), int(rr[i]),
                                f"{ti[i,0]:.6g}", f"{ti[i,1]:.6g}", f"{ti[i,2]:.6g}", int(ci[i]), int(ri[i]),
                                f"{r[i]:.6g}", ";".join(flags) or "ok"])
    js = {"scenarios": [{k: m[k] for k in ("id", "F_R", "F_G", "mode", "K", "R", "r_true", "r_nominal",
                                            "bias_nominal_rel", "degenerate_int", "depths", "runtime_s",
                                            "theory_checks", "min_pseudo_true")}
                        | {"r_pseudo_true_min": m.get("r_pseudo_true_min"),
                           "degenerate_reason": m.get("degenerate_reason"),
                           "ref": m["ref"], "int": m["int"], "summary": m["summary"]} for m in metas]}
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(js, f, indent=1, default=float)
    # text table
    L = ["IRB marginal follow-up: summary (r in units of 1e-2; rel = relative to true r_G)", ""]
    L.append(f"{'FR':>4} {'FG':>4} {'mode':<15}{'K':>3} {'est':<4} {'r_true':>6} {'r_nom':>7} {'nomBias':>8} "
             f"{'mean':>7} {'median':>7} {'std':>6} {'p2.5':>7} {'p97.5':>7} {'w10T':>5} {'w20T':>5} {'w10N':>5} "
             f"{'w20N':>5} {'fail':>5} {'nonph':>5} {'neg':>4}")
    for r in rows:
        if r["estimator"] == "q1":
            continue
        def f(v, s=100, fmt="{:7.3f}"):
            return fmt.format(v * s) if isinstance(v, (int, float)) and v is not None else " " * (len(fmt.format(0)) )
        if r["status"] != "OK":
            L.append(f"{r['F_R']:>4} {r['F_G']:>4} {r['mode']:<15}{r['K']:>3} {r['estimator']:<4} "
                     f"{r['r_true']*100:6.3f} {r['r_nominal']*100:7.3f} {r['bias_nominal_rel']*100:7.1f}%  "
                     f"SKIPPED: p_int = 1 exactly (no decay); fits not attempted")
            continue
        wn10 = r.get("frac_within_10pct_nominal"); wn20 = r.get("frac_within_20pct_nominal")
        L.append(f"{r['F_R']:>4} {r['F_G']:>4} {r['mode']:<15}{r['K']:>3} {r['estimator']:<4} "
                 f"{r['r_true']*100:6.3f} {r['r_nominal']*100:7.3f} {r['bias_nominal_rel']*100:7.1f}% "
                 f"{r['mean']*100:7.3f} {r['median']*100:7.3f} {r['std']*100:6.3f} {r['p2_5']*100:7.3f} "
                 f"{r['p97_5']*100:7.3f} {r['frac_within_10pct_true']:5.2f} {r['frac_within_20pct_true']:5.2f} "
                 f"{(f'{wn10:5.2f}' if wn10 is not None else '  n/a')} {(f'{wn20:5.2f}' if wn20 is not None else '  n/a')} "
                 f"{r['fail_ref']+r['fail_int']:5d} {r['nonphys_ref']+r['nonphys_int']:5d} {r['neg_r']:4d}")
    L.append("")
    L.append("w10T/w20T: fraction of ALL runs within 10%/20% of true r_G; w10N/w20N: of nominal r_IRB (n/a if r_nom=0).")
    L.append("fail = ref+int non-converged fits (after one retry); nonph = converged fits with lambda<=0; neg = r_est<0.")
    with open(os.path.join(RES, "summary.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if not args.aggregate_only:
        only = {int(x) for x in args.only.split(",") if x}
        t0 = time.time()
        log(f"start reps={args.reps}")
        for sc in scenarios():
            if only and sc["id"] not in only:
                continue
            if os.path.exists(os.path.join(DATA, f"scen_{sc['id']:02d}.json")):
                log(f"skip scenario {sc['id']} (exists)")
                continue
            meta = run_scenario(sc, args.reps)
            s = meta["summary"]
            msg = " ".join(f"{e}:{s[e].get('median', float('nan'))*100:.3f}" for e in ("P00", "H", "min"))
            log(f"done {sc['id']:02d} FR={sc['F_R']} FG={sc['F_G']} {sc['mode']:<14} K={sc['K']:<2} "
                f"r_true={meta['r_true']*100:.2f} r_nom={meta['r_nominal']*100:.3f} med(e-2) {msg} "
                f"[{meta['runtime_s']:.1f}s, total {time.time()-t0:.0f}s]")
        log(f"all scenarios done in {time.time()-t0:.0f}s")
    aggregate()
    log("aggregated results written")


if __name__ == "__main__":
    main()
