# irb_marginal_followup — circuit-level 2q RB + CZ-interleaved RB, marginal-readout estimators

## Reproduce
```
python3 validate_shortcut.py                 # numpy only, ~5 s   -> validation/
python3 run_irb_sim.py --reps 1000           # numpy only, ~140 s -> data/, results/, progress.log (resumable)
uv run --no-project --with matplotlib --with numpy python make_figures.py   # -> figures/
```
Seeds are fixed: scenario i uses SeedSequence(20261022, spawn_key=(i,)), and oracle banks use key-derived seeds.

## Exact model
- Reference Clifford C, then D_aR ∘ U_ZZ(θR). Target CZ, then D_aG ∘ U_ZZ(±θG). Recovery is ideal. Readout is ideal.
- U_ZZ(θ)=exp(-iθZZ/2) and D_a(ρ)=aρ+(1-a)I/4. Favg = [1+3a·pu(θ)]/4 with pu=(16cos²(θ/2)-1)/15.
- Pure depolarizing: a=(4F-1)/3, θ=0. Pure coherent: a=1, θ=2asin(√(5(1-F)/4)). Mixed: a=pu(θ)=√pF (exact F, not additive halves).
- ZZ commutes with CZ, so an interleaved block has noise D_{aR·aG} ∘ U_ZZ(θR±θG) after the uniform Clifford CZ·C.
  This gives p_ref = aR·pu(θR) and p_int = aR·aG·pu(θR±θG).
- r_true = 1-F_G. The nominal asymptotic IRB value is r_IRB = 0.75(1-p_int/p_ref).
- Simulation uses the exact toggling-frame shortcut: iid uniform signed non-identity Paulis P_k. Each step applies
  exp(-iθ_eff s_k P_k/2) to |00>, and the final state is ρ = a_eff^m|ψ><ψ| + (1-a_eff^m)I/4.
  `validate_shortcut.py` checks this against direct density-matrix simulation with the full 11520-element 2q Clifford group.
- Each experiment has independent reference and interleaved data sets, and K fresh circuits at every depth.
  One multinomial(S=500) draw per circuit feeds every estimator: P00, q1, H=(q1+q2)/2, and min(q1,q2) formed after pooling the K circuits.
  No random trajectories are shared between scenarios, depths or protocols.
- Depths: m = unique(max(1, round(t/λ))) for t ∈ {.05,.1,.2,.4,.7,1,1.5,2,3,4,6}, computed separately for each curve from its analytic λ.
- Fits: A·exp(-λm)+B with A, B, λ all free (λ sign unconstrained) by weighted least squares.
  The weights are ORACLE variances from an independent 20000-circuit ensemble per (curve, depth): exact shot+circuit variance for
  the linear estimators, and the pooled-min variance by resampling. The fit starts at (0.9A₀, B₀+0.02, 0.8λ_true) with one retry from (A₀, B₀, 1.25λ_true).
  Converged means a finite Gauss-Newton decrement < 1e-10. Non-converged fits are counted and excluded from the statistics, and
  negative r is never clipped.
- Degenerate cases: coherent opposing with F_R=F_G gives θ_eff=0 and a_eff=1, so p_int=1 exactly. The interleaved fits are skipped, and the reason
  and nominal r are recorded.

## Environment (versions actually used)
- Simulation and validation: Python 3.9.6, numpy 1.26.2. No scipy.
- Figures: uv 0.13.1 ephemeral env (`uv run --no-project --with matplotlib --with numpy`), Python 3.13.15, numpy 2.5.4, matplotlib 3.11.2.
  See `requirements.txt`.

## Scope notes for readers
- There are 40 scenarios: 4 (F_R,F_G) pairs × 5 noise modes × K∈{10,50}. **Four of them are degenerate and are excluded from fitting**:
  coherent ZZ opposing with F_R=F_G=0.95 or 0.99, at K=10 and K=50.
  In those cases p_int=1 exactly, so the interleaved curve is flat and λ_int cannot be identified. Only the nominal r_IRB (negative) is reported.
  **The other 36 combinations are fitted stochastically**, with 1000 independent experiments each.
- The reported `p2_5`/`p97_5` are **empirical 2.5/97.5 percentiles of the 1000 Monte Carlo estimates**, i.e. the sampling spread
  of the estimator. They are **not** confidence intervals from individual fits.
- The fractions "within 10/20%" are taken over all 1000 runs, and failed fits count as misses.
- Histograms span the **full finite sample range**: no tails are clipped, and `make_figures.py` aborts if any sample falls outside the bins.
  Bins and x-axes are shared within each row, and the truth/nominal lines are always drawn.
- The bundle `publication/reproducibility_bundle.tar.gz` holds code, README, results (the per-run CSV gzipped), validation, raw
  scenario npz/json data and figures. It leaves out `__pycache__` and the run logs.
