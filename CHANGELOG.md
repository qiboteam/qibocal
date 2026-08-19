# Changelog

## Unreleased

### `utils.py` reorganization (see discussion #1660)

Large, heterogeneous `utils.py` files were split into themed modules, following
the style already used by `ramsey`, `zz_interaction` and `cross_resonance`.

**Global `qibocal.protocols`** — the former 1263-line `utils.py` was split into:

- `constants.py` — shared constants and enums (`FeatExtractionError`, `PowerLevel`, ...)
- `physics.py` — physics helpers (Lorentzian models, `effective_qubit_temperature`, `baseline_als`, ...)
- `fitting.py` — fitting/guessing helpers (`chi2_reduced*`, `guess_frequency`, `quinn_fernandes_algorithm`, `to_range`, ...)
- `processing.py` — data processing (`compute_qnd`, `clustering`, `peaks_finder`, `zca_whiten`, ...)
- `reporting.py` — report helpers (`eval_magnitude`, `round_report`, `table_dict`, ...)
- `plotting.py` — plotting helpers (`evaluate_grid`, `plot_results`)

**Per-protocol splits:**

- `coherence/` → `acquisition.py`, `fitting.py`, `plotting.py`
- `flux_dependence/` → `parameters.py`, `acquisition.py`, `physics.py`, `fitting.py`, `plotting.py`
- `randomized_benchmarking/` → `types.py`, `circuit_generation.py`, `acquisition.py`, and fit helpers in the existing `fitting.py`
- `rabi/` → `fitting.py`, `plotting.py`, `acquisition.py`
- `two_qubit_interaction/cross_resonance/` → remaining `utils.py` content folded into a new `processing.py`; `utils.py` removed

**Backward compatibility:** every split folder except `cross_resonance/` keeps a
deprecated `utils.py` shim that re-exports the original public names, so existing
`...<folder>.utils.<name>` imports keep working. The shim in each module is
deprecated and will be removed in a future release — import from the themed
modules directly instead. Small, coherent `utils.py` files (e.g. `zz_interaction`,
`chevron`, `chsh`, `signal_experiments`) were left unchanged.
