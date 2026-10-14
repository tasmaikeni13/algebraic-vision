# Phase 1 research

This directory is isolated from `algebraic_vision/`. It adds mathematical
references, simulations and formal analysis; it does not implement a production
third arm. The current [analytical report](../../results/phase1/REPORT.md) records
**NO-GO**: the prompt's strict dominance gate remains unsatisfied.

```bash
virtualenv .venv                         # or python -m venv .venv
.venv/bin/python -m pip install -r requirements-phase1.txt
.venv/bin/python -m analysis.phase1.run
.venv/bin/python -m analysis.phase1.verify
# Deliberately fails with exit 2 if the research gate is not satisfied:
.venv/bin/python -m analysis.phase1.run --require-pass
```

The runner sets CPU, FP64 reference support and single-threaded BLAS defaults
before importing JAX. It never initializes multi-host training or launches a
sweep. Override `--out`, `--seeds`, `--prototype-seeds`, `--steps`, or
`--sections symbolic,precision` for exploratory runs. Reduced coverage is
recorded in provenance and cannot satisfy the full evidence gate. By default,
Monte Carlo seeds are 42-49, prototype seeds are 42/43/44, and training is
200 full-batch steps at each of five logarithmically spaced learning rates.

| Prompt requirement | Implementation and authoritative evidence |
| --- | --- |
| Rational identity, domain, series, bounds, Q/K/kappa derivatives | `symbolic.py`, `symbolic.json`, `V2Kernel.lean`, `V2Floor.lean` |
| Projective recurrence, orthogonality, determinant, relative offsets | `references.py`, `positions.json`, `V2Positions.lean` |
| Conditional local rank and grid injectivity | symbolic determinant, every grid offset for G=14/16/32/64, exact conditional Lean determinant; no arbitrary-frequency/global claim |
| Pool fixed points, timelike lower bound, outlier derivative | `symbolic.py`, `precision.json`, `V2Centroid.lean`, regression tests including N=1 counterexample |
| Head integrability, propriety and zero-transcendental conflict | symbolic Hessian/curl, 50-digit finite example, actual CE HLO negative control, power/Brier alternative gradients |
| 12/24-layer F(q), G(c), spectral gaps | `propagation.json`, matched-depth 64D-head surrogate; not full-sized ViTs |
| Backward Jacobian spectrum, Lyapunov exponents | full per-layer and composed spectra of declared 64D state at layers 12/24 |
| Figure 5 entropy, saturation, Q gradient across alpha=1..256 | `stress.json`, `figures/figure5_score_runaway.pdf`; score and input-coordinate derivatives distinguished |
| N=197/256, foreground k=1/3/5, four attention conditions | `crowding.json`, paired query-demand scenarios and explicit gradient SNR |
| High-norm scratchpad outliers 5..100 sqrt(d) | `outliers.json`, actual LN-before-GAP baseline plus separate raw GAP diagnostic |
| FP32/BF16 forward TV and backward error through alpha=100 | `precision.json`, own FP64 or 50-digit references, boundary u/kappa/pool regimes |
| Micro-convergence, learning rates 1e-4..1e-2, NTK, MSLR | `convergence.json`, 2/4 layers, full loss traces, held-out accuracy, rank and spectra; endpoint censoring explicit |
| Exact versus quarter exponent and tau=2/4 | symbolic curl, head-only convergence records; diagnostics do not certify purity of a CE forward |
| rgelu versus coupled gate | matched propagation and synthetic convergence ablations; no claim of selecting a full ViT-L gate |
| Lean build, zero sorry axioms, existing v1 proofs preserved | `verify.py`, all-theorem axiom audit, `validation.json`, `verification.log` |
| Self-correction and quantitative telemetry | original, sharp/balanced, rational-tail and inverse-energy variants, `gate.json`, report iteration record |
| Theoretical work and rsqrt counts | `purity.json`, leading matmul FLOPs and scalar root counts, CPU HLO of primitive and synthetic update graphs; no TPU speed claim |
| Baselines frozen; stop at Phase 1 | validation compares all existing production, experiment, script, dependency, test and v1 proof files with the original Git commit |

Every data file records source hashes. Report generation checks its actual
algorithm dependencies against those hashes and refuses a favorable decision
with missing or stale evidence. `manifest.json` hashes the generated artifacts.
`verify.py` rebuilds the Lean library and audits all locally declared theorems
and lemmas, then runs both original and research tests and regenerates the
report. The report generator can be rerun without simulations:

```bash
.venv/bin/python -m analysis.phase1.report
```

The baseline numerical reference implements the same v1 real-valued kernel
with row scaling before squaring to make its FP64 oracle robust to overflow.
Regression tests compare its forward and backward behavior to the frozen
production primitives on their finite range. It is not silently installed
into the baseline training arm.

The original exact-MLE forward, unconditional positional claim and universal
dominance request cannot be assumed as facts. Counterexamples are preserved,
and a green test run means the investigation is reproducible, not that v2 is
approved for the next phase.
