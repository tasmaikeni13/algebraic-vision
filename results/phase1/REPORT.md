# Phase 1 analytical report

**Decision: GO for the refined `v2_retained_e8` research candidate on the declared Phase 1 checks.** The initial proposal and root-free attention candidates remain rejected. The final revision retains E8 attention, which proves an exact attention-gradient and precision match to v1; it does **not** eliminate E8 score attenuation or quadratic attention roots. All later phases remain unstarted.

This is a scoped research gate, not a theorem that one complete model dominates another on every distribution. `dominance_proven=false` in [gate.json](gate.json) explicitly records that limit. No full-width training, ImageNet accuracy or TPU latency result is implied. User authorization allowed choosing an appropriate numerical equivalence criterion; the selected attention ultimately needs **zero tolerance**.

## Selected equations and design decision

The accessible derivation is in [theory.md](../../theory.md). The frozen `standard` and `algebraic` implementations, experiments, original tests and original proofs are unchanged.

* **Attention:** keep exactly the existing normalized E8 score link, no Klein lift or column balancing. The same expression, precision policy and real-valued derivatives make its preservation and perturbation behavior identical to v1 in the same Q/K coordinates.
* **Positions:** two SO(2) planes in SO(4), with phases `c*theta_x+r*theta_y` and `c*theta_x-r*theta_y`. Learn each frequency through `w=(1/G)*(1/4+3/(4*(1+raw^2)))`. Positive bounded frequencies maintain rank two and prevent phase wrapping on the grid. Homogeneous projective recurrence removes table-renormalization roots.
* **Pool:** `LN(sum a_i*h_i/sum a_i)`, with `a_i=d/(d+||h_i||^2)`, applied to the raw residual tokens before the final LN. This replaces the Lorentz proposal with a redescending rational centroid.
* **MLP:** `f(x)=x/2*(1+c*x/(1+c*|x|))`, c=0.851. This C2 rational gate has derivative strictly between zero and one, matches rgelu's value/slope/curvature at zero, and uses no roots. Its exact analytic JVP is checked against ordinary differentiation; it is not a surrogate gradient.
* **Head:** genuine strictly proper power score a=7/8 on `rho(z/4)^8`, with three sequential roots of the row normalizer. Exact CE's forward logarithm and the non-conservative quarter-power field are rejected. Raw loss values from different proper scores are not directly comparable.

## Checks and coverage

| Requirement | Result |
| --- | --- |
| High-scale attention gradients | PASS |
| FP32/BF16 attention stability and finite pool boundaries | PASS |
| Rank and grid positional injectivity | PASS |
| Outlier angular robustness | PASS |
| Per-step theoretical work and roots | PASS |
| Algebraic training graph | PASS |
| Synthetic stable learning-rate endpoint | PASS |

Evidence integrity issues: **none**. Coverage issues: **none**. Verification current: **True**. Formal build/axioms verified: **True**. Tests and frozen baselines verified: **True**.

The original nine-section experiment matrix is preserved in its JSON files and [the initial iteration report](iterations_0_2.md). The final revision additionally runs both stress distributions at eight paired seeds 42–49, all 11 scales 1–256, radial counterexamples through 1024, and FP32/BF16 errors at N=32/197/256 and scales 1/3/10/30/100. [revision_attention.json](revision_attention.json) records every result: minimum paired query-gradient ratio **1.0**, maximum forward difference **0.0**, and maximum backward difference **0.0**. Matching a kernel to itself is deliberate retention of a sound component, not evidence that the failed new kernels improved.

Crowding comparisons at N=197/256 and foreground counts 1/3/5 include all original balanced and unbalanced candidates. The selected attention inherits exactly v1's foreground mass and gradient SNR for identical Q/K; unchanged evidence is reused explicitly rather than counted as new independent samples. Original u/kappa/light-cone boundary cases and 50-digit references remain in [precision.json](precision.json). The selected pool is separately finite, in value and VJP, through scale 1e6 in both dtypes: [revision_pool_precision.json](revision_pool_precision.json).

Learned positions are checked at G=14/16/32/64, initialization, equal and mixed frequency extremes, and FP32/BF16 output boundaries. Every rounded grid matrix is distinct in each declared test. The exact argument holds for all finite raw parameters: positive angles give nonzero rank; each relative phase has magnitude below 4<2*pi; both phases zero force both offsets zero. Rounded rotations are not machine-exact orthogonal matrices. [Position data](revision_positions.json).

The selected inverse-energy pool is the identical previously tested `v2_inverse_energy` component. On every N/count/scale cell, its mean angular corruption is no larger than actual v1 **LN-before-GAP**, with worst ratio **0.724912**. All paired samples remain in [outliers.json](outliers.json); raw GAP remains a separate diagnostic. Positive weights and fixed points are formalized. Isolated-outlier influence is O(1/||h||), derivative O(1/||h||^2), with other tokens fixed. This does not guarantee smaller error on every arbitrary token collection.

## Deep propagation, convergence and NTK

[revision_propagation.json](revision_propagation.json) records F(q), G(c), PF gaps and spectra at every layer for standard, v1 and the selected revision, at three paired seeds. Depths are 12 and 24. Forward heads are 64D over 64 tokens; exact backward singular spectra use the complete 64D state of four 16D tokens, with composed spectra and Lyapunov exponents at 12/24. The synthetic widths are explicit: these are neither full-width ViT-B/L Jacobians nor an infinite-width mean-field theorem. [Propagation figure](figures/revision_propagation.pdf).

The complete 2/4-layer, width-32 classification study uses three paired seeds, five LRs from 1e-4 to 1e-2, 200 SGD-momentum steps per run, 64 training and 64 held-out examples. Every candidate run is finite. All six paired seed/depth comparisons retain v1's stable endpoint of 0.01, so MSLR remains right-censored there. Complete loss and task-accuracy traces, held-out results, and empirical 32-output NTK spectra/conditioning are in [revision_convergence.json](revision_convergence.json).

| Layers | Arm | Stable at LR=.01 | Mean step reaching 95% train accuracy | Mean held-out accuracy at .01 |
| --- | --- | ---: | ---: | ---: |
| 2 | standard | 3/3 | 23.00 | 0.9896 |
| 2 | algebraic_v1 | 3/3 | 23.00 | 0.9896 |
| 2 | v2_retained_e8 | 3/3 | 6.33 | 1.0000 |
| 4 | standard | 3/3 | 18.67 | 0.9948 |
| 4 | algebraic_v1 | 3/3 | 18.00 | 0.9948 |
| 4 | v2_retained_e8 | 3/3 | 6.00 | 0.9948 |

Task-accuracy convergence is measured directly, separately from each objective's loss. Head link, loss scale, gate, pool and positions change together, so these runs do not isolate one cause of any speed difference. No larger MSLR or universal accuracy improvement is asserted. [Accuracy traces](figures/revision_convergence.pdf). Earlier rgelu/ALU/Brier and exact/quarter-gradient diagnostics remain in the original studies.

## Theoretical per-step operation comparison

[costs.json](costs.json) contains forward plus complete VJP counts at B/L widths, depths and N=197/256. Matrix FLOPs use 2*M*N*K; scalar arithmetic and division count one each. Roots are counted separately. The conservative table below also charges every min/max, comparison and select as one unit, and every fixed scan iteration is included. It does not hide these operations in the FLOP saving.

Common embedding, QKVO/MLP/classifier linear products, residual operations and internal norms cancel. The changed pool's norm, head loss, all position-frequency gradients and full attention/PV backward are included. The selected position parameters decrease by D/2 per layer, so an identical elementwise SGD/Adam-style optimizer costs no more. The proof assumes the same storage/recomputation convention, no extra rematerialization, and dense shapes. CPU lowered/compiled estimates are supplied separately and also favor this selection; they are not TPU timing predictions. See [JAX's AOT documentation](https://docs.jax.dev/en/latest/aot.html) for compiler-specific cost analysis.

For each shape the source-program count is affine in batch size B>=1: pointwise counts are linear, batch reductions cost B-1 per fixed group, and no operation multiplies two independent batch dimensions. The count difference is `delta(1)+(B-1)*slope`. Both terms are nonpositive, with a separate B=3 regression crosscheck; the general affine inequality is also proved in Lean.

| Shape | N | Work difference at B=1 | Work difference per extra example | Root difference at B=1 |
| --- | ---: | ---: | ---: | ---: |
| B/16 | 197 | -104,212,882 | -99,701,062 | -14,653,639 |
| B/16 | 256 | -134,671,297 | -129,561,493 | -19,022,082 |
| L/16 | 197 | -275,723,486 | -263,649,350 | -39,076,039 |
| L/16 | 256 | -356,294,637 | -342,612,885 | -50,725,122 |

These are **v2 minus v1**, so negative means less work. Attention roots remain; the large saving comes from replacing rgelu, removing positional-table roots, pooling once after the centroid and reducing head root depth. The static source count is an audited operation convention, not a formally verified compiler or hardware cycle model.

## Formal verification and self-correction record

All original v1 proofs remain included. New Lean modules prove rational equivalence/bounds, projective orthogonality/determinant/powers, paired SO(4) relative shifts, conditional rank, the Lorentz denominator bound, inverse-energy positivity/fixed points, learned-frequency bounds/rank, paired-phase injectivity, rational gate slope bounds and the affine cost inequality. Gate derivative identities and C2 matching are additionally checked by SymPy, 50-digit oracles and numerical JVP tests. Every local theorem/lemma is axiom-audited; only standard Lean axioms are permitted. [Formal inventory](../../formal/README.md), [validation](validation.json), [verification log](verification.log).

The initial iterations corrected the advertised margin range, arbitrary-frequency injectivity, floating-point exactness, unqualified N=1 outlier attenuation, the quarter-gradient integrability claim and the pure exact-MLE forward claim. Sharpening and column balancing did not cure all gradient/precision failures. Rational-tail attention improved several metrics but retained concrete regressions. Details and every original failed cell remain in [iterations_0_2.md](iterations_0_2.md).

Additional screens cover 75 higher-order projected candidates, 225 with column-energy priors, and 96 rational/Padé candidates, using separate selection seeds 0/1 and precision seeds 1000/1001. Exact scripts, raw ratios and limits of their exploratory provenance are archived in [the exploration directory](../../analysis/phase1/exploration/README.md). Their later optimized analytic differentials were tested independently. Padé/Newton links approach E8 closely and save roots, but their audited direct arithmetic cost can regress; compiler-specific estimates alone were not used to override that finding.

The final self-correction is conservative retention: keep the attention component that already meets the gradient/precision checks, and select independently justified changes elsewhere. This achieves the declared five-axis Phase 1 acceptance without claiming a successful root-free attention replacement, curing all middle-block attenuation, or obtaining universal model dominance.

## Reproduce

Install [requirements-phase1.txt](../../requirements-phase1.txt), then run `python -m analysis.phase1.run` followed by `python -m analysis.phase1.verify`. The full runner includes the original studies, final revision and operation counts. Use `python -m analysis.phase1.run --report-only --require-pass` to check cached, source-validated evidence; missing/stale evidence or a failed gate exits 2. Individual final studies are available through `python -m analysis.phase1.revision_study --sections ...`; cost analysis through `python -m analysis.phase1.costs`. CPU is selected explicitly, so none initializes pod training. [manifest.json](manifest.json) hashes artifacts. Reduced matrices cannot pass the coverage check.
