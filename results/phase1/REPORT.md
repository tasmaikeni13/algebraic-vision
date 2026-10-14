# Phase 1 analytical report

**Decision: NO-GO. The requested strict dominance gate is not satisfied.** This repository contains a completed set of Phase 1 investigations and executable counterexamples, not an approved production `algebraic_v2`. Tests passing do not turn failed mathematical or statistical gates into a pass. No Phase 2-5 implementation, TPU kernel engineering, sweeps, or main runs were performed.

## Evidence and scope

Coverage issues: **none**. Algorithm/source integrity issues: **none**. Verification matches current sources: **True**.

8 paired Monte Carlo seeds (42, 43, 44, 45, 46, 47, 48, 49) cover Gaussian and Student-t stress, N=197/256 crowding with k=1/3/5 foreground patches, d=768/1024 outlier pooling, and FP32/BF16 perturbations through scale 100. 3 paired seeds (42, 43, 44) cover 12/24-layer signal propagation and 2/4-layer, 200-step synthetic patch classification on five learning rates from 1e-4 to 1e-2. Every JSON records source hashes, versions, seeds, backend and elapsed time. Research runs use CPU and FP64 where stated; these are not TPU speed or full ViT-B/L accuracy measurements.

The exact spectral study uses the complete 64 by 64 Jacobian of a 4-token, 16-channel state; the separate forward propagation study uses 64 tokens and 64-dimensional heads. The layer counts match ViT-B/L. Width, training data and parameter counts do not. F(q) and G(c) are empirical trajectories, not an infinite-width proof. [Propagation data](propagation.json) includes every layer's singular spectrum, stack spectra and Lyapunov exponents at depths 12 and 24.

## Axis 1: exact algebra, derivatives and necessary corrections

All 34 symbolic and high-precision assertions passed. [Full symbolic record](symbolic.json).

* For kappa>|u|, the positive square root is (kappa²+u²)/(kappa²-u²), so the radical base equals (kappa+u)/(kappa-u). Raising to n proves the claimed identity. For n=8 the Taylor series is `1 + 16u/kappa + 128u²/kappa² + 688u³/kappa³ + 2816u⁴/kappa⁴ + O(u⁵)`. For |u|<=1 and kappa>1 the explicit bounds in the prompt hold. The log derivatives are `16*kappa/(kappa²-u²)` in u and `-16*u/(kappa²-u²)` in kappa. Query/key derivatives include the ball-lift differential `I/r - qqᵀ/r³` and the orthogonal relative position map; complete formulas are in the JSON. At 50 digits the identity's maximum relative residual is 1.6437502967067003223764299272981613554713460812646e-48.
* `1+1/rho(t)` ranges over (1,infinity), not (1.05,2]. The research correction is `1.05+0.95/(1+t²)`. A nonzero initial t is needed if this parameter is to learn immediately; its derivative vanishes at zero.
* Homogeneous recurrence has squared norm `(1+w²)^c`; it never reaches zero from (1,0). The polynomial identity proves orthogonality over the reals. Floating-point rotations are **not machine-exact**. Common binary rescaling preserves the represented projective point over the reals and prevents overflow for the tested bounded frequencies.
* Arbitrary SO(4) frequencies need not have rank two. The phase Jacobian has determinant `-4*(atan(wx1)*atan(wy2)+atan(wx2)*atan(wy1))`. Tying wx1=wx2=wx>0 and wy1=wy2=wy>0 makes it `-8*atan(wx)*atan(wy) != 0`. With wx<=1/(4G), wy<=1/(5G), all tested grid offsets stay within the non-wrapping region. Local rank alone is insufficient to rule out global periodic aliases. Also, independent planes in the frozen v1 model can jointly have rank two; a null-line in one plane does not imply a whole-head null-line.
* Lorentz pooling preserves identical tokens, and for nonempty finite tokens with d>0 its denominator obeys `D >= N²*d/(d+max ||h_i||²)>0`. The isolated-outlier derivative is O(1/||h_i||) when N>=2 and the other finite tokens are fixed. N=1 is a counterexample to the unqualified attenuation claim because pooling is exactly the identity.
* Cancellation correction: with b_i=h_i/sqrt(d+||h_i||²), the **same real-valued** denominator is `D = N*sum_i d/(d+||h_i||²) + N*sum_i ||b_i-mean(b)||²`. This O(Nd) form uses nonnegative summands. Tests cover its true VJP, not a clamped or surrogate backward pass.
* The exact CE gradient is conservative: off-diagonal Hessian entries are `-p_i*p_j*b_i*b_j`, where b_i=(n/tau)/sqrt(1+(z_i/tau)²). Replacing only the outside factor by the quarter power produces curl `p_i*p_j*(a_j*b_i-a_i*b_j)`, equal to 0.0077594620635399727138018619232534345929552371353331 in a concrete 50-digit example. Reject the quarter-power field as the gradient of the advertised objective, regardless of whether a toy optimizer converges.
* CE is strictly proper in p (KL excess risk); smoothed labels target the smoothed distribution. Its value contains log. A finite graph of arithmetic, square roots and piecewise comparisons is piecewise algebraic, so a custom arithmetic VJP cannot supply the exact non-algebraic CE value. [JAX's custom derivative contract](https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html) separates primal values from derivative rules. The HLO negative control detects the forward log.
* Research alternatives use the genuine algebraic Brier loss or the existing strictly proper power-score family with a=7/8, n=8, tau=4, requiring three nested roots rather than six. These are changed objectives, explicitly named and differentiated normally; they are not described as exact MLE.

## Axis 2: propagation, spectra and score runaway

The prompt's attenuation number is reproduced: g8(94)=0.084799830400508798304005935978630478354908967491372. This verifies the formula, not a new trained-model Figure 5 replication.

At alpha=100 on Gaussian Q,K, mean query VJP norms are v1=0.149374, proposed v2=0.0154562, sharpened v2=0.0559622, and rational-tail v2=0.149403. Proposed v2 retains 10.347% of v1's gradient norm despite high attention entropy. Bounding scores is not by itself gradient preservation. The lift's radial and tangential derivatives are d/(d+||q||²)^(3/2) and 1/sqrt(d+||q||²).

Both pre-projection Q derivatives and score-space derivatives are recorded. Score coordinates differ between kernels and cannot be compared as if they had the same units. Balanced attention couples rows; its complete VJP is measured, while inapplicable row-local Jacobian statistics are null. [Stress data](stress.json), [Figure 5 plot](figures/figure5_score_runaway.pdf), [propagation plot](figures/propagation.pdf).

The following means summarize the separate forward and exact-small-Jacobian studies. The singular values belong to the composed small state, not the 64-token forward study. Full per-layer spectra and Lyapunov exponents are retained in the data.

| Variant | Depth | Forward variance | Patch cosine | PF gap | Mean minimum stack SV | Mean maximum stack SV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 12 | 2.913 | 0.3275 | 0.8242 | 0.05158 | 16.39 |
| algebraic_v1 | 12 | 2.886 | 0.3258 | 0.8351 | 0.05335 | 16.12 |
| v2_proposed | 12 | 2.845 | 0.3379 | 0.93 | 0.05612 | 11.17 |
| v2_sharp | 12 | 2.969 | 0.3112 | 0.7583 | 0.02002 | 146 |
| v2_balanced | 12 | 2.94 | 0.3101 | 0.7869 | 0.02611 | 22.16 |
| v2_rational_tail | 12 | 2.886 | 0.3294 | 0.8469 | 0.05686 | 11.36 |
| standard | 24 | 5.547 | 0.5216 | 0.9013 | 0.04069 | 20.46 |
| algebraic_v1 | 24 | 5.529 | 0.5273 | 0.8873 | 0.04238 | 20.2 |
| v2_proposed | 24 | 5.466 | 0.5397 | 0.9526 | 0.04269 | 15.24 |
| v2_sharp | 24 | 5.589 | 0.4973 | 0.8464 | 0.01091 | 211.9 |
| v2_balanced | 24 | 5.543 | 0.5 | 0.8485 | 0.01769 | 39.19 |
| v2_rational_tail | 24 | 5.503 | 0.5249 | 0.9054 | 0.04316 | 15.92 |

## Axis 3: paired Monte Carlo, precision and convergence

Column balancing improves foreground recovery when most queries are background, but loses it when all queries want the same sparse foreground. For N=197,k=1, sharpening alone recovers about 0.9998 foreground mass; balancing falls to about 0.8303 in the all-signal-query scenario. Thus A3 is not selected as a general default: its extra N² reduction pass has no universal SNR benefit. [Crowding data and explicit gradient SNR definition](crowding.json).

The actual baseline applies final LayerNorm **per token before GAP**. At outlier norm 100*sqrt(d), averaged over all tested counts, sizes and seeds, raw GAP corruption is 66.929 degrees, actual baseline corruption is 1.946, proposed Lorentz pooling is 2.749, and inverse-energy pooling is 0.850. Comparing v2 only to raw GAP would have hidden its regression against v1. The second pooling correction is the rational centroid `sum_i a_i*h_i/sum_i a_i`, a_i=d/(d+||h_i||²). It preserves fixed points and has redescending isolated-outlier influence O(1/||h_i||), derivative O(1/||h_i||²); it changes the geometric pooling objective. [Outlier data](outliers.json), [robustness and precision plot](figures/robustness_precision.pdf).

Finite-precision tests compare each kernel with its own FP64 reference, not one kernel's answers with another's. Boundary u near +/-1 and kappa near 1.05 have a 50-digit mpmath probability and diagonal-derivative oracle. Native BF16 rational arithmetic and FP32 nonlinear accumulation are reported separately. Nearly collinear high-norm pooling is compared to an exact fixed point and a separate FP64 VJP. The naive denominator becomes nonfinite or grossly inaccurate at large scales; the algebraically identical positive-sum denominator remains finite through 1e6. [Precision data](precision.json). Relative errors near a zero reference gradient can be large even when absolute errors are small; they must not be interpreted as task accuracy.

The table reports the largest stable learning rate observed anywhere in the declared grid for each configuration and the best *mean* held-out accuracy across the grid; it does not certify stability at every seed or a larger LR. Full per-seed results, complete loss traces, empirical 32-output NTK eigenvalues, numerical ranks and positive-subspace condition numbers are in [convergence.json](convergence.json). The MSLR estimate is right-censored at 0.01 whenever that endpoint is stable. Loss values from different proper scores are not comparable. [Convergence plot](figures/micro_convergence.pdf).

| Depth | Synthetic configuration | Largest observed stable LR | Best mean held-out accuracy |
| --- | --- | ---: | ---: |
| 2 | algebraic_v1/rgelu/power/lorentz | 0.01 | 0.995 |
| 2 | standard/rgelu/power/lorentz | 0.01 | 0.995 |
| 2 | v2_proposed/rgelu/power/lorentz | 0.01 | 1.000 |
| 2 | v2_rational_tail/alu/power/inverse_energy | 0.01 | 1.000 |
| 2 | v2_sharp/alu/brier/inverse_energy | 0.01 | 0.990 |
| 2 | v2_sharp/alu/power/inverse_energy | 0.01 | 1.000 |
| 2 | v2_sharp/rgelu/power/lorentz | 0.01 | 1.000 |
| 4 | algebraic_v1/rgelu/power/lorentz | 0.01 | 1.000 |
| 4 | standard/rgelu/power/lorentz | 0.01 | 1.000 |
| 4 | v2_proposed/rgelu/power/lorentz | 0.01 | 0.995 |
| 4 | v2_rational_tail/alu/power/inverse_energy | 0.01 | 1.000 |
| 4 | v2_sharp/alu/brier/inverse_energy | 0.01 | 1.000 |
| 4 | v2_sharp/alu/power/inverse_energy | 0.01 | 1.000 |
| 4 | v2_sharp/rgelu/power/lorentz | 0.01 | 1.000 |

Both rgelu and the coupled algebraic ALU gate are evaluated. These small prototypes do not select a gate for a 304M-parameter ViT-L. Head-only diagnostics compare tau=2/4 and both exponents over the same LR grid; optimization observations do not override the curl counterexample.

## Axis 4: Lean and arithmetic verification

Formal build verified: **True**. Tests and frozen-baseline check verified: **True**. See [validation.json](validation.json) and [verification log](verification.log). The pinned Lean/Mathlib versions are unchanged. V1 modules are unchanged; new V2 modules prove rational equivalence, positivity, explicit bounds/floor, projective orthogonality/determinant and power recurrence, the actual paired SO(4) relative-offset identity, the conditional rank determinant, and the vector-norm centroid denominator bound. Axiom auditing covers every locally declared theorem and lemma, including v1. Only Lean's standard axioms are allowed; no `sorryAx` is accepted. See [Lean instructions](../../formal/README.md).

[Primitive HLO audit](purity.json) covers forward and backward research primitives and two-layer synthetic loss/gradient graphs on CPU, with a transcendental negative control. This is intentionally not the Phase 2 full-size TPU model audit. Per head at N=197 the original attention requires 38,809 scalar root evaluations versus 394 for the ball lift (591 with column balancing), or zero for the rational-tail correction. All retain the shared 4*N²*d_h leading QK/PV matmul FLOPs. Projective tables remove recurrence roots; the decoupled power head halves the sequential root depth. Division, reduction, MXU scheduling and compiler fusion still matter: these counts do not prove a TPU latency improvement or total training-step FLOP dominance.

The existing v1 head already avoids per-class nested roots: it roots the **row normalizer** six times. For K classes its forward loss therefore uses K+6 scalar root elements, versus K+3 for the corrected power head. A claim of removing three roots from every class would overstate the actual saving. The formulas in [gate.json](gate.json) account for this implementation detail, forward LayerNorm/MLP/pooling roots and shared positional tables.

The table gives batch-one model/loss forward root elements and the leading **block matmul** forward/backward FLOPs, derived at the requested B/L shapes without implementing production models. Shared leading matmuls are `3*L*(24*N*D²+4*N²*D)`; the factor three counts the forward and two linear backward products. Optimizer, embedding/head matmuls, rematerialization and remaining elementwise work are excluded and explicitly prevent a total-step FLOP certificate. In particular, the full model's MLP roots remain even when attention becomes root-free.

| Shape | N | Shared block train matmul FLOPs | v1 forward roots | Projected v2 forward roots | Tail + inverse-energy forward roots |
| --- | ---: | ---: | ---: | ---: | ---: |
| B/16 | 197 | 104,684,728,320 | 12,985,659 | 7,324,874 | 7,267,940 |
| B/16 | 256 | 137,707,388,928 | 19,029,230 | 9,518,317 | 9,444,332 |
| L/16 | 197 | 368,397,287,424 | 34,623,267 | 19,527,842 | 19,376,348 |
| L/16 | 256 | 483,183,820,800 | 50,738,414 | 25,375,981 | 25,179,116 |

## Axis 5: self-correction record and strict gate

Iteration 0 verifies the proposed formula and rejects the advertised margin range, unconditional positional injectivity, machine-exact floating-point orthogonality, unqualified outlier attenuation, quarter-power conservative gradient, and pure exact CE forward claims.

Iteration 1 tests kappa=1.1 and gain=2, tied grid-safe frequencies, the positive-sum Lorentz denominator and genuine algebraic proper losses. Foreground sharpness and numerical stability improve, but pre-projection gradient preservation and actual-baseline pooling comparisons still fail.

Iteration 2 tests a root-free unbounded rational-tail score link: for t=|s|/8, f(t)=1+t+t²/(2+t); use f on positive scores and 1/f on negative scores, followed by three squarings and normalization. It matches E8 through second order at zero and avoids the lift's radial saturation. With inverse-energy pooling and the ALU/power-head ablations, it improves the random high-scale gradient study and outlier study. It still regresses in individual finite-precision cells and an explicit radial query case. All affected symbolic, numerical, prototype, purity and formal checks are rerun; no failed metric is suppressed.

An implementation regression test also exposed the absolute-value subgradient at exactly s=0: writing the positive branch entirely through abs(s) incorrectly returned zero slope. The final implementation keeps its linear positive term signed, while both evaluated branches remain finite. The analytic derivative test includes exactly zero, and all numerical studies were rerun after this correction. No surrogate derivative hides the defect.

Ratios below one are bad for gradients; ratios above one are bad for errors/corruption. These are sample gates, strictly compared without a post-hoc tolerance. They do not constitute a population dominance theorem.

| Candidate | Minimum random paired gradient ratio, alpha 10-100 | Minimum radial ratio, alpha 10-100 | Worst FP error ratio of means | Worst pool corruption ratio of means | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| v2_proposed | 0.08849 | 0.18121201493220615 | 0.8437 | 1.442 | NO-GO |
| v2_sharp | 0.3311 | 0.011422553279320374 | 1.439 | 1.442 | NO-GO |
| v2_balanced | 0.3031 | not measured | 1.325 | 1.442 | NO-GO |
| v2_rational_tail | 1 | 0.7149548472662943 | 1.092 | 0.7249 | NO-GO |

The self-correction loop has produced improved candidates and concrete remaining counterexamples. The directive's demand to keep iterating *until provable domination* has **not been fulfilled**. A finite Monte Carlo study cannot turn a false unconditional claim into a theorem, and this report does not assert a successful phase gate. The honest result is NO-GO, with all failures retained and no later phase started. Implementing a production candidate requires a mathematically consistent objective and an explicit, defensible statistical acceptance criterion.

## Reproduction and artifact integrity

From the repository root, create an environment using `requirements-phase1.txt`, then run `python -m analysis.phase1.run` and `python -m analysis.phase1.verify`. Run `python -m analysis.phase1.run --require-pass` to receive exit status 2 when the research gate is not satisfied. Results are deterministic for the recorded seeds on the recorded CPU/JAX versions; execution times and plots' PDF metadata may differ. Full references and a requirement/evidence checklist are in [the research README](../../analysis/phase1/README.md). [manifest.json](manifest.json) hashes generated artifacts; each data file independently hashes its research dependencies.
