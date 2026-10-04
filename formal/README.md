# Lean proofs

Machine-checked proofs of the properties the paper relies on, in Lean 4 with
Mathlib. The toolchain is pinned in `lean-toolchain` and Mathlib in
`lake-manifest.json`.

```bash
cd formal
lake exe cache get   # prebuilt Mathlib
lake build
```

Every theorem below depends only on Lean's standard axioms (`propext`,
`Classical.choice`, `Quot.sound`); there is no `sorry`.

| Property | Theorems | Module |
| --- | --- | --- |
| The kernel base `rho x = x + sqrt(1 + x^2)` is positive and increasing; so is `E_n = rho^n` | `rho_pos`, `rho_mono`, `kernel_pos`, `kernel_mono` | `RadicalKernel` |
| Reciprocal symmetry `rho x * rho (-x) = 1` | `rho_mul_rho_neg` | `RadicalKernel` |
| Ratio bound `E(x) / E(y) <= (x / y)^n` for `0 < y <= x` | `rho_ratio_le`, `kernel_ratio_le` | `RadicalKernel` |
| The same bound after scaling both scores by any `c > 0` | `kernel_ratio_scale_free` | `RadicalKernel` |
| The exponential ratio `exp(cx) / exp(cy)` is unbounded in `c` | `exp_ratio_unbounded` | `RadicalKernel` |
| Attention floor: scores in `[a, b]`, `a > 0`, give every key at least `(a/b)^n / N` of the mass at every scale | `attention_floor` | `RadicalKernel` |
| With a sink `Omega >= 0` the visible mass is `1 - Omega / D` (below one if `Omega > 0`); weights are positive | `sink_mass`, `sink_mass_lt_one`, `weight_pos` | `Attention` |
| Score-Jacobian entries are bounded by `g / 4` | `jacobian_diag_le`, `jacobian_offdiag_le` | `Attention` |
| Accumulating over disjoint key tiles equals the dense result, with no running maximum | `tiled_eq_dense` | `Attention` |
| ALU gate: `abs u < 1`, reflection, derivative range `1/2 +- 2 sqrt(6)/9`, stationary point | `abs_unit_lt_one`, `gate_reflection`, `dalu_le`, `le_dalu`, `abs_dalu_le`, `dalu_factor` | `Activation` |
| Radical logistic: values in `(0, 1)`, `sigma(y) + sigma(-y) = 1`, slope at most `1/4` | `radicalSigmoid_mem`, `radicalSigmoid_symm`, `sigmoid_slope_le` | `Activation` |
| Cayley rotations are orthogonal with determinant one and commute | `cayley_orthogonal`, `cayley_det`, `cayley_commute` | `Cayley2D` |
| 2-D positions: rotated logits depend only on the offset between query and key (axial and learned mixed rotations) | `relative_logit`, `cayley2d_relative`, `mixed_cayley_relative` | `Cayley2D` |
| Power score: per-class Bregman term nonnegative, positive off the diagonal; the score is strictly proper | `bregman_nonneg`, `bregman_pos`, `excess_score_eq`, `power_score_proper`, `power_score_strictly_proper` | `PowerScore` |

Scope. The statements are about real numbers, not floating point. The radical
logistic is stated in terms of `e = E_n(y) > 0` (so `E_n(-y) = 1/e`).
Derivative formulas (the log-derivative of `E_n`, the attention Jacobian, the
gradient through the radical link) and asymptotic statements are checked
symbolically and at 50 digits by `analysis/symbolic_checks.py` instead.
Neither establishes model accuracy or speed.
