import AlgebraicVision.RadicalKernel

/-!
# Algebraic gates

ALU gate. With `u = y / sqrt (1 + y ^ 2)` the gate is `F = (1 + u) / 2` and
the derivative of `ALU_c(x) = x F(c x)` is the cubic `h u = 1/2 + u - u^3/2`.
We prove `|u| < 1`, the reflection `F(y) + F(-y) = 1`, the exact range
`[1/2 - 2r/3, 1/2 + 2r/3]` of `h` on `[-1, 1]` with `r = sqrt (2/3)`
(i.e. `1/2 ± 2 sqrt 6 / 9`), and the factorisation that locates the
stationary point at `u = (1 - sqrt 5) / 2`.

Radical logistic. `sigma_n(y) = 1 / (1 + E_n(-y))` is the gate of the
model's MLP (`rgelu(x) = x sigma_8(1.702 x)`). We prove that it lies in
`(0, 1)`, that `sigma_n(y) + sigma_n(-y) = 1`, and that its slope
`sigma (1 - sigma) g` is at most `1/4`.
-/

namespace AlgebraicVision

open Real

/-- Normalised coordinate `u = y / sqrt (1 + y ^ 2)`. -/
noncomputable def unit (y : ℝ) : ℝ := y / sqrt (1 + y ^ 2)

theorem abs_unit_lt_one (y : ℝ) : |unit y| < 1 := by
  have hs : 0 < sqrt (1 + y ^ 2) := sqrt_pos.mpr (by positivity)
  unfold unit
  rw [abs_div, abs_of_pos hs, div_lt_one hs]
  exact abs_lt_sqrt_one_add_sq y

theorem unit_neg (y : ℝ) : unit (-y) = -unit y := by
  unfold unit
  rw [neg_sq, neg_div]

/-- Gate reflection: `F(y) + F(-y) = 1`. -/
theorem gate_reflection (y : ℝ) :
    (1 + unit y) / 2 + (1 + unit (-y)) / 2 = 1 := by
  rw [unit_neg]
  ring

/-- Derivative of the ALU as a polynomial in the cached `u`. -/
noncomputable def dalu (u : ℝ) : ℝ := 1 / 2 + u - u ^ 3 / 2

/-- Upper bound, attained at `u = r = sqrt (2/3)`. -/
theorem dalu_le {u r : ℝ} (hr : r ^ 2 = 2 / 3) (hr0 : 0 < r) (hu : -1 ≤ u) :
    dalu u ≤ 1 / 2 + 2 * r / 3 := by
  have h2r : 1 ≤ 2 * r := by nlinarith
  have key : dalu u - (1 / 2 + 2 * r / 3) = -(1 / 2) * (u - r) ^ 2 * (u + 2 * r) := by
    unfold dalu
    have : r ^ 3 = 2 / 3 * r := by rw [pow_succ, hr]
    nlinarith [this]
  nlinarith [mul_nonneg (sq_nonneg (u - r)) (show 0 ≤ u + 2 * r by linarith)]

/-- Lower bound, attained at `u = -r`. -/
theorem le_dalu {u r : ℝ} (hr : r ^ 2 = 2 / 3) (hr0 : 0 < r) (hu : u ≤ 1) :
    1 / 2 - 2 * r / 3 ≤ dalu u := by
  have h2r : 1 ≤ 2 * r := by nlinarith
  have key : dalu u - (1 / 2 - 2 * r / 3) = (1 / 2) * (u + r) ^ 2 * (2 * r - u) := by
    unfold dalu
    have : r ^ 3 = 2 / 3 * r := by rw [pow_succ, hr]
    nlinarith [this]
  nlinarith [mul_nonneg (sq_nonneg (u + r)) (show 0 ≤ 2 * r - u by linarith)]

/-- The ALU is `(1/2 + 2 sqrt 6 / 9)`-Lipschitz: `|h u| ≤ 1/2 + 2 r / 3` on
`[-1, 1]`. Numerically this is `1.0443`, against `1.1289` for GELU. -/
theorem abs_dalu_le {u r : ℝ} (hr : r ^ 2 = 2 / 3) (hr0 : 0 < r)
    (hu0 : -1 ≤ u) (hu1 : u ≤ 1) : |dalu u| ≤ 1 / 2 + 2 * r / 3 := by
  rw [abs_le]
  constructor
  · have := le_dalu hr hr0 hu1
    linarith
  · exact dalu_le hr hr0 hu0

/-- Stationary points of the ALU: `h u = -(u + 1)(u^2 - u - 1) / 2`, so on
`(-1, 1)` the only one is `u = (1 - sqrt 5) / 2`. -/
theorem dalu_factor (u : ℝ) : dalu u = -((u + 1) * (u ^ 2 - u - 1)) / 2 := by
  unfold dalu
  ring

/-- The radical logistic `sigma (e) = 1 / (1 + 1/e)` written in terms of
`e = E_n(y) > 0` (so that `E_n(-y) = 1/e` by `rho_mul_rho_neg`). -/
noncomputable def radicalSigmoid (e : ℝ) : ℝ := 1 / (1 + 1 / e)

/-- Symmetry `sigma(y) + sigma(-y) = 1`: with `e = E_n(y)` the two values are
`1/(1 + 1/e)` and `1/(1 + e)`. -/
theorem radicalSigmoid_symm {e : ℝ} (he : 0 < e) :
    radicalSigmoid e + radicalSigmoid (1 / e) = 1 := by
  unfold radicalSigmoid
  have h1 : (1 : ℝ) + 1 / e ≠ 0 := by positivity
  have h2 : (1 : ℝ) + 1 / (1 / e) ≠ 0 := by positivity
  field_simp
  ring

theorem radicalSigmoid_mem {e : ℝ} (he : 0 < e) :
    0 < radicalSigmoid e ∧ radicalSigmoid e < 1 := by
  unfold radicalSigmoid
  have h : 0 < 1 / e := by positivity
  constructor
  · positivity
  · rw [div_lt_one (by linarith)]
    linarith

/-- The gate derivative `sigma (1 - sigma) g` is at most `1/4` for any slope
factor `0 ≤ g ≤ 1`. -/
theorem sigmoid_slope_le {s g : ℝ} (hs0 : 0 ≤ s) (hs1 : s ≤ 1) (hg0 : 0 ≤ g)
    (hg1 : g ≤ 1) : s * (1 - s) * g ≤ 1 / 4 := by
  have h1 : s * (1 - s) ≤ 1 / 4 := by nlinarith [sq_nonneg (s - 1 / 2)]
  have h2 : 0 ≤ s * (1 - s) := mul_nonneg hs0 (by linarith)
  nlinarith

end AlgebraicVision
