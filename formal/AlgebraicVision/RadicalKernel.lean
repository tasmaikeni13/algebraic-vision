import Mathlib

/-!
# The radical kernel

`rho x = x + sqrt (1 + x ^ 2)` is the base of the attention kernel
`E_n(s) = rho (s / n) ^ n`. We prove that it is positive and increasing, that
`rho x * rho (-x) = 1`, and the scale-free ratio bound
`y * rho x ≤ x * rho y` for `0 < y ≤ x`, which gives
`E(x) / E(y) ≤ (x / y) ^ n` at every score scale. The exponential has no such
bound: `exp (c x) / exp (c y)` grows without limit in `c`. As a consequence,
when all scores lie in `[a, b]` with `0 < a`, every key keeps at least a
fraction `(a / b) ^ n / N` of the kernel mass at every scale.
-/

namespace AlgebraicVision

open Real Finset

/-- Radical base of the attention kernel. -/
noncomputable def rho (x : ℝ) : ℝ := x + sqrt (1 + x ^ 2)

lemma abs_lt_sqrt_one_add_sq (x : ℝ) : |x| < sqrt (1 + x ^ 2) := by
  rw [← sqrt_sq_eq_abs]
  exact sqrt_lt_sqrt (sq_nonneg x) (by linarith)

theorem rho_pos (x : ℝ) : 0 < rho x := by
  have h1 := abs_lt_sqrt_one_add_sq x
  have h2 := neg_abs_le x
  unfold rho
  linarith

/-- Reciprocal symmetry, the analogue of `exp x * exp (-x) = 1`. -/
theorem rho_mul_rho_neg (x : ℝ) : rho x * rho (-x) = 1 := by
  have h : sqrt (1 + x ^ 2) ^ 2 = 1 + x ^ 2 := sq_sqrt (by positivity)
  unfold rho
  rw [neg_sq]
  nlinarith [h]

theorem kernel_pos (n : ℕ) (x : ℝ) : 0 < rho x ^ n := pow_pos (rho_pos x) n

/-- Core inequality: for `0 < y ≤ x`, `y * rho x ≤ x * rho y`. -/
theorem rho_ratio_le {x y : ℝ} (hy : 0 < y) (hxy : y ≤ x) :
    y * rho x ≤ x * rho y := by
  have hx : 0 < x := lt_of_lt_of_le hy hxy
  have h1 : y * sqrt (1 + x ^ 2) = sqrt (y ^ 2 * (1 + x ^ 2)) := by
    rw [sqrt_mul (sq_nonneg y), sqrt_sq hy.le]
  have h2 : x * sqrt (1 + y ^ 2) = sqrt (x ^ 2 * (1 + y ^ 2)) := by
    rw [sqrt_mul (sq_nonneg x), sqrt_sq hx.le]
  have hsq : y ^ 2 ≤ x ^ 2 := pow_le_pow_left₀ hy.le hxy 2
  have key : y * sqrt (1 + x ^ 2) ≤ x * sqrt (1 + y ^ 2) := by
    rw [h1, h2]
    apply sqrt_le_sqrt
    nlinarith [hsq]
  unfold rho
  nlinarith [key]

/-- Kernel ratio bound: `y^n * E(x) ≤ x^n * E(y)` for `0 < y ≤ x`. -/
theorem kernel_ratio_le (n : ℕ) {x y : ℝ} (hy : 0 < y) (hxy : y ≤ x) :
    y ^ n * rho x ^ n ≤ x ^ n * rho y ^ n := by
  rw [← mul_pow, ← mul_pow]
  exact pow_le_pow_left₀ (mul_pos hy (rho_pos x)).le (rho_ratio_le hy hxy) n

/-- Scale-free form: multiplying both scores by any `c > 0` keeps the ratio of
kernel values below `(x / y) ^ n`. This is the property that prevents the
attention distribution from collapsing as scores grow. -/
theorem kernel_ratio_scale_free (n : ℕ) {c x y : ℝ} (hc : 0 < c) (hy : 0 < y)
    (hxy : y ≤ x) : y ^ n * rho (c * x) ^ n ≤ x ^ n * rho (c * y) ^ n := by
  have h := kernel_ratio_le n (mul_pos hc hy) (mul_le_mul_of_nonneg_left hxy hc.le)
  rw [mul_pow, mul_pow, mul_assoc, mul_assoc] at h
  exact le_of_mul_le_mul_left h (pow_pos hc n)

/-- In contrast, the exponential ratio is unbounded in the scale. -/
theorem exp_ratio_unbounded {x y : ℝ} (hxy : y < x) (M : ℝ) :
    ∃ c > 0, M < exp (c * x) / exp (c * y) := by
  refine ⟨(max M 1) / (x - y), div_pos (lt_max_of_lt_right one_pos)
    (sub_pos.mpr hxy), ?_⟩
  rw [← exp_sub, ← mul_sub, div_mul_cancel₀ _ (sub_pos.mpr hxy).ne']
  calc M ≤ max M 1 := le_max_left _ _
    _ < max M 1 + 1 := by linarith
    _ ≤ exp (max M 1) := add_one_le_exp _

/-- `rho` is monotone: `rho y - rho x = (y - x) (1 + (x + y) / S)` with
`S = sqrt(1 + x^2) + sqrt(1 + y^2) > |x + y|`. -/
theorem rho_mono {x y : ℝ} (hxy : x ≤ y) : rho x ≤ rho y := by
  have hx := abs_lt_sqrt_one_add_sq x
  have hy := abs_lt_sqrt_one_add_sq y
  have sx : sqrt (1 + x ^ 2) ^ 2 = 1 + x ^ 2 := sq_sqrt (by positivity)
  have sy : sqrt (1 + y ^ 2) ^ 2 = 1 + y ^ 2 := sq_sqrt (by positivity)
  have ax := abs_nonneg x
  have ay := abs_nonneg y
  have lx := neg_abs_le x
  have ux := le_abs_self x
  have ly := neg_abs_le y
  have uy := le_abs_self y
  unfold rho
  -- (sqrt(1+y^2) - sqrt(1+x^2)) * S = y^2 - x^2 and S > -(x + y)
  nlinarith [mul_nonneg (sub_nonneg.mpr hxy)
    (show 0 ≤ sqrt (1 + x ^ 2) + sqrt (1 + y ^ 2) + (x + y) by linarith),
    sq_nonneg (sqrt (1 + y ^ 2) - sqrt (1 + x ^ 2))]

theorem kernel_mono (n : ℕ) {x y : ℝ} (hxy : x ≤ y) : rho x ^ n ≤ rho y ^ n :=
  pow_le_pow_left₀ (rho_pos x).le (rho_mono hxy) n

/-- Attention floor at every scale. If all scores lie in `[a, b]` with
`0 < a`, then for every scale `c > 0` each kernel value is at least
`(a / b)^n / N` of the total, `N` the number of keys:
`a^n * sum_k E(c x_k) <= N * b^n * E(c x_j)`. Softmax has no such floor. -/
theorem attention_floor {ι : Type*} (s : Finset ι) (x : ι → ℝ) (n : ℕ)
    {a b c : ℝ} (hc : 0 < c) (ha : 0 < a)
    (hx : ∀ i ∈ s, a ≤ x i ∧ x i ≤ b) {j : ι} (hj : j ∈ s) :
    a ^ n * ∑ k ∈ s, rho (c * x k) ^ n ≤
      (s.card : ℝ) * b ^ n * rho (c * x j) ^ n := by
  have hxj := hx j hj
  have hpos : 0 < x j := lt_of_lt_of_le ha hxj.1
  have term : ∀ k ∈ s, a ^ n * rho (c * x k) ^ n ≤ b ^ n * rho (c * x j) ^ n := by
    intro k hk
    have hxk := hx k hk
    rcases le_total (x k) (x j) with h | h
    · -- smaller score: monotonicity and a <= b
      have h1 : rho (c * x k) ^ n ≤ rho (c * x j) ^ n :=
        kernel_mono n (mul_le_mul_of_nonneg_left h hc.le)
      have hab : a ≤ b := le_trans hxj.1 hxj.2
      have h2 : a ^ n ≤ b ^ n := pow_le_pow_left₀ ha.le hab n
      have h3 : 0 ≤ rho (c * x k) ^ n := (kernel_pos n _).le
      calc a ^ n * rho (c * x k) ^ n ≤ b ^ n * rho (c * x k) ^ n :=
            mul_le_mul_of_nonneg_right h2 h3
        _ ≤ b ^ n * rho (c * x j) ^ n :=
            mul_le_mul_of_nonneg_left h1 (pow_nonneg (ha.le.trans hab) n)
    · -- larger score: the scale-free ratio bound
      have hr := kernel_ratio_scale_free n hc hpos h
      have hxkb : x k ^ n ≤ b ^ n := pow_le_pow_left₀ (hpos.le.trans h) hxk.2 n
      have hxja : a ^ n ≤ x j ^ n := pow_le_pow_left₀ ha.le hxj.1 n
      have e0 : 0 ≤ rho (c * x k) ^ n := (kernel_pos n _).le
      have e1 : 0 ≤ rho (c * x j) ^ n := (kernel_pos n _).le
      calc a ^ n * rho (c * x k) ^ n ≤ x j ^ n * rho (c * x k) ^ n :=
            mul_le_mul_of_nonneg_right hxja e0
        _ ≤ x k ^ n * rho (c * x j) ^ n := hr
        _ ≤ b ^ n * rho (c * x j) ^ n := mul_le_mul_of_nonneg_right hxkb e1
  calc a ^ n * ∑ k ∈ s, rho (c * x k) ^ n
      = ∑ k ∈ s, a ^ n * rho (c * x k) ^ n := by rw [mul_sum]
    _ ≤ ∑ k ∈ s, b ^ n * rho (c * x j) ^ n := sum_le_sum term
    _ = (s.card : ℝ) * b ^ n * rho (c * x j) ^ n := by
        rw [sum_const, nsmul_eq_mul, mul_assoc]

end AlgebraicVision
