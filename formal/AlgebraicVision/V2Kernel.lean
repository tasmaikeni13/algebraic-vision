import AlgebraicVision.RadicalKernel

/-! Phase 1: rational kernel over ℝ. Every denominator/domain hypothesis is
explicit; these are exact-real proofs, not floating-point guarantees. -/

namespace AlgebraicVision.V2

lemma margin_pos {κ u : ℝ} (h : |u| < κ) : 0 < κ ^ 2 - u ^ 2 := by
  have hp : 0 < κ := lt_of_le_of_lt (abs_nonneg u) h
  have hl := neg_abs_le u
  have hr := le_abs_self u
  nlinarith [mul_pos (show 0 < κ - u by linarith) (show 0 < κ + u by linarith)]

theorem weierstrass_base {κ u : ℝ} (h : |u| < κ) :
    AlgebraicVision.rho (2 * κ * u / (κ ^ 2 - u ^ 2)) = (κ + u) / (κ - u) := by
  have hd := margin_pos h
  have hs : 0 ≤ (κ ^ 2 + u ^ 2) / (κ ^ 2 - u ^ 2) := by positivity
  have he : 1 + (2 * κ * u / (κ ^ 2 - u ^ 2)) ^ 2 =
      ((κ ^ 2 + u ^ 2) / (κ ^ 2 - u ^ 2)) ^ 2 := by
    field_simp
    ring
  have hm : κ - u ≠ 0 := by
    have := le_abs_self u
    linarith
  unfold AlgebraicVision.rho
  rw [he, Real.sqrt_sq hs]
  field_simp
  ring

theorem weierstrass_kernel {κ u : ℝ} (h : |u| < κ) (n : ℕ) :
    AlgebraicVision.rho (2 * κ * u / (κ ^ 2 - u ^ 2)) ^ n =
      ((κ + u) / (κ - u)) ^ n := by rw [weierstrass_base h]

theorem rational_positive {κ u : ℝ} (h : |u| < κ) (n : ℕ) :
    0 < ((κ + u) / (κ - u)) ^ n := by
  rw [← weierstrass_kernel h n]
  exact AlgebraicVision.kernel_pos n _

theorem rational_bounds {κ u : ℝ} (hk : 1 < κ) (hu : |u| ≤ 1) (n : ℕ) :
    ((κ - 1) / (κ + 1)) ^ n ≤ ((κ + u) / (κ - u)) ^ n ∧
    ((κ + u) / (κ - u)) ^ n ≤ ((κ + 1) / (κ - 1)) ^ n := by
  have hlo : -1 ≤ u := (abs_le.mp hu).1
  have hhi : u ≤ 1 := (abs_le.mp hu).2
  have h1 : 0 < κ + 1 := by linarith
  have h2 : 0 < κ - 1 := by linarith
  have h3 : 0 < κ - u := by linarith
  have h4 : 0 ≤ (κ - 1) / (κ + 1) := by positivity
  have h5 : 0 ≤ (κ + u) / (κ - u) := div_nonneg (by linarith) h3.le
  constructor
  · apply pow_le_pow_left₀ h4
    apply (div_le_div_iff₀ h1 h3).mpr
    nlinarith
  · apply pow_le_pow_left₀ h5
    apply (div_le_div_iff₀ h3 h2).mpr
    nlinarith

end AlgebraicVision.V2
