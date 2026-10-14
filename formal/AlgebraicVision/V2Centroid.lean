import Mathlib

namespace AlgebraicVision.V2

open Finset

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

noncomputable def klein (d : ℝ) (h : E) : E :=
  (Real.sqrt (d + ‖h‖ ^ 2))⁻¹ • h

theorem klein_norm_sq {d : ℝ} (hd : 0 < d) (h : E) :
    ‖klein d h‖ ^ 2 = ‖h‖ ^ 2 / (d + ‖h‖ ^ 2) := by
  have hp : 0 ≤ d + ‖h‖ ^ 2 := by positivity
  simp [klein, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs, inv_pow,
    Real.sq_sqrt hp, div_eq_mul_inv, mul_comm]

theorem klein_norm_bound {d R : ℝ} (hd : 0 < d) (hR : 0 ≤ R)
    (h : E) (hh : ‖h‖ ^ 2 ≤ R) :
    ‖klein d h‖ ≤ Real.sqrt (R / (d + R)) := by
  apply Real.le_sqrt_of_sq_le
  rw [klein_norm_sq hd]
  apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
  nlinarith

theorem centroid_denominator_lower {ι : Type*} (s : Finset ι) (h : ι → E)
    {d R : ℝ} (hd : 0 < d) (hR : 0 ≤ R)
    (hmax : ∀ i ∈ s, ‖h i‖ ^ 2 ≤ R) :
    (s.card : ℝ) ^ 2 * d / (d + R) ≤
      (s.card : ℝ) ^ 2 - ‖∑ i ∈ s, klein d (h i)‖ ^ 2 := by
  have hdR : 0 < d + R := by positivity
  have hb : ‖∑ i ∈ s, klein d (h i)‖ ≤
      (s.card : ℝ) * Real.sqrt (R / (d + R)) := by
    calc ‖∑ i ∈ s, klein d (h i)‖ ≤ ∑ i ∈ s, ‖klein d (h i)‖ := norm_sum_le _ _
      _ ≤ ∑ _i ∈ s, Real.sqrt (R / (d + R)) := by
        exact Finset.sum_le_sum (fun i hi => klein_norm_bound hd hR (h i) (hmax i hi))
      _ = _ := by rw [Finset.sum_const, nsmul_eq_mul]
  have hs := pow_le_pow_left₀ (norm_nonneg (∑ i ∈ s, klein d (h i))) hb 2
  rw [mul_pow, Real.sq_sqrt (div_nonneg hR hdR.le)] at hs
  have hs' : ‖∑ i ∈ s, klein d (h i)‖ ^ 2 * (d + R) ≤ (s.card : ℝ) ^ 2 * R := by
    apply (le_div_iff₀ hdR).mp
    convert hs using 1
    ring
  apply (div_le_iff₀ hdR).mpr
  nlinarith

theorem centroid_denominator_positive {ι : Type*} (s : Finset ι) (hne : s.Nonempty)
    (h : ι → E) {d R : ℝ} (hd : 0 < d) (hR : 0 ≤ R)
    (hmax : ∀ i ∈ s, ‖h i‖ ^ 2 ≤ R) :
    0 < (s.card : ℝ) ^ 2 - ‖∑ i ∈ s, klein d (h i)‖ ^ 2 := by
  have hn : 0 < (s.card : ℝ) := by exact_mod_cast Finset.card_pos.mpr hne
  have hpos : 0 < (s.card : ℝ) ^ 2 * d / (d + R) := by positivity
  exact lt_of_lt_of_le hpos (centroid_denominator_lower s h hd hR hmax)

theorem centroid_denominator_max {ι : Type*} (s : Finset ι) (hne : s.Nonempty)
    (h : ι → E) {d : ℝ} (hd : 0 < d) :
    0 < (s.card : ℝ) ^ 2 * d / (d + s.sup' hne (fun i => ‖h i‖ ^ 2)) ∧
    (s.card : ℝ) ^ 2 * d / (d + s.sup' hne (fun i => ‖h i‖ ^ 2)) ≤
      (s.card : ℝ) ^ 2 - ‖∑ i ∈ s, klein d (h i)‖ ^ 2 := by
  let i := Classical.choose hne
  have hi : i ∈ s := Classical.choose_spec hne
  have hR : 0 ≤ s.sup' hne (fun j => ‖h j‖ ^ 2) :=
    (sq_nonneg ‖h i‖).trans (Finset.le_sup' (fun j => ‖h j‖ ^ 2) hi)
  have hn : 0 < (s.card : ℝ) := by exact_mod_cast Finset.card_pos.mpr hne
  constructor
  · positivity
  · exact centroid_denominator_lower s h hd hR
      (fun j hj => Finset.le_sup' (fun k => ‖h k‖ ^ 2) hj)

end AlgebraicVision.V2
