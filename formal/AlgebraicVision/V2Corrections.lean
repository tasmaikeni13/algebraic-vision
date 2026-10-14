import AlgebraicVision.V2Centroid

/-! Algebraic self-correction candidates. Positivity and fixed-point properties
do not assert statistical or optimization dominance. -/

namespace AlgebraicVision.V2

noncomputable def tailPositive (t : ℝ) : ℝ := 1 + t + t ^ 2 / (2 + t)

theorem tail_positive {t : ℝ} (ht : 0 ≤ t) : 0 < tailPositive t := by
  unfold tailPositive
  positivity

noncomputable def tailBase (t : ℝ) : ℝ :=
  if 0 ≤ t then tailPositive t else (tailPositive (-t))⁻¹

theorem tail_base_positive (t : ℝ) : 0 < tailBase t := by
  unfold tailBase
  split_ifs with h
  · exact tail_positive h
  · exact inv_pos.mpr (tail_positive (by linarith))

theorem tail_base_reciprocal (t : ℝ) : tailBase t * tailBase (-t) = 1 := by
  unfold tailBase
  split_ifs with h1 h2 h2
  · have : t = 0 := by linarith
    subst t
    norm_num [tailPositive]
  · rw [neg_neg]
    exact mul_inv_cancel₀ (ne_of_gt (tail_positive h1))
  · exact inv_mul_cancel₀ (ne_of_gt (tail_positive h2))
  · linarith

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

noncomputable def energyWeight (d : ℝ) (h : E) : ℝ := d / (d + ‖h‖ ^ 2)

omit [NormedSpace ℝ E] in
theorem energy_weight_positive {d : ℝ} (hd : 0 < d) (h : E) :
    0 < energyWeight d h := div_pos hd (by positivity)

omit [NormedSpace ℝ E] in
theorem energy_denominator_positive {ι : Type*} (s : Finset ι) (hne : s.Nonempty)
    (h : ι → E) {d : ℝ} (hd : 0 < d) : 0 < ∑ i ∈ s, energyWeight d (h i) := by
  exact Finset.sum_pos (fun i _ => energy_weight_positive hd (h i)) hne

theorem energy_centroid_fixed {ι : Type*} (s : Finset ι) (hne : s.Nonempty)
    (h : E) {d : ℝ} (hd : 0 < d) :
    (∑ _i ∈ s, energyWeight d h)⁻¹ • (∑ _i ∈ s, energyWeight d h • h) = h := by
  have hp : (∑ _i ∈ s, energyWeight d h) ≠ 0 :=
    ne_of_gt (energy_denominator_positive s hne (fun _ => h) hd)
  rw [← Finset.sum_smul, smul_smul, inv_mul_cancel₀ hp, one_smul]

end AlgebraicVision.V2
