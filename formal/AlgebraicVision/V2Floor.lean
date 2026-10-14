import AlgebraicVision.V2Kernel

namespace AlgebraicVision.V2

open Finset

theorem normalized_rational_floor {ι : Type*} (s : Finset ι) (u : ι → ℝ)
    {κ : ℝ} (hk : 1 < κ) (hu : ∀ i ∈ s, |u i| ≤ 1) (n : ℕ)
    {j : ι} (hj : j ∈ s) :
    ((κ - 1) / (κ + 1)) ^ n /
        ((s.card : ℝ) * ((κ + 1) / (κ - 1)) ^ n) ≤
      ((κ + u j) / (κ - u j)) ^ n /
        (∑ i ∈ s, ((κ + u i) / (κ - u i)) ^ n) := by
  have hn : 0 < (s.card : ℝ) := by exact_mod_cast Finset.card_pos.mpr ⟨j, hj⟩
  have hlo : 0 < ((κ - 1) / (κ + 1)) ^ n := by
    apply pow_pos
    apply div_pos <;> linarith
  have hhi : 0 < ((κ + 1) / (κ - 1)) ^ n := by
    apply pow_pos
    apply div_pos <;> linarith
  have hsum : (∑ i ∈ s, ((κ + u i) / (κ - u i)) ^ n) ≤
      (s.card : ℝ) * ((κ + 1) / (κ - 1)) ^ n := by
    calc _ ≤ ∑ _i ∈ s, ((κ + 1) / (κ - 1)) ^ n := by
            exact Finset.sum_le_sum (fun i hi => (rational_bounds hk (hu i hi) n).2)
      _ = _ := by rw [Finset.sum_const, nsmul_eq_mul]
  have hpos : 0 < ∑ i ∈ s, ((κ + u i) / (κ - u i)) ^ n := by
    apply Finset.sum_pos
    · intro i hi
      exact rational_positive (lt_of_le_of_lt (hu i hi) hk) n
    · exact ⟨j, hj⟩
  apply (div_le_div_iff₀ (mul_pos hn hhi) hpos).mpr
  calc _ ≤ ((κ - 1) / (κ + 1)) ^ n *
          ((s.card : ℝ) * ((κ + 1) / (κ - 1)) ^ n) :=
        mul_le_mul_of_nonneg_left hsum hlo.le
    _ ≤ _ := mul_le_mul_of_nonneg_right (rational_bounds hk (hu j hj) n).1
        (mul_pos hn hhi).le

end AlgebraicVision.V2
