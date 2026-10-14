import Mathlib

/-!
# Attention with a sink

Weights `p j = e j / (Ω + ∑ e)` with positive kernel values `e` and a sink
mass `Ω ≥ 0`. The visible mass is `1 - Ω / D`, strictly below one when
`Ω > 0`, so a head may attend to nothing. Entries of the score Jacobian
`g_j p_i (δ_ij - p_j)` are bounded by `g / 4`. Tiled accumulation of
numerators and denominators over disjoint key tiles reproduces the dense
result exactly, with no running maximum.
-/

namespace AlgebraicVision

open Finset

variable {ι : Type*}

theorem sink_mass (s : Finset ι) (e : ι → ℝ) {Ω : ℝ} (he : ∀ i ∈ s, 0 < e i)
    (hΩ : 0 ≤ Ω) (hs : s.Nonempty) :
    ∑ i ∈ s, e i / (Ω + ∑ j ∈ s, e j) = 1 - Ω / (Ω + ∑ j ∈ s, e j) := by
  have hpos : 0 < Ω + ∑ j ∈ s, e j := by
    have := sum_pos he hs
    linarith
  rw [← sum_div, eq_sub_iff_add_eq, ← add_div, div_eq_one_iff_eq hpos.ne']
  ring

theorem sink_mass_lt_one (s : Finset ι) (e : ι → ℝ) {Ω : ℝ}
    (he : ∀ i ∈ s, 0 < e i) (hΩ : 0 < Ω) (hs : s.Nonempty) :
    ∑ i ∈ s, e i / (Ω + ∑ j ∈ s, e j) < 1 := by
  rw [sink_mass s e he hΩ.le hs]
  have hpos : 0 < Ω + ∑ j ∈ s, e j := by
    have := sum_pos he hs
    linarith
  have : 0 < Ω / (Ω + ∑ j ∈ s, e j) := div_pos hΩ hpos
  linarith

theorem weight_pos (s : Finset ι) (e : ι → ℝ) {Ω : ℝ} (he : ∀ i ∈ s, 0 < e i)
    (hΩ : 0 ≤ Ω) (hs : s.Nonempty) {i : ι} (hi : i ∈ s) :
    0 < e i / (Ω + ∑ j ∈ s, e j) := by
  have := sum_pos he hs
  exact div_pos (he i hi) (by linarith)

/-- Diagonal Jacobian entry: `|g p (1 - p)| ≤ g / 4` for `0 ≤ g`, `p ∈ [0, 1]`. -/
theorem jacobian_diag_le {g p : ℝ} (hg : 0 ≤ g) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    |g * (p * (1 - p))| ≤ g / 4 := by
  have h1 : 0 ≤ p * (1 - p) := mul_nonneg hp0 (by linarith)
  have h2 : p * (1 - p) ≤ 1 / 4 := by nlinarith [sq_nonneg (p - 1 / 2)]
  rw [abs_of_nonneg (mul_nonneg hg h1)]
  nlinarith

/-- Off-diagonal entry: `|g p q| ≤ g / 4` when `p, q ≥ 0` and `p + q ≤ 1`. -/
theorem jacobian_offdiag_le {g p q : ℝ} (hg : 0 ≤ g) (hp : 0 ≤ p) (hq : 0 ≤ q)
    (hpq : p + q ≤ 1) : |g * (p * q)| ≤ g / 4 := by
  have h1 : 0 ≤ p * q := mul_nonneg hp hq
  have h2 : p * q ≤ 1 / 4 := by nlinarith [sq_nonneg (p - q)]
  rw [abs_of_nonneg (mul_nonneg hg h1)]
  nlinarith

/-- Tiled accumulation over disjoint key tiles equals the dense output. -/
theorem tiled_eq_dense {κ : Type*} [DecidableEq ι] (tiles : Finset κ)
    (T : κ → Finset ι) (hdisj : (tiles : Set κ).PairwiseDisjoint T)
    (e v : ι → ℝ) (Ω : ℝ) :
    (∑ t ∈ tiles, ∑ j ∈ T t, e j * v j) / (Ω + ∑ t ∈ tiles, ∑ j ∈ T t, e j)
      = (∑ j ∈ tiles.biUnion T, e j * v j) /
          (Ω + ∑ j ∈ tiles.biUnion T, e j) := by
  rw [sum_biUnion hdisj, sum_biUnion hdisj]

end AlgebraicVision
