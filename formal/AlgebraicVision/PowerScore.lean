import Mathlib

/-!
# Strict propriety of the power score

For `a > 0, a ≠ 1` the power score is the Bregman divergence of
`φ t = (t ^ a - t) / (a (a - 1))`. Per class the divergence is

  `B a y p = (y ^ a - p ^ a - a p ^ (a - 1) (y - p)) / (a (a - 1))`,

which is nonnegative for `y ≥ 0, p > 0` by Bernoulli's inequality, and
strictly positive when `y ≠ p`. Summed over classes it is the excess
expected score of predicting `p` when the target distribution is `q`, so the
expected score is uniquely minimised at `p = q`: the loss is strictly
proper, hence top-1 consistent.
-/

namespace AlgebraicVision

open Real Finset

/-- Per-class Bregman term of the power score. -/
noncomputable def bregman (a y p : ℝ) : ℝ :=
  (y ^ a - p ^ a - a * p ^ (a - 1) * (y - p)) / (a * (a - 1))

/-- `p ^ a * (1 + a (y / p - 1)) = p ^ a + a p ^ (a - 1) (y - p)`. -/
lemma tangent_rewrite {a y p : ℝ} (hp : 0 < p) :
    p ^ a * (1 + a * (y / p - 1)) = p ^ a + a * p ^ (a - 1) * (y - p) := by
  rw [rpow_sub_one hp.ne']
  field_simp

/-- `(y / p) ^ a * p ^ a = y ^ a`. -/
lemma ratio_rewrite {a y p : ℝ} (hy : 0 ≤ y) (hp : 0 < p) :
    (y / p) ^ a * p ^ a = y ^ a := by
  rw [div_rpow hy hp.le, div_mul_cancel₀ _ (rpow_pos_of_pos hp a).ne']

theorem bregman_nonneg {a y p : ℝ} (ha : 0 < a) (ha1 : a ≠ 1) (hy : 0 ≤ y)
    (hp : 0 < p) : 0 ≤ bregman a y p := by
  have hs : -1 ≤ y / p - 1 := by
    have := div_nonneg hy hp.le
    linarith
  have hpa : 0 < p ^ a := rpow_pos_of_pos hp a
  have e1 : (1 + (y / p - 1)) = y / p := by ring
  rcases lt_or_gt_of_ne ha1 with h | h
  · -- 0 < a < 1: concave power, numerator ≤ 0, denominator < 0.
    have bern := rpow_one_add_le_one_add_mul_self hs ha.le h.le
    rw [e1] at bern
    have num : y ^ a - p ^ a - a * p ^ (a - 1) * (y - p) ≤ 0 := by
      have := mul_le_mul_of_nonneg_right bern hpa.le
      rw [ratio_rewrite hy hp, mul_comm (1 + _) _, tangent_rewrite hp] at this
      linarith
    have den : a * (a - 1) < 0 := mul_neg_of_pos_of_neg ha (by linarith)
    exact div_nonneg_of_nonpos num den.le
  · -- a > 1: convex power, numerator ≥ 0, denominator > 0.
    have bern := one_add_mul_self_le_rpow_one_add hs h.le
    rw [e1] at bern
    have num : 0 ≤ y ^ a - p ^ a - a * p ^ (a - 1) * (y - p) := by
      have := mul_le_mul_of_nonneg_right bern hpa.le
      rw [ratio_rewrite hy hp, mul_comm (1 + _) _, tangent_rewrite hp] at this
      linarith
    exact div_nonneg num (mul_pos ha (by linarith)).le

theorem bregman_pos {a y p : ℝ} (ha : 0 < a) (ha1 : a ≠ 1) (hy : 0 ≤ y)
    (hp : 0 < p) (hne : y ≠ p) : 0 < bregman a y p := by
  have hs : -1 ≤ y / p - 1 := by
    have := div_nonneg hy hp.le
    linarith
  have hs0 : y / p - 1 ≠ 0 := by
    intro h0
    apply hne
    have : y / p = 1 := by linarith
    rwa [div_eq_one_iff_eq hp.ne'] at this
  have hpa : 0 < p ^ a := rpow_pos_of_pos hp a
  have e1 : (1 + (y / p - 1)) = y / p := by ring
  rcases lt_or_gt_of_ne ha1 with h | h
  · have bern := rpow_one_add_lt_one_add_mul_self hs hs0 ha h
    rw [e1] at bern
    have num : y ^ a - p ^ a - a * p ^ (a - 1) * (y - p) < 0 := by
      have := mul_lt_mul_of_pos_right bern hpa
      rw [ratio_rewrite hy hp, mul_comm (1 + _) _, tangent_rewrite hp] at this
      linarith
    exact div_pos_of_neg_of_neg num (mul_neg_of_pos_of_neg ha (by linarith))
  · have bern := one_add_mul_self_lt_rpow_one_add hs hs0 h
    rw [e1] at bern
    have num : 0 < y ^ a - p ^ a - a * p ^ (a - 1) * (y - p) := by
      have := mul_lt_mul_of_pos_right bern hpa
      rw [ratio_rewrite hy hp, mul_comm (1 + _) _, tangent_rewrite hp] at this
      linarith
    exact div_pos num (mul_pos ha (by linarith))

/-- Expected power score of predicting `p` under target distribution `q`
(up to terms that do not depend on `p`). -/
noncomputable def expectedScore {ι : Type*} (s : Finset ι) (a : ℝ)
    (p q : ι → ℝ) : ℝ :=
  ∑ c ∈ s, (p c ^ a / a - q c * p c ^ (a - 1) / (a - 1))

/-- The excess expected score equals the summed Bregman divergence. -/
theorem excess_score_eq {ι : Type*} (s : Finset ι) {a : ℝ} (ha : 0 < a)
    (ha1 : a ≠ 1) (p q : ι → ℝ) (hp : ∀ c ∈ s, 0 < p c)
    (hq : ∀ c ∈ s, 0 < q c) :
    expectedScore s a p q - expectedScore s a q q =
      ∑ c ∈ s, bregman a (q c) (p c) := by
  unfold expectedScore bregman
  rw [← sum_sub_distrib]
  apply sum_congr rfl
  intro c hc
  have hpc := hp c hc
  have hqc := hq c hc
  have hpa : p c ^ a = p c ^ (a - 1) * p c := by
    rw [rpow_sub_one hpc.ne', div_mul_cancel₀ _ hpc.ne']
  have hqa : q c ^ a = q c ^ (a - 1) * q c := by
    rw [rpow_sub_one hqc.ne', div_mul_cancel₀ _ hqc.ne']
  have ha0 : a ≠ 0 := ha.ne'
  have ha10 : a - 1 ≠ 0 := sub_ne_zero.mpr ha1
  rw [hpa, hqa]
  field_simp
  ring

/-- Propriety: the expected score is minimised by the true distribution. -/
theorem power_score_proper {ι : Type*} (s : Finset ι) {a : ℝ} (ha : 0 < a)
    (ha1 : a ≠ 1) (p q : ι → ℝ) (hp : ∀ c ∈ s, 0 < p c)
    (hq : ∀ c ∈ s, 0 < q c) :
    expectedScore s a q q ≤ expectedScore s a p q := by
  have h := excess_score_eq s ha ha1 p q hp hq
  have : 0 ≤ ∑ c ∈ s, bregman a (q c) (p c) :=
    sum_nonneg fun c hc => bregman_nonneg ha ha1 (hq c hc).le (hp c hc)
  linarith

/-- Strict propriety: any other prediction has strictly larger expected
score. -/
theorem power_score_strictly_proper {ι : Type*} (s : Finset ι) {a : ℝ}
    (ha : 0 < a) (ha1 : a ≠ 1) (p q : ι → ℝ) (hp : ∀ c ∈ s, 0 < p c)
    (hq : ∀ c ∈ s, 0 < q c) {c₀ : ι} (hc₀ : c₀ ∈ s) (hne : q c₀ ≠ p c₀) :
    expectedScore s a q q < expectedScore s a p q := by
  have h := excess_score_eq s ha ha1 p q hp hq
  have : 0 < ∑ c ∈ s, bregman a (q c) (p c) :=
    sum_pos' (fun c hc => bregman_nonneg ha ha1 (hq c hc).le (hp c hc))
      ⟨c₀, hc₀, bregman_pos ha ha1 (hq c₀ hc₀).le (hp c₀ hc₀) hne⟩
  linarith

end AlgebraicVision
