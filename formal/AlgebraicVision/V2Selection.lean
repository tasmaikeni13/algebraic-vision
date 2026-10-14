import AlgebraicVision.V2Corrections
import AlgebraicVision.V2Positions

/-! Exact-real properties of the final Phase 1 refinement. The slope profile
is identified with the derivative by the separate symbolic analysis. These
theorems do not assert population accuracy or floating-point exactness. -/

namespace AlgebraicVision.V2

noncomputable def boundedFrequency (cap raw : ℝ) : ℝ :=
  cap * (1 / 4 + 3 / (4 * (1 + raw ^ 2)))

theorem bounded_frequency_bounds {cap : ℝ} (hc : 0 < cap) (raw : ℝ) :
    cap / 4 < boundedFrequency cap raw ∧ boundedFrequency cap raw ≤ cap := by
  have hd : 0 < 4 * (1 + raw ^ 2) := by positivity
  have hp : 0 < 3 / (4 * (1 + raw ^ 2)) := div_pos (by norm_num) hd
  have hu : 3 / (4 * (1 + raw ^ 2)) ≤ (3 / 4 : ℝ) := by
    rw [div_le_iff₀ hd]
    nlinarith [sq_nonneg raw]
  unfold boundedFrequency
  constructor
  · nlinarith [mul_pos hc hp]
  · nlinarith [mul_nonneg (le_of_lt hc) (sub_nonneg.mpr hu)]

theorem bounded_frequency_rank {cap : ℝ} (hc : 0 < cap) (rx ry : ℝ) :
    ((2 * Real.arctan (boundedFrequency cap rx)) *
      (-(2 * Real.arctan (boundedFrequency cap ry))) -
      (2 * Real.arctan (boundedFrequency cap ry)) *
      (2 * Real.arctan (boundedFrequency cap rx))) ≠ 0 := by
  apply tied_cayley_angle_rank
  · have := (bounded_frequency_bounds hc rx).1
    linarith
  · have := (bounded_frequency_bounds hc ry).1
    linarith

theorem paired_phases_zero {a b dr dc : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)
    (hp : dc * a + dr * b = 0) (hm : dc * a - dr * b = 0) :
    dr = 0 ∧ dc = 0 := by
  constructor
  · exact (mul_eq_zero.mp (show dr * b = 0 by linarith)).resolve_right hb
  · exact (mul_eq_zero.mp (show dc * a = 0 by linarith)).resolve_right ha

noncomputable def softsignGate (c x : ℝ) : ℝ :=
  x / 2 * (1 + c * x / (1 + c * |x|))

noncomputable def softsignSlope (c x : ℝ) : ℝ :=
  if 0 ≤ x then 1 - 1 / (2 * (1 + c * x) ^ 2)
  else 1 / (2 * (1 - c * x) ^ 2)

theorem softsign_gate_zero (c : ℝ) : softsignGate c 0 = 0 := by
  simp [softsignGate]

theorem softsign_gate_reflection (c x : ℝ) :
    softsignGate c x - softsignGate c (-x) = x := by
  unfold softsignGate
  rw [abs_neg]
  ring

theorem softsign_slope_bounds {c : ℝ} (hc : 0 < c) (x : ℝ) :
    0 < softsignSlope c x ∧ softsignSlope c x < 1 := by
  unfold softsignSlope
  split_ifs with hx
  · have hp : 0 ≤ c * x := mul_nonneg (le_of_lt hc) hx
    have hd : 0 < 2 * (1 + c * x) ^ 2 := by positivity
    have hpos : 0 < 1 / (2 * (1 + c * x) ^ 2) := div_pos (by norm_num) hd
    have hle : 1 / (2 * (1 + c * x) ^ 2) ≤ (1 / 2 : ℝ) := by
      rw [div_le_iff₀ hd]
      nlinarith [sq_nonneg (c * x)]
    constructor <;> linarith
  · have hp : c * x ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (le_of_lt hc) (by linarith)
    have hbase : 0 < 1 - c * x := by linarith
    have hd : 0 < 2 * (1 - c * x) ^ 2 := by positivity
    have hpos : 0 < 1 / (2 * (1 - c * x) ^ 2) := div_pos (by norm_num) hd
    have hle : 1 / (2 * (1 - c * x) ^ 2) ≤ (1 / 2 : ℝ) := by
      rw [div_le_iff₀ hd]
      nlinarith [sq_nonneg (c * x)]
    constructor <;> linarith

-- An exact polynomial witness used in the rejected Padé investigation.
theorem pade_residual_certificate {a : ℝ} (ha : 0 ≤ a) :
    0 ≤ (4 + 6 * a + 5 * a ^ 2 + 2 * a ^ 3) ^ 2 * (1 + a ^ 2) - 256 * a ^ 5 := by
  have hw : 0 ≤ a ^ 4 * (53 * a - 80) ^ 2 + 225 * a ^ 4 +
      212 * a ^ 8 + 1060 * a ^ 7 + 6572 * a ^ 3 + 4876 * a ^ 2 + 2544 * a + 848 := by
    positivity
  have he : a ^ 4 * (53 * a - 80) ^ 2 + 225 * a ^ 4 +
      212 * a ^ 8 + 1060 * a ^ 7 + 6572 * a ^ 3 + 4876 * a ^ 2 + 2544 * a + 848 =
      53 * ((4 + 6 * a + 5 * a ^ 2 + 2 * a ^ 3) ^ 2 * (1 + a ^ 2) - 256 * a ^ 5) := by ring
  rw [he] at hw
  linarith

theorem newton_residual (r a : ℝ) (hr : r ≠ 0) :
    ((r + a / r) / 2) ^ 2 - a = (r ^ 2 - a) ^ 2 / (4 * r ^ 2) := by
  field_simp
  ring

theorem affine_cost_nonpositive {initial slope batch : ℝ}
    (hi : initial ≤ 0) (hs : slope ≤ 0) (hb : 1 ≤ batch) :
    initial + (batch - 1) * slope ≤ 0 := by
  have := mul_nonpos_of_nonneg_of_nonpos (show 0 ≤ batch - 1 by linarith) hs
  linarith

end AlgebraicVision.V2
