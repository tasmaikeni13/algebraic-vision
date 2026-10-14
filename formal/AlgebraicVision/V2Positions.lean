import AlgebraicVision.Cayley2D

namespace AlgebraicVision.V2

open Matrix

theorem projective_polynomial (v u : ℝ) :
    (v ^ 2 - u ^ 2) ^ 2 + (2 * u * v) ^ 2 = (v ^ 2 + u ^ 2) ^ 2 := by ring

theorem projective_norm_step (v u w : ℝ) :
    (v - w * u) ^ 2 + (u + w * v) ^ 2 = (1 + w ^ 2) * (v ^ 2 + u ^ 2) := by ring

def homogeneous (w : ℝ) : ℕ → ℝ × ℝ
  | 0 => (1, 0)
  | n + 1 => let p := homogeneous w n; (p.1 - w * p.2, p.2 + w * p.1)

theorem homogeneous_nonzero (w : ℝ) (n : ℕ) :
    0 < (homogeneous w n).1 ^ 2 + (homogeneous w n).2 ^ 2 := by
  induction n with
  | zero => norm_num [homogeneous]
  | succ n ih =>
    simp only [homogeneous]
    rw [projective_norm_step]
    exact mul_pos (by positivity) ih

noncomputable def projective (v u : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![(v ^ 2 - u ^ 2) / (v ^ 2 + u ^ 2), -(2 * u * v) / (v ^ 2 + u ^ 2);
     (2 * u * v) / (v ^ 2 + u ^ 2), (v ^ 2 - u ^ 2) / (v ^ 2 + u ^ 2)]

theorem projective_orthogonal {v u : ℝ} (h : v ^ 2 + u ^ 2 ≠ 0) :
    (projective v u)ᵀ * projective v u = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [projective, Matrix.mul_apply, Fin.sum_univ_two] <;> field_simp <;> ring

theorem projective_det {v u : ℝ} (h : v ^ 2 + u ^ 2 ≠ 0) :
    (projective v u).det = 1 := by
  rw [Matrix.det_fin_two]
  simp [projective]
  field_simp
  ring

theorem projective_step {v u : ℝ} (h : v ^ 2 + u ^ 2 ≠ 0) (w : ℝ) :
    projective (v - w * u) (u + w * v) = AlgebraicVision.cayley w * projective v u := by
  have hw : 1 + w ^ 2 ≠ 0 := by positivity
  have hn : (v - w * u) ^ 2 + (u + w * v) ^ 2 ≠ 0 := by
    rw [projective_norm_step]
    exact mul_ne_zero hw h
  unfold projective AlgebraicVision.cayley
  simp only [projective_norm_step]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Fin.sum_univ_two] <;>
    field_simp <;> ring

theorem projective_power (w : ℝ) (n : ℕ) :
    projective (homogeneous w n).1 (homogeneous w n).2 = AlgebraicVision.cayley w ^ n := by
  induction n with
  | zero => ext i j; fin_cases i <;> fin_cases j <;> norm_num [homogeneous, projective]
  | succ n ih =>
    simp only [homogeneous]
    rw [projective_step (ne_of_gt (homogeneous_nonzero w n)), ih, pow_succ']

-- Both generators act on both planes: column=(+wx,+wx), row=(+wy,-wy).
noncomputable def biCol (wx : ℝ) : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ :=
  AlgebraicVision.colGen wx * AlgebraicVision.rowGen wx

noncomputable def biRow (wy : ℝ) : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ :=
  AlgebraicVision.colGen wy * (AlgebraicVision.rowGen wy)⁻¹

lemma col_generator_det (w : ℝ) :
    ((AlgebraicVision.colGen w : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
      Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ).det = 1 := by
  change (Matrix.fromBlocks (AlgebraicVision.cayley w) 0 0 1).det = 1
  rw [Matrix.det_fromBlocks_zero₂₁, AlgebraicVision.cayley_det, Matrix.det_one, mul_one]

lemma row_generator_det (w : ℝ) :
    ((AlgebraicVision.rowGen w : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
      Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ).det = 1 := by
  change (Matrix.fromBlocks 1 0 0 (AlgebraicVision.cayley w)).det = 1
  rw [Matrix.det_fromBlocks_zero₂₁, AlgebraicVision.cayley_det, Matrix.det_one, one_mul]

lemma row_inverse_det (w : ℝ) :
    (((AlgebraicVision.rowGen w)⁻¹ : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
      Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ).det = 1 := by
  have he := congrArg
    (fun a : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ =>
      (a : Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ).det)
    (inv_mul_cancel (AlgebraicVision.rowGen w))
  change ((((AlgebraicVision.rowGen w)⁻¹ : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
      Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ) *
    (AlgebraicVision.rowGen w : Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ)).det =
      (1 : Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ).det at he
  simpa only [Matrix.det_mul, row_generator_det, Matrix.det_one, mul_one] using he

theorem bi_col_det (wx : ℝ) :
    ((biCol wx : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
      Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ).det = 1 := by
  change ((AlgebraicVision.colGen wx : Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ) *
    (AlgebraicVision.rowGen wx : Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ)).det = 1
  rw [Matrix.det_mul, col_generator_det, row_generator_det, mul_one]

theorem bi_row_det (wy : ℝ) :
    ((biRow wy : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
      Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ).det = 1 := by
  change ((AlgebraicVision.colGen wy : Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ) *
    (((AlgebraicVision.rowGen wy)⁻¹ : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
      Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ)).det = 1
  rw [Matrix.det_mul, col_generator_det, row_inverse_det, mul_one]

lemma col_col_commute (a b : ℝ) : Commute (AlgebraicVision.colGen a) (AlgebraicVision.colGen b) := by
  apply Subtype.ext
  change Matrix.fromBlocks (AlgebraicVision.cayley a) 0 0 1 *
      Matrix.fromBlocks (AlgebraicVision.cayley b) 0 0 1 =
    Matrix.fromBlocks (AlgebraicVision.cayley b) 0 0 1 *
      Matrix.fromBlocks (AlgebraicVision.cayley a) 0 0 1
  simp only [Matrix.fromBlocks_multiply]
  simp [AlgebraicVision.cayley_commute]

lemma row_row_commute (a b : ℝ) : Commute (AlgebraicVision.rowGen a) (AlgebraicVision.rowGen b) := by
  apply Subtype.ext
  change Matrix.fromBlocks 1 0 0 (AlgebraicVision.cayley a) *
      Matrix.fromBlocks 1 0 0 (AlgebraicVision.cayley b) =
    Matrix.fromBlocks 1 0 0 (AlgebraicVision.cayley b) *
      Matrix.fromBlocks 1 0 0 (AlgebraicVision.cayley a)
  simp only [Matrix.fromBlocks_multiply]
  simp [AlgebraicVision.cayley_commute]

theorem bi_commute (wx wy : ℝ) : Commute (biCol wx) (biRow wy) := by
  unfold biCol biRow
  exact ((col_col_commute wx wy).mul_right
    ((AlgebraicVision.col_row_commute wx wy).inv_right)).mul_left
      ((AlgebraicVision.col_row_commute wy wx).symm.mul_right
        ((row_row_commute wx wy).inv_right))

theorem bicayley_relative (wx wy : ℝ) (r₁ c₁ r₂ c₂ : ℤ)
    (q k : Fin 2 ⊕ Fin 2 → ℝ) :
    (((biCol wx ^ c₁ * biRow wy ^ r₁ : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
        Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ) *ᵥ q) ⬝ᵥ
      (((biCol wx ^ c₂ * biRow wy ^ r₂ : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
        Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ) *ᵥ k) =
    q ⬝ᵥ (((biCol wx ^ (c₂ - c₁) * biRow wy ^ (r₂ - r₁) :
        Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
        Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ) *ᵥ k) :=
  AlgebraicVision.relative_logit (bi_commute wx wy) r₁ c₁ r₂ c₂ q k

-- a,b are the two nonzero Cayley angles. This states the exact local rank
-- condition; it deliberately does not claim arbitrary frequencies are injective.
theorem tied_rank_nonzero {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) :
    (a * (-b) - b * a) ≠ 0 := by
  have h := mul_ne_zero ha hb
  intro he
  apply h
  nlinarith

theorem tied_cayley_angle_rank {wx wy : ℝ} (hx : 0 < wx) (hy : 0 < wy) :
    ((2 * Real.arctan wx) * (-(2 * Real.arctan wy)) -
      (2 * Real.arctan wy) * (2 * Real.arctan wx)) ≠ 0 := by
  apply tied_rank_nonzero
  · exact ne_of_gt (mul_pos (by norm_num) (Real.arctan_pos.mpr hx))
  · exact ne_of_gt (mul_pos (by norm_num) (Real.arctan_pos.mpr hy))

end AlgebraicVision.V2
