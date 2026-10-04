import Mathlib

/-!
# Two-dimensional Cayley rotations

`R w = [[1 - w^2, -2w], [2w, 1 - w^2]] / (1 + w^2)` is a rotation: it is
orthogonal with determinant one. The axial 2-D position map sends grid cell
`(r, c)` to `A ^ c * B ^ r` for two commuting orthogonal maps (one acting on
the column pairs, one on the row pairs). We prove the relative-position
identity for any such map: the attention logit between positions `p` and
`q` depends only on `q - p`. Any two Cayley rotations commute, so the
identity also holds for the learned variant in which one feature pair turns
with both the column and the row, by independent learned angles.
-/

namespace AlgebraicVision

open Matrix

/-- The Cayley rotation matrix. -/
noncomputable def cayley (w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![(1 - w ^ 2) / (1 + w ^ 2), -(2 * w) / (1 + w ^ 2);
     (2 * w) / (1 + w ^ 2), (1 - w ^ 2) / (1 + w ^ 2)]

theorem cayley_orthogonal (w : ℝ) : (cayley w)ᵀ * cayley w = 1 := by
  have h : (1 + w ^ 2) ≠ 0 := by positivity
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [cayley, Matrix.mul_apply, Fin.sum_univ_two] <;> field_simp <;> ring

theorem cayley_det (w : ℝ) : (cayley w).det = 1 := by
  have h : (1 + w ^ 2) ≠ 0 := by positivity
  rw [Matrix.det_fin_two]
  simp [cayley]
  field_simp
  ring

/-- Group form of the relative-position property: for commuting `a, b`,
`(a^c₁ b^r₁)⁻¹ (a^c₂ b^r₂) = a^(c₂-c₁) b^(r₂-r₁)`. -/
theorem axial_relative {G : Type*} [Group G] {a b : G} (hab : Commute a b)
    (r₁ c₁ r₂ c₂ : ℤ) :
    (a ^ c₁ * b ^ r₁)⁻¹ * (a ^ c₂ * b ^ r₂) = a ^ (c₂ - c₁) * b ^ (r₂ - r₁) := by
  have hc : Commute (a ^ (c₂ - c₁)) (b ^ (-r₁)) := hab.zpow_zpow _ _
  calc (a ^ c₁ * b ^ r₁)⁻¹ * (a ^ c₂ * b ^ r₂)
      = b ^ (-r₁) * (a ^ (-c₁) * a ^ c₂) * b ^ r₂ := by
        rw [_root_.mul_inv_rev, _root_.zpow_neg, _root_.zpow_neg]
        group
    _ = b ^ (-r₁) * a ^ (c₂ - c₁) * b ^ r₂ := by
        rw [← _root_.zpow_add, neg_add_eq_sub]
    _ = a ^ (c₂ - c₁) * b ^ (-r₁) * b ^ r₂ := by rw [hc.eq]
    _ = a ^ (c₂ - c₁) * b ^ (r₂ - r₁) := by
        rw [mul_assoc, ← _root_.zpow_add, neg_add_eq_sub]

section Relative

variable {n : Type*} [Fintype n] [DecidableEq n]

omit [DecidableEq n] in
/-- Logits of rotated queries and keys: for any `M, N`,
`⟪M q, N k⟫ = ⟪q, (Mᵀ N) k⟫`. -/
theorem rotated_logit (M N : Matrix n n ℝ) (q k : n → ℝ) :
    (M *ᵥ q) ⬝ᵥ (N *ᵥ k) = q ⬝ᵥ ((Mᵀ * N) *ᵥ k) := by
  calc (M *ᵥ q) ⬝ᵥ (N *ᵥ k) = (q ᵥ* Mᵀ) ⬝ᵥ (N *ᵥ k) := by
        rw [Matrix.vecMul_transpose]
    _ = ((q ᵥ* Mᵀ) ᵥ* N) ⬝ᵥ k := Matrix.dotProduct_mulVec _ _ _
    _ = (q ᵥ* (Mᵀ * N)) ⬝ᵥ k := by rw [Matrix.vecMul_vecMul]
    _ = q ⬝ᵥ ((Mᵀ * N) *ᵥ k) := (Matrix.dotProduct_mulVec _ _ _).symm

/-- The 2-D relative-position identity for orthogonal commuting generators:
`⟪P(r₁,c₁) q, P(r₂,c₂) k⟫ = ⟪q, P(r₂-r₁, c₂-c₁) k⟫` with `P(r,c) = A^c B^r`. -/
theorem relative_logit {A B : Matrix.orthogonalGroup n ℝ} (hAB : Commute A B)
    (r₁ c₁ r₂ c₂ : ℤ) (q k : n → ℝ) :
    (((A ^ c₁ * B ^ r₁ : Matrix.orthogonalGroup n ℝ) : Matrix n n ℝ) *ᵥ q) ⬝ᵥ
        (((A ^ c₂ * B ^ r₂ : Matrix.orthogonalGroup n ℝ) : Matrix n n ℝ) *ᵥ k) =
      q ⬝ᵥ (((A ^ (c₂ - c₁) * B ^ (r₂ - r₁) : Matrix.orthogonalGroup n ℝ) :
        Matrix n n ℝ) *ᵥ k) := by
  rw [rotated_logit, ← axial_relative hAB r₁ c₁ r₂ c₂]
  congr 2

end Relative

theorem cayley_mem_orthogonal (w : ℝ) :
    cayley w ∈ Matrix.orthogonalGroup (Fin 2) ℝ := by
  rw [Matrix.mem_orthogonalGroup_iff]
  exact _root_.mul_eq_one_comm.mp (cayley_orthogonal w)

/-- Column generator: rotate the first pair by `R wx`, leave the second. -/
noncomputable def colGen (wx : ℝ) : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ :=
  ⟨Matrix.fromBlocks (cayley wx) 0 0 1, by
    rw [Matrix.mem_orthogonalGroup_iff, Matrix.fromBlocks_transpose,
      Matrix.fromBlocks_multiply]
    have h := (Matrix.mem_orthogonalGroup_iff _ _).mp (cayley_mem_orthogonal wx)
    simp [h, Matrix.fromBlocks_one]⟩

/-- Row generator: leave the first pair, rotate the second by `R wy`. -/
noncomputable def rowGen (wy : ℝ) : Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ :=
  ⟨Matrix.fromBlocks 1 0 0 (cayley wy), by
    rw [Matrix.mem_orthogonalGroup_iff, Matrix.fromBlocks_transpose,
      Matrix.fromBlocks_multiply]
    have h := (Matrix.mem_orthogonalGroup_iff _ _).mp (cayley_mem_orthogonal wy)
    simp [h, Matrix.fromBlocks_one]⟩

theorem col_row_commute (wx wy : ℝ) : Commute (colGen wx) (rowGen wy) := by
  apply Subtype.ext
  show Matrix.fromBlocks (cayley wx) 0 0 1 * Matrix.fromBlocks 1 0 0 (cayley wy) =
    Matrix.fromBlocks 1 0 0 (cayley wy) * Matrix.fromBlocks (cayley wx) 0 0 1
  rw [Matrix.fromBlocks_multiply, Matrix.fromBlocks_multiply]
  simp

/-- The model's axial Cayley positions: for a query at grid cell `(r₁, c₁)`
and a key at `(r₂, c₂)` the rotated logit depends only on the offset
`(r₂ - r₁, c₂ - c₁)`. -/
theorem cayley2d_relative (wx wy : ℝ) (r₁ c₁ r₂ c₂ : ℤ)
    (q k : Fin 2 ⊕ Fin 2 → ℝ) :
    (((colGen wx ^ c₁ * rowGen wy ^ r₁ :
        Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
        Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ) *ᵥ q) ⬝ᵥ
      (((colGen wx ^ c₂ * rowGen wy ^ r₂ :
        Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
        Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ) *ᵥ k) =
    q ⬝ᵥ (((colGen wx ^ (c₂ - c₁) * rowGen wy ^ (r₂ - r₁) :
        Matrix.orthogonalGroup (Fin 2 ⊕ Fin 2) ℝ) :
        Matrix (Fin 2 ⊕ Fin 2) (Fin 2 ⊕ Fin 2) ℝ) *ᵥ k) :=
  relative_logit (col_row_commute wx wy) r₁ c₁ r₂ c₂ q k

/-- Any two Cayley rotations commute (planar rotations form an abelian group). -/
theorem cayley_commute (a b : ℝ) : cayley a * cayley b = cayley b * cayley a := by
  have ha : (1 + a ^ 2) ≠ 0 := by positivity
  have hb : (1 + b ^ 2) ≠ 0 := by positivity
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [cayley, Matrix.mul_apply, Fin.sum_univ_two] <;> field_simp <;> ring

/-- Learned ("mixed") 2-D rotations: a feature pair turns by `R(wx)^col` and
`R(wy)^row` with independent learned `wx, wy`. Logits still depend only on
the 2-D offset between query and key. -/
theorem mixed_cayley_relative (wx wy : ℝ) (r₁ c₁ r₂ c₂ : ℤ) (q k : Fin 2 → ℝ) :
    let A : Matrix.orthogonalGroup (Fin 2) ℝ := ⟨cayley wx, cayley_mem_orthogonal wx⟩
    let B : Matrix.orthogonalGroup (Fin 2) ℝ := ⟨cayley wy, cayley_mem_orthogonal wy⟩
    (((A ^ c₁ * B ^ r₁ : Matrix.orthogonalGroup (Fin 2) ℝ) :
        Matrix (Fin 2) (Fin 2) ℝ) *ᵥ q) ⬝ᵥ
      (((A ^ c₂ * B ^ r₂ : Matrix.orthogonalGroup (Fin 2) ℝ) :
        Matrix (Fin 2) (Fin 2) ℝ) *ᵥ k) =
    q ⬝ᵥ (((A ^ (c₂ - c₁) * B ^ (r₂ - r₁) : Matrix.orthogonalGroup (Fin 2) ℝ) :
        Matrix (Fin 2) (Fin 2) ℝ) *ᵥ k) := by
  intro A B
  exact relative_logit (Subtype.ext (cayley_commute wx wy)) r₁ c₁ r₂ c₂ q k

end AlgebraicVision
