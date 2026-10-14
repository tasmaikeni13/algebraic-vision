# AUTONOMOUS RESEARCH & ENGINEERING DIRECTIVE: ALGEBRAIC VISION v2 (`algebraic_v2`)

## 0. MISSION, CONSTRAINTS & OPERATIONAL AUTONOMY

You are an autonomous Principal Machine Learning Researcher, Formal Verification Mathematician, and TPU Kernel Engineer operating inside the `algebraic-vision` repository.

### Core Mandates
1. **Preserve Existing Baselines Untouched:** Keep the `standard` ViT and the original `algebraic` ViT (`v1`) completely intact as frozen reference baselines. Introduce the new architecture strictly as a third experimental arm (`algebraic_v2`).
2. **Do Not Ask for Implementation Hand-Holding:** Inspect the workspace autonomously to understand its abstractions, configuration schemas, training loops, Pallas kernels, symbolic scripts, Lean 4 formalizations, and multi-host TPU launchers. Design your own module boundaries, code structure, and test harnesses.
3. **The Strict Algebraic Purity Constraint:** Every forward pass, backward pass (VJP), and training loss operation in `v1` and `v2` must compile to TPU v4 HLO using **only** `{+, -, *, /, rsqrt, min, max, comparisons, select}`. Zero transcendental instructions (`exp`, `log`, `sin`, `cos`, `tanh`, `logistic`, `erf`, `pow`) are permitted anywhere in the compiled training step graph.
4. **Hardware Target:** You are operating on a **Google Cloud TPU v4-32 Pod** (4 hosts, 4 TPU v4 chips / 8 TensorCores per host = 16 chips / 32 TensorCores total). Every training, sweep, and evaluation pipeline must be engineered to saturate all 16 chips across all 4 hosts via multi-host data/model parallelism, `bfloat16` MXU matmuls, `float32` accumulation, and high-throughput in-memory/sharded data pipelines.
5. **Execution Stop-Gates (Phases 4 & 5):** You will autonomously execute Phases 1, 2, and 3 end-to-end (including symbolic proofs, Monte Carlo simulations, Lean 4 builds, unit tests, Pallas kernel benchmarks, and 1-step multi-host TPU smoke tests). For **Phase 4 (Hyperparameter Sweeps)** and **Phase 5 (Main Runs)**, you must build, validate, and smoke-test all configurations and launch scripts, but **DO NOT launch the multi-hour full sweeps or main training runs**—the human operator will trigger those personally.

---

## 1. ARCHITECTURAL BASELINES & THE `v2` CANDIDATE FORMULATION

### 1.1 Existing Baselines (`standard` vs. `v1`)
- **`standard` ViT:** Softmax attention $\operatorname{softmax}(q_i^\top k_j / \sqrt{d_h})$, GELU MLP gate, learned absolute 1D position embeddings, Global Average Pooling (GAP), and softmax cross-entropy loss with label smoothing $\epsilon_{\text{ls}} = 0.1$.
- **`algebraic_v1` ViT:**
  - **Attention:** Unnormalized score $s_{ij} = \langle q_i, k_j \rangle / \sqrt{d_h}$ passed into the radical exponential $E_8(s_{ij}) / \sum_k E_8(s_{ik})$, where $E_n(s) = \left(s/n + \sqrt{1 + s^2/n^2}\right)^n$.
  - **MLP Gate:** Radical GELU $\operatorname{rgelu}(x) = x \, \sigma_8(1.702 x)$ with $\sigma_n(y) = (1 + E_n(-y))^{-1}$.
  - **Positions:** Learned absolute embeddings + commuting 2D planar Cayley rotations $R(w_i^x)^c R(w_i^y)^r \in SO(2)$ with `rsqrt` renormalization, where $R(w) = \frac{1}{1+w^2}\begin{pmatrix} 1-w^2 & -2w \\ 2w & 1-w^2 \end{pmatrix}$.
  - **Pooling:** Flat Euclidean Global Average Pooling $\bar{h} = \frac{1}{N}\sum_{i=1}^N h_i$ after LayerNorm.
  - **Loss:** Strictly proper power score $L_a(p, y) = \frac{1}{a}\sum_c p_c^a - \frac{1}{a-1}\sum_c y_c p_c^{a-1}$ with $a = 63/64$ (computed via 6 nested `rsqrt` calls) on the link $p_c = E_{64}(z_c) / \sum_k E_{64}(z_k)$.

### 1.2 Diagnosed Mathematical & Hardware Bottlenecks in `v1`
1. **Middle-Block Score Runaway & Gradient Attenuation:** In trained `v1` ViT-B/16 models, unnormalized attention scores grow to median $\vert{}s\vert{} \approx 94$ in blocks 2–5. Because the logarithmic derivative of $E_8(s)$ is $g_8(s) = (1 + s^2/64)^{-1/2}$, a score of $\vert{}s\vert{} = 94$ attenuates the attention Jacobian by $g_8(94) \approx 0.085$ (killing $>91\%$ of the gradient magnitude).
2. **$O(N^2)$ `rsqrt` and Branching Overhead inside Pallas:** Evaluating $E_8(s_{ij})$ requires $N^2 = 38{,}809$ `rsqrt` and sign-selection operations per head inside the inner loop, making the `v1` Pallas kernel slower than standard Softmax ($1.132\text{ ms}$ vs. $0.891\text{ ms}$).
3. **1D Diagonal Null-Line Degeneracy in 2D Cayley Rotations:** Because planar rotations $R(w_i^x)^{\Delta c} R(w_i^y)^{\Delta r} \in SO(2)$ act on a 1-parameter Lie group (Lie rank 1), any 2D patch offset $(\Delta r, \Delta c) \neq (0, 0)$ satisfying $\Delta c \arctan w_i^x + \Delta r \arctan w_i^y = 0 \pmod\pi$ receives the identity rotation $R = I_2$, creating directional blind spots along diagonal lines. Furthermore, computing grid powers via repeated `rsqrt` renormalization incurs unnecessary XLU overhead.
4. **High-Norm Background Patch Artifacts in Flat Pooling:** Without register tokens, deeper ViT blocks store high-norm global scratchpad artifacts in redundant background patches ($\Vert{}h_i\Vert{}_2 \gg \sqrt{d}$), which corrupt flat Euclidean Global Average Pooling.
5. **Deactivated Logit Regularization & Nested-Root Latency in the $E_{64}$ Loss:** Using $E_{64}(z_c)$ pushes the hyperbolic curvature radius to $\tau = 64$, so $g_{64}(z) = (1 + z^2/4096)^{-1/2} \approx 1$ provides virtually zero logit-norm regularization across normal logit scales while requiring 6 sequential nested `rsqrt` calls per class.

### 1.3 Candidate `algebraic_v2` Mathematical Specification
Test and refine the following four candidate upgrades for `algebraic_v2`:

#### Upgrade A: Weierstrass–Cayley Rational Attention Kernel (plus Optional Column Pre-Conditioning)
- **Step A1 — $O(N d_h)$ Beltrami–Klein Hyperbolic Ball Lift (Outside the $O(N^2)$ Loop):**
  Project queries and keys into the open unit ball using only $2N$ `rsqrt` calls per head (with $c = \sqrt{d_h}$):
  $$\tilde{q}_i = \frac{q_i}{\sqrt{d_h + \Vert{}q_i\Vert{}_2^2}}, \qquad \tilde{k}_j = \frac{k_j}{\sqrt{d_h + \Vert{}k_j\Vert{}_2^2}}, \qquad u_{ij} = \langle R_{r_i, c_i} \tilde{q}_i,\, R_{r_j, c_j} \tilde{k}_j \rangle \in (-1, 1).$$
  *(Optionally include a learned per-channel or per-head pre-projection gain $g_q, g_k$ so the head can modulate its radial saturation closer to the boundary $\Vert{}u_{ij}\Vert{} \to 1$ when sharp attention is needed.)*
- **Step A2 — $O(N^2)$ `rsqrt`-Free Weierstrass–Cayley Kernel (Inside Pallas):**
  By the exact algebraic identity $E_n\!\left(\frac{2n\kappa u}{\kappa^2 - u^2}\right) \equiv \left(\frac{\kappa + u}{\kappa - u}\right)^n$, evaluate the unnormalized weight with a learned per-head margin $\kappa_h = 1 + \frac{1}{\rho_1(\tau_h)} \in (1.05, 2]$ (initialized at $\kappa_h = 2$, using 3 squarings and 1 division for $n=8$):
  $$w_{ij} = \left(\frac{\kappa_h + u_{ij}}{\kappa_h - u_{ij}}\right)^8.$$
- **Step A3 — Optional One-Step Biconjugate Column Balancing (Test as an Ablation Variant):**
  To prevent redundant background patches from dominating the denominator, test pre-balancing key columns by their inverse root mean energy $C_j = \frac{1}{N}\sum_{\ell=1}^N w_{\ell j}$:
  $$\hat{w}_{ij} = \frac{w_{ij}}{\sqrt{C_j + \epsilon}}, \qquad p_{ij} = \frac{\hat{w}_{ij}}{\sum_{k=1}^N \hat{w}_{ik}}.$$
  *(Note: Evaluate in Phase 1 whether Step A3 improves background-clutter SNR enough to justify the two-pass column reduction over pure row-streaming Step A2.)*

#### Upgrade B: `rsqrt`-Free Projective Powers + Rank-2 $SO(4)$ Bi-Cayley 2D Positional Encoding
- **Part B1 — Exact Homogeneous Projective Polynomial Recurrence (Zero `rsqrt`):**
  For any scalar frequency $w$ and grid index $c \in \{0, \dots, G-1\}$, compute homogeneous projective coordinates $[u_c : v_c] \in \mathbb{R}\mathbb{P}^1$ via the exact recurrence $(v_0, u_0) = (1, 0)$ and $(v_c, u_c) = (v_{c-1} - w u_{c-1},\; u_{c-1} + w v_{c-1})$ (applying a pure power-of-two scale factor $2^{-k}$ if $\vert{}v_c\vert{} + \vert{}u_c\vert{}$ grows large at high resolutions), and project once onto $SO(2)$:
  $$R(w)^c = \frac{1}{v_c^2 + u_c^2} \begin{pmatrix} v_c^2 - u_c^2 & -2 u_c v_c \\ 2 u_c v_c & v_c^2 - u_c^2 \end{pmatrix}.$$
  By the polynomial identity $(v_c^2 - u_c^2)^2 + (2u_c v_c)^2 \equiv (v_c^2 + u_c^2)^2$, $R(w)^c \in SO(2)$ is machine-exact with **zero square roots**.
- **Part B2 — Rank-2 Commuting Isoclinic $SO(4)$ Coupling (Eliminating Diagonal Null-Lines):**
  Group channels into 4D blocks $\mathbf{x} \in \mathbb{R}^4$ (or equivalently, pair two orthogonal 2D planes via a fixed orthogonal Hadamard/butterfly basis) so that one generator rotates by $R(w_1^x)^c R(w_1^y)^r$ and the commuting orthogonal generator rotates by $R(w_2^x)^c R(w_2^y)^{-r}$. Because $SO(2)_L \times SO(2)_R \subset SO(4)$ has **Lie rank 2**, $R_{4\text{D}}(\Delta r, \Delta c) = I_4 \iff (\Delta r, \Delta c) = (0, 0)$ locally, eliminating the 1D diagonal null-line while preserving exact 2D relative shift invariance $R_{4\text{D}}(r_1, c_1)^\top R_{4\text{D}}(r_2, c_2) = R_{4\text{D}}(r_2 - r_1, c_2 - c_1)$.

#### Upgrade C: Pre-Norm Lorentzian Geodesic Centroid Patch Pooling
- Replace flat Global Average Pooling with the closed-form **Einstein–Lorentz Geodesic Midpoint** applied to the **unnormalized final residual stream** $h_1, \dots, h_N \in \mathbb{R}^d$ *before* the final LayerNorm (so LayerNorm does not erase patch norm differences prior to outlier attenuation):
  $$w_i = \frac{1}{\sqrt{d + \Vert{}h_i\Vert{}_2^2}}, \qquad M = \sum_{i=1}^N w_i h_i, \qquad h_{\text{pool}} = \frac{\sqrt{d}\, M}{\sqrt{N^2 - \Vert{}M\Vert{}_2^2}},$$
  followed by the final LayerNorm and linear classification head.

#### Upgrade D: Decoupled-Curvature Exact Algebraic MLE Head (Conservative Custom VJP)
- Replace the $a = 63/64$ power loss on $E_{64}(z_c)$ with a decoupled curvature radius $\tau = 2$ (or $\tau = 4$) and exponent $n = 8$:
  $$\rho_\tau(z) = \frac{z}{\tau} + \sqrt{1 + \left(\frac{z}{\tau}\right)^2}, \qquad p_c = \frac{\rho_\tau(z_c)^n}{\sum_{k=1}^K \rho_\tau(z_k)^n}.$$
- Use a custom VJP that returns the **exact conservative gradient** of the cross-entropy objective $-\sum_c y_c \log p_c$ (with label-smoothed targets $y_c$) using pure arithmetic and a single `rsqrt`:
  $$\frac{\partial L}{\partial z_c} = \frac{(n/\tau)(p_c - y_c)}{\sqrt{1 + (z_c / \tau)^2}}.$$
  *(Critical Mathematical Note: Compare the exact $-1/2$ exponent $\left(1 + (z_c/\tau)^2\right)^{-1/2}$ against the non-conservative $-1/4$ Riemannian preconditioned exponent $\left(1 + (z_c/\tau)^2\right)^{-1/4}$ in Phase 1; verify Poincaré integrability and optimization stability before selecting the winner.)*
- Also evaluate whether the MLP gate should remain `rgelu(x)` or use a safe coupled algebraic gate for `ViT-L/16`.

---

## 2. FIVE-PHASE EXECUTION PLAN

### PHASE 1: Exhaustive Breadth-First Mathematical, Statistical, Monte Carlo & Formal Analysis (With Autonomous Self-Correction Loop)

Before writing any production training code, build an automated verification and simulation suite comparing `standard`, `algebraic_v1`, and `algebraic_v2` across five orthogonal axes:

1. **Axis 1 — Symbolic & Exact Real-Algebraic Analysis (`SymPy` + 50-digit `mpmath`):**
   - Prove the Weierstrass–Cayley rational equivalence $E_n\!\left(\frac{2n\kappa u}{\kappa^2 - u^2}\right) \equiv \left(\frac{\kappa + u}{\kappa - u}\right)^n$, compute its Taylor expansion around $u=0$, its asymptotic range $\left[\left(\frac{\kappa-1}{\kappa+1}\right)^n, \left(\frac{\kappa+1}{\kappa-1}\right)^n\right]$, and its exact logarithmic derivative with respect to $q_i, k_j, \kappa_h$.
   - Verify the polynomial identity $(v_c^2 - u_c^2)^2 + (2u_c v_c)^2 \equiv (v_c^2 + u_c^2)^2$ for the projective recurrence and prove exact 2D shift-invariance and Rank-2 local injectivity ($\det J_{\text{pos}}(\Delta r, \Delta c) \neq 0$) of the $SO(4)$ Bi-Cayley rotation across the 2D patch grid.
   - Verify the Lorentzian Geodesic Centroid fixed-point identity ($h_1 = \dots = h_N = h \implies h_{\text{pool}} \equiv h$), strict timelike positivity $N^2 - \Vert{}M\Vert{}_2^2 > 0$, and outlier attenuation rate $\frac{\partial h_{\text{pool}}}{\partial h_i} = O(\Vert{}h_i\Vert{}_2^{-1})$ as $\Vert{}h_i\Vert{}_2 \to \infty$.
   - Check the Poincaré integrability condition ($\frac{\partial g_c}{\partial z_k} = \frac{\partial g_k}{\partial z_c}$) and strict propriety of the decoupled MLE head VJP.

2. **Axis 2 — Deep Signal Propagation, Mean-Field & Spectral Theory ($L=12$ for ViT-B, $L=24$ for ViT-L):**
   - Track the forward feature variance map $q^{(l+1)} = \mathcal{F}(q^{(l)})$, patch cosine-similarity map $c^{(l+1)} = \mathcal{G}(c^{(l)})$, and Perron–Frobenius spectral gap $1 - \vert{}\lambda_2(P)\vert{}$ across 12 and 24 layers to measure resistance to oversmoothing / rank collapse.
   - Compute the backward Layer-to-Layer Jacobian Singular Value Spectrum (dynamical isometry / Lyapunov exponents) across $L=12$ and $L=24$ layers.
   - **The Figure 5 Score-Runaway Stress Test:** Plot attention entropy $H(p)/\log N$, fraction of saturated rows ($\Vert{}J_{\text{attn}}\Vert{}_F < 10^{-4}$), and expected gradient norm $\Vert{}\partial L / \partial Q\Vert{}_F$ as pre-projection query-key norms scale across $\alpha \in [2^0, 2^8]$. Verify whether `v2` eliminates the middle-block gradient attenuation ($\vert{}s\vert{} \approx 94 \implies g_8(94) = 0.085$) observed in `v1`.

3. **Axis 3 — High-Dimensional Monte Carlo Stress Suites:**
   - **Needle-in-a-Patch-Haystack (Background Crowding) Monte Carlo:** Simulate $N \in \{197, 256\}$ patches where $N - k$ patches are correlated background clutter and $k \in \{1, 3, 5\}$ patches carry foreground signal. Measure foreground attention mass recovery and backward gradient Signal-to-Noise Ratio (SNR) across `standard`, `v1`, `v2 (no col-balance)`, and `v2 (with col-balance)`.
   - **Register-Artifact Outlier Corruption Monte Carlo:** Inject synthetic high-norm background scratchpad tokens ($\Vert{}h_{\text{outlier}}\Vert{}_2 \in [5\sqrt{d}, 100\sqrt{d}]$) into the final layer and measure the angular corruption of the pooled representation under `standard`/`v1` flat GAP vs. `v2` Pre-Norm Lorentzian Geodesic Centroid.
   - **IEEE-754 Finite-Precision (`bfloat16` / `float32`) Perturbation Monte Carlo:** Against 50-digit `mpmath` / `float64` ground truth, measure forward Total Variation error and backward relative gradient error across input scales $\alpha \in [1, 100]$, explicitly testing boundary regimes ($u_{ij} \to \pm 1$, $\kappa_h \to 1.05$, and $N^2 - \Vert{}M\Vert{}_2^2 \to 0$).
   - **Micro-Convergence & Empirical NTK Conditioning:** Train 2-to-4 layer synthetic vision/associative prototypes across learning rates $\text{lr} \in [10^{-4}, 10^{-2}]$ to verify that `v2` preserves or expands `v1`'s Maximum Stable Learning Rate (MSLR) while accelerating convergence.

4. **Axis 4 — Formal Verification in Lean 4 (`formal/`):**
   - Extend the Lean 4 proof suite over $\mathbb{R}$ to formally prove the core `v2` theorems (without breaking any existing `v1` theorems):
     - Rational Weierstrass–Cayley identity, positivity, and explicit upper/lower weight floor bounds.
     - Projective recurrence orthogonality ($\det R = 1, R^\top R = I$) and 2D relative offset invariance of $SO(4)$ Bi-Cayley rotations.
     - Strict positivity of the Lorentzian Centroid denominator $N^2 - \Vert{}M\Vert{}_2^2 \ge \frac{N^2 d}{d + \max_i \Vert{}h_i\Vert{}_2^2} > 0$.
   - Ensure `lake build` succeeds with zero `sorry` axioms.

5. **Axis 5 — THE SELF-CORRECTION LOOP (MANDATORY):**
   - **Gate Criterion:** `algebraic_v2` must strictly match or outperform `algebraic_v1` on (a) gradient preservation at high pre-activation scale ($\alpha \in [10, 100]$), (b) finite-precision `bfloat16` stability, (c) spatial positional injectivity, (d) outlier pooling robustness, and (e) per-step theoretical FLOP / `rsqrt` count.
   - **If ANY `v2` component underperforms `v1` or exhibits a numerical/optimization flaw** (for instance, if the Klein-ball projection bounds maximum attention sharpness too tightly when $\kappa_h = 2$ without pre-norm learned QK scale gains, or if column balancing degrades causal/row-local variance), **you must autonomously diagnose the mathematical root cause, formulate an improved algebraic equation (strictly within the zero-transcendental constraint), and re-run Axes 1–4 until `v2` provably dominates `v1`.** Document every iteration and its quantitative telemetry in a Phase 1 analytical report.

---

### PHASE 2: Multi-Scale Model & Dataset Engineering (`ViT-B/16` & `ViT-L/16`)

Once `algebraic_v2` passes all Phase 1 analytical and formal gates:
1. **Implement `algebraic_v2`** cleanly alongside `standard` and `algebraic` (`v1`) so any experiment can select `arm in {"standard", "algebraic", "algebraic_v2"}`.
2. **Scale Configurations:**
   - **`ViT-B/16` (~86M parameters):** 12 layers, width $d = 768$, 12 heads ($d_h = 64$), MLP width $3072$, patch size $16 \times 16$ at $224 \times 224$ resolution ($N = 196$ or $197$ patches).
   - **`ViT-L/16` (~304M parameters):** 24 layers, width $d = 1024$, 16 heads ($d_h = 64$), MLP width $4096$, patch size $16 \times 16$ at $224 \times 224$ resolution.
3. **Appropriate Datasets & Training Regimes per Scale:**
   - Configure the data pipelines, augmentations, regularization, and token budgets appropriately for both `ViT-B/16` (ImageNet-1k with matched paired seeds, deterministic random-resized crops/flips, and minival split) and `ViT-L/16` (supporting higher-capacity ImageNet-1k / ImageNet-21k or strong-augmentation regimes such as RandAugment/Mixup/CutMix/stochastic depth where appropriate for 304M parameters so `ViT-L/16` does not severely overfit, while keeping the comparison 100% matched and fair across all 3 arms for every seed).
4. **HLO Purity Audit:** Run the TPU compiler HLO instruction scanner on both `ViT-B/16` and `ViT-L/16` for `algebraic_v1` and `algebraic_v2`. Confirm **0 transcendental instructions** and record the exact `rsqrt` instruction reduction achieved by `v2` over `v1`.

---

### PHASE 3: FlashAttention & Algebraic Pallas TPU v4 Kernels

Engineer fused Pallas TPU v4 attention kernels (forward pass and analytic backward pass) for all three architectures (`standard` FlashAttention, `algebraic_v1`, and `algebraic_v2`) supporting both `ViT-B/16` and `ViT-L/16` shapes:
1. **Exploit `v2`'s Hardware Advantage:** Move the Klein-ball `rsqrt` and 2D projective Bi-Cayley rotations outside the $O(N^2)$ Pallas loop into $O(N d_h)$ pre-processing so the inner Pallas loop executes pure MXU `bfloat16` matmuls plus 3 squarings and 1 division per score tile—with zero `rsqrt` instructions inside the $N \times N$ kernel.
2. **Pragmatic Performance Target:** Do not obsess over theoretical micro-optimizations; as long as the Pallas kernels are numerically verified against dense JAX references (forward and backward in `float32`/`bfloat16`), satisfy the HLO purity audit, and achieve wall-clock latency matching `standard` (or within a small margin), they pass Phase 3.

---

### PHASE 4: Budgeted Hyperparameter Sweeps (Build & Smoke-Test Only)

Prepare the complete budgeted hyperparameter sweep infrastructure for **all 3 architectures (`standard`, `algebraic_v1`, `algebraic_v2`) across 3 paired seeds (`42, 43, 44`)** for both `ViT-B/16` and `ViT-L/16`:
1. Ensure identical data order, augmentation streams, and shared parameter initialization across all 3 arms for each seed.
2. Configure independent learning-rate / weight-decay candidate grids selected strictly on the held-out `minival` split, with automated fairness verification scripts confirming that paired runs differ only in arm-defining fields and selected hyperparameters.
3. Optimize multi-host execution for the **TPU v4-32 pod** (4 hosts, 16 chips).
4. **Verification Gate:** Run 1-to-5 step distributed smoke tests across the TPU v4-32 pod for all 3 architectures at both `ViT-B/16` and `ViT-L/16` scales to verify compilation, memory footprint, throughput (img/s), and checkpoint/selection serialization. **STOP after smoke verification; do not launch the full hyperparameter sweeps.**

---

### PHASE 5: Full 3-Arm × 3-Seed Main Run Pipelines (`ViT-B/16` & `ViT-L/16` — Build & Smoke-Test Only)

Prepare the production launchers, resumable checkpointing, and post-training evaluation/reporting pipelines for the main runs:
1. **Stage 5A (`ViT-B/16` Main Matrix):** 3 architectures (`standard`, `algebraic_v1`, `algebraic_v2`) $\times$ 3 paired seeds (`42, 43, 44`).
2. **Stage 5B (`ViT-L/16` Main Matrix):** 3 architectures (`standard`, `algebraic_v1`, `algebraic_v2`) $\times$ 3 paired seeds (`42, 43, 44`).
3. Include automated post-run analysis scripts computing validation Top-1, NLL, Expected Calibration Error (ECE), paired seed differences, 95% confidence intervals, one-sided paired $t$-tests, per-block attention entropy / score magnitude diagnostics, and publication-ready figures.
4. **Verification Gate:** Smoke-test the Phase 5 launch and evaluation scripts end-to-end on synthetic/short steps across the TPU v4-32 pod, then output the exact copy-paste shell commands for the human operator to execute Phase 4 and Phase 5 personally.
