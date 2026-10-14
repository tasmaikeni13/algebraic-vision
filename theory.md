# Algebraic Vision: the ideas and equations behind v1 and v2

Algebraic Vision asks how much of a Vision Transformer can be expressed using
arithmetic, reciprocal square roots and comparisons. The standard ViT is the
reference; **v1** is the existing algebraic model; **v2** is the refined design
studied in Phase 1. V2 has research references and evidence, but is not yet a
registered production model. The existing models remain unchanged.

The distinction between the **initial v2 proposal** and the **selected Phase 1
revision** matters. Several attractive identities are correct without producing
a better numerical model. The final revision retains v1's E8 attention and
changes positions, pooling, the MLP gate and the classification loss. It does
not claim to solve E8's score attenuation or remove its attention square roots.

## 1. The shared Transformer

An image becomes patch tokens, with learned absolute position embeddings. Each
pre-normalized residual block has the form

$$
U=X+\operatorname{MultiHeadAttention}(\operatorname{LN}(X)),\qquad
X^+=U+W_2\,f(W_1\operatorname{LN}(U)).
$$

Biases and head concatenation are omitted here. LayerNorm itself is algebraic:
subtract the channel mean, multiply by
$\operatorname{rsqrt}(\operatorname{var}(x)+\epsilon)$, then apply learned
scale and bias. Matrix products use BF16 operands with FP32 accumulation;
sensitive nonlinear arithmetic and reductions use FP32.

The requested scales are 12 blocks, width 768 and 12 heads for ViT-B/16, and
24 blocks, width 1024 and 16 heads for ViT-L/16. Both use 64 channels per head
and an MLP width of four times the model width. Phase 1 studies these layer
counts with declared synthetic widths, and counts operations at the full
tensor shapes. It does not train full ViT-B/L models.

## 2. V1: replace the exponential with a radical link

For an order $n=2^m$, define

$$
\rho(t)=t+\sqrt{1+t^2},\qquad
E_n(s)=\rho(s/n)^n.
$$

This function is positive, satisfies $E_n(s)E_n(-s)=1$, and matches the
exponential through second order at zero. Its tails grow polynomially rather
than exponentially. Powers of two need only repeated squaring. A square root
can be computed as $a\operatorname{rsqrt}(a)$. On negative inputs, the
reciprocal identity
$\rho(t)=1/(\sqrt{1+t^2}-t)$ avoids cancellation.

V1 attention uses

$$
s_{ij}=q_i^\top k_j/\sqrt{d_h},\qquad
p_{ij}=\frac{E_8(s_{ij})}{\sum_k E_8(s_{ik})},\qquad
o_i=\sum_jp_{ij}v_j.
$$

The canonical model has no sink mass. The implementation also supports sink
ablations, which add a nonnegative mass to the denominator. Common scaling
of weights cancels during normalization; subtracting a common score does
**not** generally cancel as it does for softmax.

The key derivative is

$$
g_n(s)=\frac{d}{ds}\log E_n(s)
=\frac{1}{\sqrt{1+(s/n)^2}},\qquad
\frac{\partial p_{ij}}{\partial s_{ik}}
=p_{ij}(\delta_{jk}-p_{ik})g_8(s_{ik}).
$$

Thus $g_8(94)\approx0.085$: large scores reduce local sensitivity. This is a
property of the link, separate from saturation caused by an almost one-hot
normalized distribution.

V1's MLP uses the radical logistic gate

$$
\sigma_n(y)=\frac{1}{1+E_n(-y)},\qquad
\operatorname{rgelu}(x)=x\sigma_8(1.702x).
$$

Its positions combine learned absolute embeddings with learned planar Cayley
rotations. Its classifier uses a proper algebraic power score, described below.

## 3. V2 attention: what was proposed, and what survived

The initial proposal moves normalization outside the quadratic attention loop:

$$
\widetilde q=\frac{q}{\sqrt{d_h+\lVert q\rVert^2}},\qquad
\widetilde k=\frac{k}{\sqrt{d_h+\lVert k\rVert^2}},\qquad
u=\widetilde q^\top\widetilde k\in(-1,1).
$$

For $\kappa>|u|$, an exact identity gives

$$
E_n\!\left(\frac{2n\kappa u}{\kappa^2-u^2}\right)
=\left(\frac{\kappa+u}{\kappa-u}\right)^n.
$$

Consequently, $w=((\kappa+u)/(\kappa-u))^8$ needs no root per score.
Its log derivatives are
$16\kappa/(\kappa^2-u^2)$ in $u$, and
$-16u/(\kappa^2-u^2)$ in $\kappa$. For $|u|\le1<\kappa$, its weights
lie between $((\kappa-1)/(\kappa+1))^8$ and its reciprocal.

The proposed learned margin had an incorrect stated range. A valid bounded
replacement is $\kappa(t)=1.05+0.95/(1+t^2)$. Sharper margins, learned gains,
higher orders, column-energy factors and rational approximations were studied.
The optional column-balancing ablation uses
$C_j=N^{-1}\sum_iw_{ij}$, then normalizes
$w_{ij}/\sqrt{C_j+\epsilon}$ across keys.

The obstacle is the ball lift's derivative. Its radial eigenvalue is
$d_h/(d_h+\lVert q\rVert^2)^{3/2}$; its tangential eigenvalue is
$1/\sqrt{d_h+\lVert q\rVert^2}$. An apparently healthy derivative in
$u$ can therefore conceal a much smaller gradient in the original query.
Balancing also helps some clutter patterns while hurting cases where every
query wants the same sparse foreground.

**The selected revision retains E8 attention exactly.** It therefore matches
v1's attention gradients and finite-precision behavior by construction. The
root-free candidates and their failures remain part of the research record.

## 4. V2 positions: two independent spatial phases, without recurrence roots

The Cayley matrix is a rational rotation:

$$
R(w)=\frac{1}{1+w^2}
\begin{pmatrix}1-w^2&-2w\\2w&1-w^2\end{pmatrix}.
$$

V1 composes column and row rotations in each two-dimensional plane. A single
plane has only one phase; however, independent v1 planes can already have
joint spatial rank two. A single-plane null direction does not prove an
entire v1 head is spatially blind.

V2 computes powers through homogeneous coordinates:

$$
(v_0,u_0)=(1,0),\qquad
(v_{c+1},u_{c+1})=(v_c-wu_c,\ u_c+wv_c),
$$
$$
R(w)^c=\frac{1}{v_c^2+u_c^2}
\begin{pmatrix}v_c^2-u_c^2&-2u_cv_c\\2u_cv_c&v_c^2-u_c^2\end{pmatrix}.
$$

The norm recurrence is
$v_{c+1}^2+u_{c+1}^2=(1+w^2)(v_c^2+u_c^2)$, so the denominator stays
positive. The identity
$(v^2-u^2)^2+(2uv)^2=(v^2+u^2)^2$ proves exact-real orthogonality and
determinant one. Floating-point results have rounding error.

Channels are grouped into two planes in four dimensions. With
$\theta_x=2\arctan w_x$ and $\theta_y=2\arctan w_y$, their phases are

$$
\phi_+=c\theta_x+r\theta_y,\qquad
\phi_-=c\theta_x-r\theta_y.
$$

The phase Jacobian in coordinates $(c,r)$ has determinant
$-2\theta_x\theta_y$. The selected learned map

$$
w(t)=\frac1G\left(\frac14+\frac{3}{4(1+t^2)}\right)
$$

keeps both frequencies in $(1/(4G),1/G]$, preserving nonzero rank for all
finite parameters. Each grid-offset phase has magnitude below $4<2\pi$,
so periodic wrapping cannot alias a nonzero offset to the identity. Commuting
orthogonal generators also give the relative-position law
$R_{r_1,c_1}^{\top}R_{r_2,c_2}=R_{r_2-r_1,c_2-c_1}$.
Angles explain the mathematics; runtime tables use no trigonometric functions.
The relative-position law describes the rotation component. Retaining learned
absolute embeddings does not make the entire network translation invariant.

## 5. V2 pooling and the root-free MLP gate

The actual v1 baseline pools **after per-token LayerNorm**:
$h_{\rm v1}=N^{-1}\sum_i\operatorname{LN}(h_i)$. Comparing a new pool only
against raw, unnormalized averaging would misstate the baseline's robustness.

The initial Lorentz pool lifts unnormalized tokens:

$$
b_i=\frac{h_i}{\sqrt{d+\lVert h_i\rVert^2}},\quad
M=\sum_i b_i,\quad
h_{\rm Lorentz}=\frac{\sqrt d\,M}{\sqrt{N^2-\lVert M\rVert^2}}.
$$

It preserves identical tokens. With $H=\max_i\lVert h_i\rVert^2$,
$N^2-\lVert M\rVert^2\ge N^2d/(d+H)>0$. A positive-sum rearrangement
avoids subtractive cancellation near this boundary. Nevertheless, its angular
corruption exceeded the actual baseline in the outlier study.

Specifically, the same denominator can be written as the nonnegative sum
$N\sum_i d/(d+\lVert h_i\rVert^2)+N\sum_i\lVert b_i-\bar b\rVert^2$.

The selected replacement is the rational inverse-energy centroid:

$$
a_i=\frac{d}{d+\lVert h_i\rVert^2},\qquad
h_{\rm v2}=\operatorname{LN}\!\left(\frac{\sum_i a_i h_i}{\sum_i a_i}\right).
$$

Its denominator is positive and identical tokens are fixed points. With other
tokens fixed, an isolated outlier's weighted contribution decreases as
$O(1/\lVert h_i\rVert)$, with pool derivative
$O(1/\lVert h_i\rVert^2)$. This is a changed pooling objective, with measured
robustness on the declared corruption suite, rather than a universal guarantee
for every possible collection of tokens.

The selected MLP gate is

$$
f_c(x)=\frac x2\left(1+\frac{cx}{1+c|x|}\right),\qquad c=0.851.
$$

It uses no roots. It is twice continuously differentiable, satisfies
$f_c(x)-f_c(-x)=x$, and matches rgelu's value, slope and curvature at zero:
$0,1/2,0.851$. Its derivative is

$$
f_c'(x)=
\begin{cases}
1-\dfrac{1}{2(1+cx)^2},&x\ge0,\\
\dfrac{1}{2(1-cx)^2},&x<0.
\end{cases}
$$

Thus $0<f_c'(x)<1$ for finite inputs. Its negative tail approaches
$-1/(2c)$. These properties do not make it identical to GELU or establish
full-size model accuracy; the gate is also evaluated in synthetic training.

## 6. Classification: a real algebraic objective and its true gradient

For logits $z$, define a decoupled radical probability link

$$
p_c=\frac{\rho(z_c/\tau)^n}{\sum_k\rho(z_k/\tau)^n},\qquad
b_c=\frac{n/\tau}{\sqrt{1+(z_c/\tau)^2}}.
$$

V1 uses $(n,\tau)=(64,64)$. The selected v2 head uses $(8,4)$, giving
stronger curvature at smaller logit scales. Both train with the strictly
proper power score

$$
L_a(p,y)=\frac1a\sum_cp_c^a-\frac1{a-1}\sum_cy_cp_c^{a-1},
$$

up to a target-only constant. V1 takes $a=63/64$; v2 takes $a=7/8$.
Labels are smoothed as $y=(1-\epsilon)y_{\rm onehot}+\epsilon/K$.
Strict propriety follows from the associated convex generator having
$\Phi''(p)=p^{a-2}>0$: expected loss is uniquely minimized at the target
distribution on the positive simplex.

For $T=\sum_j(p_j^a-y_jp_j^{a-1})$, the actual logit gradient is

$$
\frac{\partial L_a}{\partial z_c}
=b_c\left(p_c^a-y_cp_c^{a-1}-p_cT\right).
$$

V2 differentiates this genuine objective. Fractional powers are evaluated
from the radical base and the row normalizer, avoiding roots of probabilities
that have already underflowed. V1 needs six sequential roots of the normalizer;
v2 needs three. They are roots **per row**, not six versus three per class.

The initially proposed exact cross-entropy VJP,
$b_c(p_c-y_c)$, is conservative, but its exact forward loss contains a
logarithm and violates the pure algebraic training-graph requirement. Replacing
the outside inverse square root with a quarter power generally introduces
nonzero curl. Neither a fake CE value nor that modified gradient is selected.
Brier loss and the existing ALU gate remain documented ablations.
The latter is $\operatorname{ALU}_c(x)=x(1+cx/\sqrt{1+c^2x^2})/2$.

## 7. What the evidence establishes

| Component | V1 | Selected Phase 1 v2 |
| --- | --- | --- |
| Attention | Normalized E8 | Same E8 expression and precision policy |
| Positions | Mixed planar Cayley powers with renormalization | Bounded, learned rank-two Bi-Cayley; homogeneous powers |
| MLP | rgelu | Rational softsign gate |
| Pool | Mean of per-token LayerNorm | Inverse-energy centroid, then LayerNorm |
| Head | E64 link, power score 63/64 | Decoupled E8 link, power score 7/8 |

The allowed runtime operations are $+, -, \times, /, \operatorname{rsqrt}$,
min/max, comparisons and selection, plus their matrix/reduction forms.
Integer powers above mean explicit multiplication, not a generic `pow`.
Entropy plots and symbolic oracles may use logarithms and trigonometry as
diagnostics; those functions are excluded from algebraic training graphs.

Signal studies track feature variance, mean patch cosine and the attention
spectral gap $1-|\lambda_2(P)|$. Backward sensitivity uses singular values
of the composed layer Jacobian $J_L\cdots J_1$, with finite-depth Lyapunov
exponents $L^{-1}\log\sigma_i$. The empirical neural tangent kernel is
$K=J_\theta f\,(J_\theta f)^\top$; its spectrum describes local parameter
conditioning. These are diagnostics, not substitutes for task accuracy.

Theoretical cost comparisons include forward values, all input and learned
frequency VJPs, every recurrence trip, and a common optimizer convention.
The selected design saves roots mainly in the MLP and positional tables;
**E8 still has quadratic attention root work**. Compiler operation estimates
are reported separately from integer source-program counts. Neither is a
TPU latency measurement.

[The Phase 1 report](results/phase1/REPORT.md) contains the rejected candidates,
Monte Carlo results, synthetic convergence and NTK studies, precision errors,
cost assumptions and gate decision. [The formal suite](formal/README.md)
establishes the stated real-algebraic identities and bounds without `sorry`.
These establish mathematical and declared research properties, not ImageNet
accuracy or full-model optimization dominance. Production integration, TPU
kernels and training runs belong to later phases and have not been started.
