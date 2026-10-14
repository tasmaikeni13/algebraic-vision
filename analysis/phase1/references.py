"""Differentiable research references, preserving the frozen baseline code.

All v2 primitives use arithmetic, rsqrt, comparisons and selects. Reductions,
matrix products and shape operations have their usual structural meaning.
FP32 accumulation is explicit when the inputs are BF16. No custom surrogate
derivatives are used: gradients are derivatives of the actual forward value.
"""

import jax
import jax.numpy as jnp


def work(x):
    x = jnp.asarray(x)
    return x.astype(jnp.promote_types(x.dtype, jnp.float32))


def square_power(x, squarings=3):
    for _ in range(squarings):
        x = x * x
    return x


def rho(x):
    x = work(x)
    r = jax.lax.rsqrt(1.0 + x * x)
    # Both branches are well defined; conjugation avoids negative-tail loss.
    return jnp.where(x >= 0, x + (1 + x * x) * r,
                     r / (1 - jnp.minimum(x * r, 0)))


def radical_rows(scores, order=8):
    base = rho(scores / order)
    base = base / jnp.max(base, axis=-1, keepdims=True)
    weights = square_power(base, order.bit_length() - 1)
    return weights / jnp.sum(weights, axis=-1, keepdims=True)


def ball_lift(q, gain=1.0):
    q = work(q) * gain
    return q * jax.lax.rsqrt(q.shape[-1] + jnp.sum(q * q, -1, keepdims=True))


def rational_rows(u, kappa=2.0, balance=False):
    weights = square_power((kappa + u) / (kappa - u))
    if balance:
        weights = weights * jax.lax.rsqrt(jnp.mean(weights, -2, keepdims=True) + 1e-6)
    return weights / jnp.sum(weights, axis=-1, keepdims=True)


def attention(q, k, arm, kappa=2.0, gain=1.0):
    if arm in ("v2", "v2_balanced"):
        q, k = ball_lift(q, gain), ball_lift(k, gain)
        u = jnp.einsum("...id,...jd->...ij", q, k)
        return rational_rows(u, kappa, arm == "v2_balanced")
    s = jnp.einsum("...id,...jd->...ij", work(q), work(k)) * q.shape[-1] ** -0.5
    if arm == "v2_tail":
        return rational_tail_rows(s)
    return jax.nn.softmax(s, -1) if arm == "standard" else radical_rows(s)


def rational_tail_rows(s):
    """Self-correction experiment: unbounded scores, no inner-loop roots.

    f(t)=1+t+t²/(2+t), t>=0; use 1/f(-t) on the negative branch.
    Positive, reciprocal, matching E8 through second order at zero. This
    avoids the ball lift's radial derivative O(alpha^-3); it is an ablation,
    not a claim of uniform superiority over the original radical kernel.
    """
    t = s/8
    a = jnp.maximum(t, -t)
    f = 1+a+a*a/(2+a)
    # Keep the linear positive term signed: differentiating max(t,-t) at
    # t=0 would otherwise return zero instead of the true unit slope.
    # Both branches are finite for every finite t in the tested range.
    base = jnp.where(t >= 0, 1+t+a*a/(2+a), 1/f)
    base = base/jnp.max(base, -1, keepdims=True)
    w = square_power(base)
    return w/jnp.sum(w, -1, keepdims=True)


def constrained_margin(t):
    # The proposed 1+1/rho(t) is not bounded below by 1.05 or above by 2.
    # This rational map attains 2 at zero and approaches 1.05 as |t| grows.
    return 1.05 + 0.95 / (1.0 + t * t)


def projective_tables(w, length):
    """Homogeneous powers; common binary scaling preserves projective value."""
    w = work(w)

    def step(carry, _):
        v, u = carry
        denom = v * v + u * u
        c, s = (v * v - u * u) / denom, 2 * u * v / denom
        vn, un = v - w * u, u + w * v
        # Input frequencies for grids are bounded; one binary scale suffices
        # for each step. The condition uses only max/comparison, not abs HLO.
        mag = jnp.maximum(vn, -vn) + jnp.maximum(un, -un)
        scale = jnp.where(mag > 16.0, 0.0625, 1.0)
        return (vn * scale, un * scale), (c, s)

    _, (c, s) = jax.lax.scan(step, (jnp.ones_like(w), jnp.zeros_like(w)),
                              None, length=length)
    return c, s


def bicayley(x, grid, wx=None, wy=None):
    """Tied positive frequencies; phases (c a+r b, c a-r b).

    Frequencies <= 1/(4G) prevent wrapping on a G by G grid. This is an
    explicit sufficient condition, not an assertion about arbitrary SO(4).
    Extra non-grid tokens, if any, retain the identity rotation.
    """
    if x.shape[-1] % 4:
        raise ValueError("Bi-Cayley needs a multiple of four channels")
    groups = x.shape[-1] // 4
    wx = jnp.full((groups,), 1.0 / (4 * grid), x.dtype) if wx is None else wx
    wy = jnp.full((groups,), 1.0 / (5 * grid), x.dtype) if wy is None else wy
    cx, sx = projective_tables(wx, grid)
    cy, sy = projective_tables(wy, grid)
    rows, cols = jnp.arange(grid * grid) // grid, jnp.arange(grid * grid) % grid
    cp, sp = cx[cols] * cy[rows] - sx[cols] * sy[rows], sx[cols] * cy[rows] + cx[cols] * sy[rows]
    cm, sm = cx[cols] * cy[rows] + sx[cols] * sy[rows], sx[cols] * cy[rows] - cx[cols] * sy[rows]
    extra = x.shape[-2] - grid * grid
    if extra < 0:
        raise ValueError("grid has more cells than tokens")
    if extra:
        ones, zeros = jnp.ones((extra, groups), x.dtype), jnp.zeros((extra, groups), x.dtype)
        cp, sp = jnp.concatenate([cp, ones]), jnp.concatenate([sp, zeros])
        cm, sm = jnp.concatenate([cm, ones]), jnp.concatenate([sm, zeros])
    a, b, c, d = jnp.split(work(x), 4, -1)
    return jnp.concatenate([cp*a-sp*b, sp*a+cp*b, cm*c-sm*d, sm*c+cm*d], -1)


def lorentz_pool(h, stable=True):
    h = work(h)
    n, d = h.shape[-2:]
    norms = d + jnp.sum(h * h, -1, keepdims=True)
    b = h * jax.lax.rsqrt(norms)
    m = jnp.sum(b, -2)
    if stable:
        centered = b - jnp.mean(b, -2, keepdims=True)
        denominator = n * (jnp.sum(d / norms, (-2, -1))
                           + jnp.sum(centered * centered, (-2, -1)))
    else:
        denominator = n * n - jnp.sum(m * m, -1)
    return d ** 0.5 * m * jax.lax.rsqrt(denominator[..., None])


def robust_pool(h):
    """Self-correction: rational inverse-energy weighted Euclidean centroid.

    Fixed points are preserved. An isolated token's influence redescends as
    O(1/||h||), unlike norm clipping/Lorentz pooling, whose influence saturates.
    Its derivative is O(1/||h||²) with the other finite tokens fixed.
    """
    h = work(h)
    d = h.shape[-1]
    w = d/(d+jnp.sum(h*h, -1, keepdims=True))
    return jnp.sum(w*h, -2)/jnp.sum(w, -2)


def head_probabilities(z, tau=4.0):
    base = rho(work(z) / tau)
    base = base / jnp.max(base, -1, keepdims=True)
    w = square_power(base)
    return w / jnp.sum(w, -1, keepdims=True)


def mle_gradient(z, y, tau=4.0, exponent=0.5):
    p = head_probabilities(z, tau)
    r = jax.lax.rsqrt(1 + (z / tau) * (z / tau))
    if exponent == 0.25:
        r = r * jax.lax.rsqrt(r)
    elif exponent != 0.5:
        raise ValueError("only the two proposed exponents are supported")
    return (8 / tau) * (p - y) * r


def head_loss(z, y, kind="power", tau=4.0):
    """Strictly proper algebraic alternatives to the impossible pure exact CE.

    Power score a=7/8 uses three nested roots, preserving the root of the
    unnormalised link so tiny probabilities need not be rooted after underflow.
    Target-dependent constants are omitted; optimization is unchanged.
    """
    if kind == "brier":
        diff = head_probabilities(z, tau) - y
        return 0.5 * jnp.sum(diff * diff, -1)
    base = rho(work(z) / tau)
    base = base / jnp.max(base, -1, keepdims=True)
    e = square_power(base)
    total = jnp.sum(e, -1, keepdims=True)
    root = total
    for _ in range(3):
        root = root * jax.lax.rsqrt(root)
    q = base / root
    p = e / total
    return jnp.sum(p / q, -1) / 0.875 + 8 * jnp.sum(y / q, -1)


def coupled_gate(x):
    # Existing ALU family, with differentiable scale and no custom surrogate.
    x = work(x)
    u = 0.7978845608028654 * x
    r = jax.lax.rsqrt(1 + u * u)
    gate = jnp.where(u >= 0, 1 + u*r, r*r / (1-jnp.minimum(u*r, 0)))
    return 0.5*x*gate


def rgelu(x):
    return x / (1 + square_power(rho(-1.702 * x / 8)))
