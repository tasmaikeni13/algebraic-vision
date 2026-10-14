"""Algebraic primitives: radical exponential, gates and Cayley rotations.

Every function here is built from +, -, *, /, square root and comparisons
only. The nonlinear ones have custom derivatives, so that autodiff never
differentiates through a square root where that would be inexact or
undefined, and so the backward pass reuses quantities from the forward pass.

Notation (see analysis/symbolic_checks.py for the verified identities)::

    rho_n(s)   = s/n + sqrt(1 + (s/n)^2)
    E_n(s)     = rho_n(s)^n                  radical exponential, n = 2^m
    E_n'(s)    = E_n(s) / sqrt(1 + (s/n)^2)
    sigma_n(y) = 1 / (1 + E_n(-y))           radical logistic
    F(y)       = (1 + y / sqrt(1 + y^2)) / 2
    ALU_c(x)   = x * F(c x)
"""

from functools import partial

import jax
import jax.numpy as jnp


def _check_order(order):
    if order < 1 or order & (order - 1):
        raise ValueError(f"order must be a power of two, got {order}")


def _rho_and_slope(s, order):
    """Return rho_n(s) and 1/sqrt(1 + (s/n)^2) without cancellation.

    For s < 0 the sum s/n + sqrt(1 + (s/n)^2) subtracts nearly equal numbers,
    so the conjugate form r / (1 - u) is used instead, where
    r = 1/sqrt(1 + x^2) and u = x r lies in (-1, 1).
    """
    x = s / order
    r = jax.lax.rsqrt(1.0 + x * x)
    u = x * r
    positive = x + (1.0 + x * x) * r
    negative = r / (1.0 - jnp.minimum(u, 0.0))
    return jnp.where(x >= 0, positive, negative), r


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _radical_exp(s, order):
    rho, _ = _rho_and_slope(s, order)
    for _ in range(order.bit_length() - 1):
        rho = rho * rho
    return rho


@_radical_exp.defjvp
def _radical_exp_jvp(order, primals, tangents):
    (s,), (ds,) = primals, tangents
    rho, slope = _rho_and_slope(s, order)
    for _ in range(order.bit_length() - 1):
        rho = rho * rho
    return rho, rho * slope * ds


def radical_exp(s, order=8):
    """E_n(s) = (s/n + sqrt(1 + s^2/n^2))^n, an algebraic stand-in for e^s.

    E_n is positive, satisfies E_n(s) E_n(-s) = 1, matches e^s to second order
    at s = 0 and tends to e^s as n grows, but grows only polynomially,
    E_n(s) ~ (2s/n)^n, for |s| >> n. Computed in float32 or wider.
    """
    _check_order(order)
    s = jnp.asarray(s)
    dtype = jnp.promote_types(s.dtype, jnp.float32)
    return _radical_exp(s.astype(dtype), int(order))


def radical_exp_slope(s, order=8):
    """d/ds log E_n(s) = 1/sqrt(1 + (s/n)^2), the kernel's local sharpness."""
    x = jnp.asarray(s, jnp.promote_types(jnp.asarray(s).dtype, jnp.float32))
    x = x / order
    return jax.lax.rsqrt(1.0 + x * x)


@jax.custom_jvp
def _alu(x, c):
    y = c * x
    r = jax.lax.rsqrt(1.0 + y * y)
    u = y * r
    # 1 + u = r^2 / (1 - u) avoids cancellation for large negative inputs.
    gate = jnp.where(y >= 0, 1.0 + u, r * r / (1.0 - jnp.minimum(u, 0.0)))
    return 0.5 * x * gate


@_alu.defjvp
def _alu_jvp(primals, tangents):
    x, c = primals
    dx, _ = tangents
    y = c * x
    r = jax.lax.rsqrt(1.0 + y * y)
    u = y * r
    gate = jnp.where(y >= 0, 1.0 + u, r * r / (1.0 - jnp.minimum(u, 0.0)))
    slope = 0.5 * gate + 0.5 * u * (1.0 - u * u)
    return 0.5 * x * gate, slope * dx


def alu(x, c=0.7978845608028654):
    """Algebraic linear unit ALU_c(x) = x (1 + cx/sqrt(1 + c^2 x^2)) / 2.

    The default c = sqrt(2/pi) matches GELU's curvature at the origin. The
    derivative lies in [-0.0443, 1.0443] for every c (GELU: [-0.129, 1.129]).
    Internal arithmetic is float32 or wider; the output keeps x's dtype.
    """
    x = jnp.asarray(x)
    dtype = jnp.promote_types(x.dtype, jnp.float32)
    return _alu(x.astype(dtype), jnp.asarray(c, dtype)).astype(x.dtype)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _radical_sigmoid(y, order):
    return 1.0 / (1.0 + _radical_exp(-y, order))


@_radical_sigmoid.defjvp
def _radical_sigmoid_jvp(order, primals, tangents):
    (y,), (dy,) = primals, tangents
    sig = 1.0 / (1.0 + _radical_exp(-y, order))
    x = y / order
    slope = jax.lax.rsqrt(1.0 + x * x)
    # sigma' = sigma (1 - sigma) g(y); written without E^2 so it cannot
    # overflow where E_n(-y) is huge.
    return sig, sig * (1.0 - sig) * slope * dy


def radical_sigmoid(y, order=8):
    """1 / (1 + E_n(-y)): an algebraic logistic function.

    Matches the logistic sigma(y) = 1/(1 + e^-y) to second order at 0,
    satisfies sigma(y) + sigma(-y) = 1, and approaches 0 and 1 like
    (n / (2|y|))^n.
    """
    _check_order(order)
    y = jnp.asarray(y)
    dtype = jnp.promote_types(y.dtype, jnp.float32)
    return _radical_sigmoid(y.astype(dtype), int(order))


def radical_gelu(x, beta=1.702, order=8):
    """x * sigma_n(beta x): GELU/SiLU-shaped gate built from E_n.

    beta = 1.702 follows the logistic approximation of GELU, beta = 1 gives
    an algebraic SiLU. Output keeps x's dtype; internals are float32.
    """
    x = jnp.asarray(x)
    xf = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    return (xf * radical_sigmoid(beta * xf, order)).astype(x.dtype)


def cayley_tables(num_positions, freqs):
    """Rotation tables (cos-like, sin-like) of R(w)^p for p < num_positions.

    R(w) = [[1 - w^2, -2w], [2w, 1 - w^2]] / (1 + w^2) is the Cayley rotation
    by angle 2 atan(w). Powers are built by repeated multiplication with a
    renormalisation after each step, so rounding cannot change the norm.
    freqs may have any shape; the result has shape (num_positions,) +
    freqs.shape, in float32, and is differentiable in freqs.
    """
    w = jnp.asarray(freqs, jnp.float32)
    c1 = (1.0 - w * w) / (1.0 + w * w)
    s1 = 2.0 * w / (1.0 + w * w)

    def step(carry, _):
        c, s = carry
        c_next = c1 * c - s1 * s
        s_next = s1 * c + c1 * s
        inv = jax.lax.rsqrt(c_next * c_next + s_next * s_next)
        return (c_next * inv, s_next * inv), (c, s)

    init = (jnp.ones_like(w), jnp.zeros_like(w))
    _, (cos_like, sin_like) = jax.lax.scan(step, init, None,
                                           length=num_positions)
    return cos_like, sin_like


def cayley_frequencies(num_pairs, max_turn=0.5, min_turn=None):
    """Frequency grid w_k for Cayley rotations, from w_0 down to min_turn.

    w_k = max_turn / (1 + a k)^2, with a chosen so that the last pair turns
    by min_turn (default max_turn / 32): a slowly decaying spectrum built
    from arithmetic and one square root.
    """
    if min_turn is None:
        min_turn = max_turn / 32.0
    k = jnp.arange(num_pairs, dtype=jnp.float32)
    ratio = max_turn / min_turn
    # (1 + a (K-1))^2 = ratio  ->  a = (sqrt(ratio) - 1)/(K - 1)
    root = ratio * jax.lax.rsqrt(jnp.float32(ratio))
    a = (root - 1.0) / max(num_pairs - 1, 1)
    return max_turn / (1.0 + a * k) ** 2


def rotate_pairs(x, cos_like, sin_like):
    """Rotate the feature pairs (x[i], x[i + d/2]) by the given tables.

    x has shape (..., tokens, d); the tables broadcast against
    (tokens, d/2). Pairing the two halves (rather than neighbours) keeps the
    rotation a pair of contiguous slices, which suits TPU vector lanes.
    """
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    c = cos_like.astype(x.dtype)
    s = sin_like.astype(x.dtype)
    return jnp.concatenate([c * a - s * b, s * a + c * b], axis=-1)
