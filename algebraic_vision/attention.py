"""Multi-head attention: softmax and the radical kernel.

Both functions take queries, keys and values shaped (batch, heads, tokens,
head_dim) and return the attended values in the same layout. Scores are
accumulated in float32; the weighted sum of values uses the value dtype on
the MXU with float32 accumulation.

Radical attention replaces e^s by E_n(beta s), optionally with a per-head
sink mass Omega_h >= 0 in the denominator (the final model uses none):

    p_ij = E_n(beta s_ij) / (Omega_h + sum_k E_n(beta s_ik)).

With a sink the visible weights sum to 1 - Omega_h / D_i, so a head can
attend to "nothing". No running maximum is needed because E_n grows only
polynomially.
"""

from functools import partial

import jax
import jax.numpy as jnp

from algebraic_vision.primitives import radical_exp, radical_exp_slope


def _scores(q, k):
    scale = q.shape[-1] ** -0.5
    return jnp.einsum("bhqd,bhkd->bhqk", q, k,
                      preferred_element_type=jnp.float32) * scale


def _combine(p, v):
    return jnp.einsum("bhqk,bhkd->bhqd", p.astype(v.dtype), v,
                      preferred_element_type=jnp.float32).astype(v.dtype)


def softmax_attention(q, k, v):
    """Standard scaled dot-product attention."""
    p = jax.nn.softmax(_scores(q, k), axis=-1)
    return _combine(p, v)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def radical_weights(s, sink, order, temperature):
    """p = E_n(beta s) / (sink + sum E_n(beta s)) along the last axis.

    s: (..., keys) float32 scores; sink broadcasts against s[..., :1].
    """
    return _radical_weights_fwd(s, sink, order, temperature)[0]


def _radical_weights_fwd(s, sink, order, temperature):
    e = radical_exp(temperature * s, order)
    inv_d = 1.0 / (sink + jnp.sum(e, axis=-1, keepdims=True))
    p = e * inv_d
    return p, (s, p, inv_d)


def _radical_weights_bwd(order, temperature, res, dp):
    # Normalised form of the quotient rule; it never forms D^2, which
    # overflows float32 once E_n reaches ~1e19 (n = 16, s ~ 120).
    s, p, inv_d = res
    inner = jnp.sum(p * dp, axis=-1, keepdims=True)
    slope = radical_exp_slope(temperature * s, order)
    ds = temperature * slope * p * (dp - inner)
    dsink = -inner * inv_d
    return ds, dsink


radical_weights.defvjp(_radical_weights_fwd, _radical_weights_bwd)


def radical_attention(q, k, v, sink, order=8, temperature=1.0):
    """Attention with the radical kernel E_n.

    sink: array of shape (heads,), the denominator mass Omega_h >= 0 (zeros
    for plain radical attention).
    """
    s = _scores(q, k)
    sink = jnp.broadcast_to(
        jnp.asarray(sink, jnp.float32).reshape(1, -1, 1, 1),
        s.shape[:-1] + (1,))
    return _combine(radical_weights(s, sink, order, temperature), v)


def attention_weights(q, k, kind, sink=None, order=8, temperature=1.0):
    """Return the weight matrix (for diagnostics) of either attention kind."""
    s = _scores(q, k)
    if kind == "softmax":
        return jax.nn.softmax(s, axis=-1)
    sink = jnp.zeros((s.shape[1],)) if sink is None else sink
    sink = jnp.broadcast_to(
        jnp.asarray(sink, jnp.float32).reshape(1, -1, 1, 1),
        s.shape[:-1] + (1,))
    return radical_weights(s, sink, order, temperature)
