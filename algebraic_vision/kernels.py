"""Fused attention kernels for TPU (Pallas): softmax and radical.

A ViT sequence (64 to 257 tokens) is short enough that one image's scores
for every head fit in VMEM, so each kernel instance handles one image: it
reads q, k, v for all heads, forms scores, weights and outputs on chip, and
writes only the outputs and one statistic per query row; the backward kernel
rebuilds the weights from it. Score and weight matrices never travel to HBM,
which is where the plain XLA lowering spends its time.

Softmax rows store the log-normaliser m + log(l) (with the usual running
maximum m); radical rows store the denominator D = Omega + sum E_n(beta s),
which needs no maximum because E_n grows polynomially.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NT = (((1,), (1,)), ((), ()))     # a @ b.T
_NN = (((1,), (0,)), ((), ()))     # a @ b
_TN = (((0,), (0,)), ((), ()))     # a.T @ b


def _radical_rho(x, order, exact=False):
    """E_n = rho^n and the slope 1/sqrt(1 + x^2), for x = beta s / n.

    The direct form x + sqrt(1 + x^2) loses relative accuracy for very
    negative x through cancellation, but only on weights that are already
    negligible next to the row's largest: the absolute error of every
    normalised weight stays below ~1e-6 (tests/test_kernels.py), far
    below the bfloat16 rounding of q and k. exact=True uses the
    cancellation-free conjugate form at the cost of a division per score.
    """
    q = 1.0 + x * x
    r = lax.rsqrt(q)
    if exact:
        u = x * r
        rho = jnp.where(x >= 0, x + q * r, r / (1.0 - jnp.minimum(u, 0.0)))
    else:
        rho = x + q * r
    for _ in range(int(math.log2(order))):
        rho = rho * rho
    return rho, r


def _fwd_kernel(sink_ref, q_ref, k_ref, v_ref, o_ref, stat_ref, *, heads,
                kind, order, coef, exact):
    for h in range(heads):
        q, k, v = q_ref[0, h], k_ref[0, h], v_ref[0, h]
        s = lax.dot_general(q, k, _NT, preferred_element_type=jnp.float32)
        s = s * coef
        if kind == "softmax":
            m = jnp.max(s, axis=1, keepdims=True)
            e = jnp.exp(s - m)
            total = jnp.sum(e, axis=1, keepdims=True)
            stat = m + jnp.log(total)
        else:
            e, _ = _radical_rho(s, order, exact)
            total = sink_ref[h] + jnp.sum(e, axis=1, keepdims=True)
            stat = total
        p = e * (1.0 / total)
        o = lax.dot_general(p.astype(v.dtype), v, _NN,
                            preferred_element_type=jnp.float32)
        o_ref[0, h] = o.astype(o_ref.dtype)
        stat_ref[0, h] = stat


def _bwd_kernel(sink_ref, q_ref, k_ref, v_ref, o_ref, do_ref, stat_ref,
                dq_ref, dk_ref, dv_ref, dsink_ref, *, heads, kind, order,
                coef, grad_coef, exact):
    del sink_ref
    for h in range(heads):
        q, k, v = q_ref[0, h], k_ref[0, h], v_ref[0, h]
        o, do = o_ref[0, h], do_ref[0, h]
        stat = stat_ref[0, h]
        s = lax.dot_general(q, k, _NT, preferred_element_type=jnp.float32)
        s = s * coef
        if kind == "softmax":
            p = jnp.exp(s - stat)
        else:
            e, r = _radical_rho(s, order, exact)
            p = e * (1.0 / stat)
        dp = lax.dot_general(do, v, _NT, preferred_element_type=jnp.float32)
        delta = jnp.sum(do.astype(jnp.float32) * o.astype(jnp.float32),
                        axis=1, keepdims=True)
        ds = p * (dp - delta)
        if kind != "softmax":
            ds = ds * r
            dsink_ref[0, h] = -delta * (1.0 / stat)
        else:
            dsink_ref[0, h] = jnp.zeros_like(delta)
        ds_lo = ds.astype(q.dtype)
        dq = lax.dot_general(ds_lo, k, _NN, preferred_element_type=jnp.float32)
        dk = lax.dot_general(ds_lo, q, _TN, preferred_element_type=jnp.float32)
        dv = lax.dot_general(p.astype(do.dtype), do, _TN,
                             preferred_element_type=jnp.float32)
        dq_ref[0, h] = (dq * grad_coef).astype(dq_ref.dtype)
        dk_ref[0, h] = (dk * grad_coef).astype(dk_ref.dtype)
        dv_ref[0, h] = dv.astype(dv_ref.dtype)


def _specs(shape):
    b, h, t, d = shape
    full = pl.BlockSpec((1, h, t, d), lambda i, sink: (i, 0, 0, 0))
    row = pl.BlockSpec((1, h, t, 1), lambda i, sink: (i, 0, 0, 0))
    return full, row


def _coefs(kind, head_dim, order, temperature):
    """Score multiplier inside the kernel and the gradient multiplier."""
    scale = head_dim ** -0.5
    if kind == "softmax":
        return scale, scale
    # Radical scores enter as x = beta s / n; dE/ds = beta E g with g applied
    # in-kernel, so dq and dk are scaled by scale * beta outside.
    return scale * temperature / order, scale * temperature


def _forward(q, k, v, sink, kind, order, temperature, exact, interpret):
    b, h, t, d = q.shape
    full, row = _specs(q.shape)
    coef, _ = _coefs(kind, d, order, temperature)
    kernel = functools.partial(_fwd_kernel, heads=h, kind=kind, order=order,
                               coef=coef, exact=exact)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(b,), in_specs=[full, full, full],
        out_specs=[full, row])
    return pl.pallas_call(
        kernel,
        out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype),
                   jax.ShapeDtypeStruct((b, h, t, 1), jnp.float32)],
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)),
        interpret=interpret)(sink, q, k, v)


def _backward(q, k, v, o, do, stat, sink, kind, order, temperature, exact,
              interpret):
    b, h, t, d = q.shape
    full, row = _specs(q.shape)
    coef, grad_coef = _coefs(kind, d, order, temperature)
    kernel = functools.partial(_bwd_kernel, heads=h, kind=kind, order=order,
                               coef=coef, grad_coef=grad_coef, exact=exact)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(b,),
        in_specs=[full, full, full, full, full, row],
        out_specs=[full, full, full, row])
    dq, dk, dv, dsink = pl.pallas_call(
        kernel,
        out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype),
                   jax.ShapeDtypeStruct(k.shape, k.dtype),
                   jax.ShapeDtypeStruct(v.shape, v.dtype),
                   jax.ShapeDtypeStruct((b, h, t, 1), jnp.float32)],
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)),
        interpret=interpret)(sink, q, k, v, o, do, stat)
    return dq, dk, dv, jnp.sum(dsink, axis=(0, 2, 3))


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def _fused(q, k, v, sink, kind, order, temperature, exact, interpret):
    return _forward(q, k, v, sink, kind, order, temperature, exact,
                    interpret)[0]


def _fused_fwd(q, k, v, sink, kind, order, temperature, exact, interpret):
    o, stat = _forward(q, k, v, sink, kind, order, temperature, exact,
                       interpret)
    return o, (q, k, v, sink, o, stat)


def _fused_bwd(kind, order, temperature, exact, interpret, res, do):
    q, k, v, sink, o, stat = res
    dq, dk, dv, dsink = _backward(q, k, v, o, do.astype(q.dtype), stat, sink,
                                  kind, order, temperature, exact, interpret)
    return dq, dk, dv, dsink.astype(sink.dtype)


_fused.defvjp(_fused_fwd, _fused_bwd)


def fused_attention(q, k, v, kind="softmax", sink=None, order=8,
                    temperature=1.0, exact=False, interpret=None):
    """Attention for (batch, heads, tokens, head_dim) inputs on one chip.

    kind: "softmax" (standard) or "radical" (E_n, optional per-head sink).
    The result matches attention.softmax_attention / radical_attention.
    """
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    heads = q.shape[1]
    if sink is None:
        sink = jnp.zeros((heads,), jnp.float32)
    sink = jnp.asarray(sink, jnp.float32).reshape(heads)
    return _fused(q, k, v, sink, kind, int(order), float(temperature),
                  bool(exact), bool(interpret))
