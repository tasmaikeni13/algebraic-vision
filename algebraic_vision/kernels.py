"""Pallas TPU kernels for standard and algebraic Vision Transformers.

Whole-row attention suits the short ViT sequences (up to 512 tokens).
Scores and weights stay in VMEM; backward rebuilds them and applies the
reference VJP. Token padding is masked out of both normalizers. The power
head tiles examples in groups of eight and recomputes probabilities in its
algebraic backward pass. CPU interpretation is available for correctness
checks; timings there do not measure TPU performance.
"""

from functools import partial
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from algebraic_vision.losses import _nested_sqrt, _reduce, smoothed_targets
from algebraic_vision.primitives import _check_order, _rho_and_slope

_NT = (((1,), (1,)), ((), ()))
_NN = (((1,), (0,)), ((), ()))
_TN = (((0,), (0,)), ((), ()))


def _dot(a, b, dims):
    return lax.dot_general(a, b, dims, preferred_element_type=jnp.float32)


def _attention_parts(q, k, sink, *, kind, order, temperature, tokens, exact):
    s = _dot(q, k, _NT) * q.shape[-1] ** -0.5
    valid = jnp.arange(k.shape[0])[None, :] < tokens
    if kind == "softmax":
        s = jnp.where(valid, s, -jnp.inf)
        e = jnp.exp(s - jnp.max(s, axis=1, keepdims=True))
        slope = jnp.ones_like(e)
        mass = 0.0
    else:
        x = s * temperature
        if exact:
            rho, slope = _rho_and_slope(x, order)
        else:
            x = x / order
            slope = lax.rsqrt(1.0 + x * x)
            rho = x + (1.0 + x * x) * slope
        e = rho
        for _ in range(order.bit_length() - 1):
            e = e * e
        e = jnp.where(valid, e, 0.0)
        mass = sink
    inv_d = 1.0 / (mass + jnp.sum(e, axis=1, keepdims=True))
    return e * inv_d, slope, inv_d


def _attention_fwd(sink_ref, q_ref, k_ref, v_ref, o_ref, **opts):
    q, k, v = q_ref[0, 0], k_ref[0, 0], v_ref[0, 0]
    sink = sink_ref[pl.program_id(1)]
    p, _, _ = _attention_parts(q, k, sink, **opts)
    o_ref[0, 0] = _dot(p.astype(v.dtype), v, _NN).astype(o_ref.dtype)


def _attention_bwd(sink_ref, q_ref, k_ref, v_ref, do_ref,
                   dq_ref, dk_ref, dv_ref, dsink_ref, **opts):
    q, k, v, do = q_ref[0, 0], k_ref[0, 0], v_ref[0, 0], do_ref[0, 0]
    p, slope, inv_d = _attention_parts(
        q, k, sink_ref[pl.program_id(1)], **opts)
    # Match the casts in the JAX weighted-value product. Using the rounded
    # forward output to compute delta would introduce an avoidable VJP error.
    dp = _dot(do, v, _NT).astype(v.dtype).astype(jnp.float32)
    inner = jnp.sum(p * dp, axis=1, keepdims=True)
    ds = p * (dp - inner)
    coef = q.shape[-1] ** -0.5
    if opts["kind"] == "radical":
        ds = ds * slope
        coef = coef * opts["temperature"]
        dsink_ref[0, 0] = -inner * inv_d
    else:
        dsink_ref[0, 0] = jnp.zeros_like(inner)
    ds = (ds * coef).astype(q.dtype)
    dq_ref[0, 0] = _dot(ds, k, _NN).astype(q.dtype)
    dk_ref[0, 0] = _dot(ds, q, _TN).astype(k.dtype)
    dv_ref[0, 0] = _dot(p.astype(v.dtype), do, _TN).astype(v.dtype)


def _attention_call(q, k, v, sink, kind, order, temperature, exact,
                    interpret, do=None):
    batch, heads, tokens, dim = q.shape
    padded = math.ceil(tokens / 128) * 128
    pad = ((0, 0), (0, 0), (0, padded - tokens), (0, 0))
    q, k, v = (jnp.pad(x, pad) for x in (q, k, v))
    full = pl.BlockSpec((1, 1, padded, dim),
                        lambda b, h, sink: (b, h, 0, 0))
    row = pl.BlockSpec((1, 1, padded, 1),
                       lambda b, h, sink: (b, h, 0, 0))
    opts = dict(kind=kind, order=order, temperature=temperature,
                exact=exact, tokens=tokens)
    compiler = pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
    shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
    if do is None:
        grid = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=(batch, heads),
            in_specs=[full, full, full], out_specs=full)
        out = pl.pallas_call(
            partial(_attention_fwd, **opts), out_shape=shape,
            grid_spec=grid, compiler_params=compiler, interpret=interpret)(sink, q, k, v)
        return out[:, :, :tokens]
    grid = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(batch, heads),
        in_specs=[full, full, full, full], out_specs=[full, full, full, row])
    dq, dk, dv, dsink = pl.pallas_call(
        partial(_attention_bwd, **opts),
        out_shape=[shape, shape, shape,
                   jax.ShapeDtypeStruct((batch, heads, padded, 1), jnp.float32)],
        grid_spec=grid, compiler_params=compiler, interpret=interpret)(
            sink, q, k, v, jnp.pad(do, pad))
    return (dq[:, :, :tokens], dk[:, :, :tokens], dv[:, :, :tokens],
            jnp.sum(dsink, axis=(0, 2, 3)))


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def _fused(q, k, v, sink, kind, order, temperature, exact, interpret):
    return _attention_call(q, k, v, sink, kind, order, temperature, exact, interpret)


def _fused_fwd(q, k, v, sink, kind, order, temperature, exact, interpret):
    out = _fused(q, k, v, sink, kind, order, temperature, exact, interpret)
    return out, (q, k, v, sink)


def _fused_bwd(kind, order, temperature, exact, interpret, res, do):
    q, k, v, sink = res
    return _attention_call(q, k, v, sink, kind, order, temperature, exact,
                           interpret, do.astype(v.dtype))


_fused.defvjp(_fused_fwd, _fused_bwd)


def fused_attention(q, k, v, kind="softmax", sink=None, order=8,
                    temperature=1.0, exact=True, interpret=None):
    """Fused standard or radical attention on a local-device ViT batch."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share (batch, heads, tokens, head_dim)")
    if not all(q.shape) or q.shape[2] > 512:
        raise ValueError("whole-row attention needs 1 to 512 tokens and nonempty axes")
    if q.dtype not in (jnp.float32, jnp.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, v must share float32 or bfloat16 dtype")
    if kind not in ("softmax", "radical"):
        raise ValueError(f"unknown attention kind {kind!r}")
    _check_order(order)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    sink = jnp.zeros((q.shape[1],), jnp.float32) if sink is None else jnp.asarray(sink, jnp.float32)
    if sink.shape != (q.shape[1],):
        raise ValueError("sink must have one mass per head")
    return _fused(q, k, v, sink, kind, int(order), float(temperature),
                  bool(exact), bool(interpret))


def _power_parts(z, y, *, exponent, order, classes, smoothing):
    rho, slope = _rho_and_slope(z, order)
    valid = jnp.arange(z.shape[-1])[None, :] < classes
    rho = rho / jnp.max(jnp.where(valid, rho, 0.0), axis=-1, keepdims=True)
    levels = abs(exponent)
    root = rho
    for _ in range(order.bit_length() - 1 - levels):
        root = root * root
    e = root
    for _ in range(levels):
        e = e * e
    e = jnp.where(valid, e, 0.0)
    total = jnp.sum(e, axis=-1, keepdims=True)
    q = root / _nested_sqrt(total, levels)
    p = e / total
    if exponent < 0:
        a = 1.0 - 2.0 ** -levels
        pa, pam1 = p / q, 1.0 / q
    else:
        a = 1.0 + 2.0 ** -levels
        pa, pam1 = p * q, q
    hi, lo = 1.0 - smoothing + smoothing / classes, smoothing / classes
    const = (hi ** a + (classes - 1) * lo ** a) / (a * (a - 1.0))
    pa_sum = jnp.sum(pa, -1, keepdims=True)
    yp = y * pam1
    yp_sum = jnp.sum(yp, -1, keepdims=True)
    values = pa_sum / a - yp_sum / (a - 1.0) + const
    dz = slope * (pa - yp - p * (pa_sum - yp_sum))
    return values, dz


def _power_fwd(z_ref, y_ref, out_ref, **opts):
    values, _ = _power_parts(z_ref[...], y_ref[...], **opts)
    out_ref[...] = values


def _power_bwd(z_ref, y_ref, g_ref, dz_ref, **opts):
    _, dz = _power_parts(z_ref[...], y_ref[...], **opts)
    dz_ref[...] = g_ref[...] * dz


def _power_call(z, y, exponent, order, smoothing, interpret, g=None):
    batch, classes = z.shape
    rows = math.ceil(batch / 8) * 8
    cols = math.ceil(classes / 128) * 128
    pad = ((0, rows - batch), (0, cols - classes))
    z, y = jnp.pad(z, pad), jnp.pad(y, pad)
    matrix = pl.BlockSpec((8, cols), lambda i: (i, 0))
    row = pl.BlockSpec((8, 1), lambda i: (i, 0))
    opts = dict(exponent=exponent, order=order, classes=classes, smoothing=smoothing)
    if g is None:
        out = pl.pallas_call(
            partial(_power_fwd, **opts),
            out_shape=jax.ShapeDtypeStruct((rows, 1), jnp.float32),
            grid=(rows // 8,), in_specs=[matrix, matrix], out_specs=row,
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            interpret=interpret)(z, y)
        return out[:batch, 0]
    g = jnp.pad(g[:, None], ((0, rows - batch), (0, 0)))
    out = pl.pallas_call(
        partial(_power_bwd, **opts),
        out_shape=jax.ShapeDtypeStruct(z.shape, jnp.float32),
        grid=(rows // 8,), in_specs=[matrix, matrix, row], out_specs=matrix,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret)(z, y, g)
    return out[:batch, :classes]


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def _fused_power(z, y, exponent, order, smoothing, interpret):
    return _power_call(z, y, exponent, order, smoothing, interpret)


def _fused_power_fwd(z, y, exponent, order, smoothing, interpret):
    return _fused_power(z, y, exponent, order, smoothing, interpret), (z, y)


def _fused_power_bwd(exponent, order, smoothing, interpret, res, g):
    z, y = res
    return _power_call(z, y, exponent, order, smoothing, interpret, g), None


_fused_power.defvjp(_fused_power_fwd, _fused_power_bwd)


def fused_power_score(logits, labels, alpha_exponent=-6, order=64,
                      smoothing=0.1, reduction="mean", interpret=None):
    """Power-score loss and exact algebraic VJP, tiled over examples.

    Logit normalization, when requested, is applied in the trainer before
    this call, so JAX propagates its derivative outside the fused head.
    """
    _check_order(order)
    if not 1 <= abs(alpha_exponent) or (1 << abs(alpha_exponent)) > order:
        raise ValueError("need 1 <= |alpha_exponent| and 2^|alpha_exponent| <= order")
    if not 0 <= smoothing <= 1:
        raise ValueError("smoothing must be between zero and one")
    if logits.ndim != 2 or not all(logits.shape) or labels.shape != logits.shape[:1]:
        raise ValueError("logits must be (batch, classes) and labels (batch,)")
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    z = logits.astype(jnp.float32)
    y = smoothed_targets(labels, z.shape[-1], smoothing)
    return _reduce(_fused_power(z, y, int(alpha_exponent), int(order),
                               float(smoothing), bool(interpret)), reduction)
