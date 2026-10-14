"""Rejected Phase 1 rational attention proposal, retained as a kernel ablation.

This is not the selected algebraic_v2 model. Its O(N*d) Klein lift is outside
Pallas; both score-tile programs contain zero roots. Exposing it separately
allows hardware verification without replacing the accepted E8 formulation.
"""

from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from algebraic_vision.kernels import _dot, _NT, _NN, _TN


def ball_lift(x):
    work = x.astype(jnp.float32)
    return (work * jax.lax.rsqrt(x.shape[-1] + jnp.sum(work * work, -1, keepdims=True))).astype(x.dtype)


def _parts(q, k, margin, tokens):
    raw = _dot(q, k, _NT)
    u = jnp.clip(raw, -1.0, 1.0)
    base = (margin + u) / (margin - u)
    e = base * base
    e = e * e
    e = e * e
    e = jnp.where(jnp.arange(k.shape[0])[None, :] < tokens, e, 0.0)
    p = e / jnp.sum(e, -1, keepdims=True)
    return p, u, raw


def reference(q, k, v, margin):
    raw = jnp.einsum("bhid,bhjd->bhij", q, k, preferred_element_type=jnp.float32)
    u = jnp.clip(raw, -1.0, 1.0)
    base = (margin[None, :, None, None] + u) / (margin[None, :, None, None] - u)
    e = base * base
    e = e * e
    e = e * e
    p = e / jnp.sum(e, -1, keepdims=True)
    return jnp.einsum("bhij,bhjd->bhid", p.astype(v.dtype), v,
                       preferred_element_type=jnp.float32).astype(v.dtype)


def _forward(margin_ref, q_ref, k_ref, v_ref, out_ref, *, tokens):
    q, k, v = q_ref[0, 0], k_ref[0, 0], v_ref[0, 0]
    p, _, _ = _parts(q, k, margin_ref[pl.program_id(1)], tokens)
    out_ref[0, 0] = _dot(p.astype(v.dtype), v, _NN).astype(v.dtype)


def _backward(margin_ref, q_ref, k_ref, v_ref, grad_ref, dq_ref, dk_ref, dv_ref, dm_ref, *, tokens):
    q, k, v, g = q_ref[0, 0], k_ref[0, 0], v_ref[0, 0], grad_ref[0, 0]
    margin = margin_ref[pl.program_id(1)]
    p, u, raw = _parts(q, k, margin, tokens)
    dp = _dot(g, v, _NT).astype(v.dtype).astype(jnp.float32)
    centered = p * (dp - jnp.sum(p * dp, -1, keepdims=True))
    inv = 16.0 / (margin * margin - u * u)
    du = centered * margin * inv
    # JAX clip has the half derivative at an exact boundary.
    clip_grad = jnp.where((raw > -1) & (raw < 1), 1.0,
                          jnp.where((raw == -1) | (raw == 1), 0.5, 0.0))
    du = (du * clip_grad).astype(q.dtype)
    dq_ref[0, 0] = _dot(du, k, _NN).astype(q.dtype)
    dk_ref[0, 0] = _dot(du, q, _TN).astype(k.dtype)
    dv_ref[0, 0] = _dot(p.astype(v.dtype), g, _TN).astype(v.dtype)
    dm_ref[0, 0] = jnp.sum(-centered * u * inv, -1, keepdims=True)


def _call(q, k, v, margin, interpret, g=None):
    batch, heads, tokens, dim = q.shape
    padded = math.ceil(tokens / 128) * 128
    pad = ((0, 0), (0, 0), (0, padded-tokens), (0, 0))
    q, k, v = (jnp.pad(x, pad) for x in (q, k, v))
    full = pl.BlockSpec((1,1,padded,dim), lambda b,h,m:(b,h,0,0))
    row = pl.BlockSpec((1,1,padded,1), lambda b,h,m:(b,h,0,0))
    compiler = pltpu.CompilerParams(dimension_semantics=("parallel","parallel"))
    shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
    if g is None:
        grid = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(batch,heads),
                                    in_specs=[full,full,full],out_specs=full)
        return pl.pallas_call(partial(_forward,tokens=tokens),out_shape=shape,
                    grid_spec=grid,compiler_params=compiler,interpret=interpret)(margin,q,k,v)[:,:,:tokens]
    grid = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1,grid=(batch,heads),
                        in_specs=[full,full,full,full],out_specs=[full,full,full,row])
    dq,dk,dv,dm = pl.pallas_call(partial(_backward,tokens=tokens),
                 out_shape=[shape,shape,shape,jax.ShapeDtypeStruct((batch,heads,padded,1),jnp.float32)],
                 grid_spec=grid,compiler_params=compiler,interpret=interpret)(margin,q,k,v,jnp.pad(g,pad))
    return dq[:,:,:tokens],dk[:,:,:tokens],dv[:,:,:tokens],jnp.sum(dm,(0,2,3))


@partial(jax.custom_vjp,nondiff_argnums=(4,))
def _fused(q,k,v,margin,interpret):
    return _call(q,k,v,margin,interpret)


def _fwd(q,k,v,margin,interpret):
    return _call(q,k,v,margin,interpret),(q,k,v,margin)


def _bwd(interpret,res,g):
    return _call(*res,interpret,g)


_fused.defvjp(_fwd,_bwd)


def fused_rational(q,k,v,margin=None,interpret=None):
    """Attention on already lifted/rotated vectors, with exact margin VJP."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or not all(q.shape):
        raise ValueError("q/k/v need a shared nonempty BHND shape")
    if q.shape[-2] > 512 or q.dtype not in (jnp.float32,jnp.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("requires float32/bfloat16, <=512 tokens")
    margin = jnp.full((q.shape[1],),2.0,jnp.float32) if margin is None else jnp.asarray(margin,jnp.float32)
    if margin.shape != (q.shape[1],):
        raise ValueError("one margin per head required")
    interpret = jax.default_backend() != "tpu" if interpret is None else interpret
    return _fused(q,k,v,margin,bool(interpret))
