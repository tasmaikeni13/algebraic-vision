"""Online tiled softmax forward and analytic recomputing backward for ViTs.

Only 128x128 score tiles live in VMEM in the forward pass. For the short
196-token ViT shape the existing fused analytic backward rebuilds a whole
row in VMEM; neither direction writes an NxN attention matrix to HBM.
"""

from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from algebraic_vision.kernels import _dot, _NT, _NN, _attention_call


def _forward(q_ref,k_ref,v_ref,o_ref,*,tokens):
    q=q_ref[0,0]
    rows,dim=q.shape
    maximum=jnp.full((rows,1),-jnp.inf,jnp.float32)
    total=jnp.zeros((rows,1),jnp.float32)
    numerator=jnp.zeros((rows,dim),jnp.float32)
    def body(i,state):
        maximum,total,numerator=state
        k=k_ref[0,0,pl.ds(i*128,128),:]
        v=v_ref[0,0,pl.ds(i*128,128),:]
        score=_dot(q,k,_NT)*dim**-0.5
        valid=jnp.arange(128)[None,:]+i*128<tokens
        score=jnp.where(valid,score,-jnp.inf)
        new_max=jnp.maximum(maximum,jnp.max(score,-1,keepdims=True))
        factor=jnp.exp(maximum-new_max)
        weights=jnp.exp(score-new_max)
        numerator=numerator*factor+_dot(weights.astype(v.dtype),v,_NN)
        total=total*factor+jnp.sum(weights,-1,keepdims=True)
        return new_max,total,numerator
    _,total,numerator=jax.lax.fori_loop(0,k_ref.shape[2]//128,body,(maximum,total,numerator))
    o_ref[0,0]= (numerator/total).astype(o_ref.dtype)


def _call(q,k,v,interpret):
    batch,heads,tokens,dim=q.shape
    padded=math.ceil(tokens/128)*128
    pad=((0,0),(0,0),(0,padded-tokens),(0,0))
    q,k,v=(jnp.pad(a,pad) for a in (q,k,v))
    query=pl.BlockSpec((1,1,128,dim),lambda b,h,i:(b,h,i,0))
    whole=pl.BlockSpec((1,1,padded,dim),lambda b,h,i:(b,h,0,0))
    out=pl.pallas_call(partial(_forward,tokens=tokens),out_shape=jax.ShapeDtypeStruct(q.shape,q.dtype),
                grid=(batch,heads,padded//128),in_specs=(query,whole,whole),out_specs=query,
                compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel","parallel","parallel")),
                interpret=interpret)(q,k,v)
    return out[:,:,:tokens]


@partial(jax.custom_vjp,nondiff_argnums=(3,))
def _flash(q,k,v,interpret):
    return _call(q,k,v,interpret)


def _fwd(q,k,v,interpret):
    return _call(q,k,v,interpret),(q,k,v)


def _bwd(interpret,res,g):
    q,k,v=res
    return _attention_call(q,k,v,jnp.zeros((q.shape[1],),jnp.float32),"softmax",8,1.0,True,interpret,g)[:3]


_flash.defvjp(_fwd,_bwd)


def flash_attention(q,k,v,interpret=None):
    if q.ndim!=4 or q.shape!=k.shape or q.shape!=v.shape or not all(q.shape) or q.shape[2]>512:
        raise ValueError("nonempty matching BHND tensors with <=512 tokens required")
    if q.dtype not in (jnp.float32,jnp.bfloat16) or k.dtype!=q.dtype or v.dtype!=q.dtype:
        raise ValueError("matching float32/bfloat16 dtypes required")
    return _flash(q,k,v,jax.default_backend()!="tpu" if interpret is None else bool(interpret))
