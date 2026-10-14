"""Matched-depth mean-field surrogates and exact small Jacobian spectra.

These are synthetic research stacks, not ViT-B/L training implementations.
Forward studies use 64-dimensional heads; exact dense spectra use a declared
small state space so every singular value can be measured rather than guessed.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from . import references as ref
from .simulations import VARIANTS, layer_norm


def init_blocks(rng, depth, width):
    def mat(a,b):
        return jnp.asarray(rng.normal(size=(a,b))/math.sqrt(a))
    return [{"q":mat(width,width),"k":mat(width,width),"v":mat(width,width),
             "o":mat(width,width),"up":mat(width,4*width),"down":mat(4*width,width)} for _ in range(depth)]


def block(x, params, variant, gate="rgelu"):
    norm=layer_norm(x)
    q,k,v=[norm@params[name] for name in ["q","k","v"]]
    grid=math.isqrt(x.shape[-2])
    if variant.startswith("v2"):
        q,k=ref.bicayley(q,grid),ref.bicayley(k,grid)
    elif variant=="algebraic_v1":
        # Two mixed planar frequency pairs, not an imposed shared null-line.
        from algebraic_vision.primitives import cayley_tables, rotate_pairs
        pairs=x.shape[-1]//2
        wx=jnp.linspace(0.01,0.12,pairs)
        wy=jnp.linspace(0.09,-0.03,pairs)
        cx,sx=cayley_tables(grid,wx)
        cy,sy=cayley_tables(grid,wy)
        row,col=jnp.arange(x.shape[-2])//grid,jnp.arange(x.shape[-2])%grid
        cp,sp=cx[col]*cy[row]-sx[col]*sy[row],sx[col]*cy[row]+cx[col]*sy[row]
        q,k=rotate_pairs(q,cp,sp),rotate_pairs(k,cp,sp)
    p=ref.attention(q,k,*VARIANTS[variant])
    x=x+0.5*((p@v)@params["o"])
    hidden=layer_norm(x)@params["up"]
    hidden=(jax.nn.gelu(hidden,approximate=False) if variant=="standard" else
            ref.coupled_gate(hidden) if gate=="alu" else ref.rgelu(hidden))
    return x+0.5*(hidden@params["down"]),p


def cosine(x):
    x=x/jnp.sqrt(jnp.sum(x*x,-1,keepdims=True))
    n=x.shape[-2]
    return (jnp.sum(x@x.T)-n)/(n*(n-1))


def run(seeds):
    forward,spectral=[],[]
    variants=[(name,"rgelu") for name in VARIANTS]+[("v2_sharp","alu")]
    for seed in seeds:
        rng=np.random.default_rng(seed+13000)
        params=init_blocks(rng,24,64)
        initial=jnp.asarray(rng.normal(size=(64,64)))
        small_params=init_blocks(rng,24,16)
        small_initial=jnp.asarray(rng.normal(size=(4,16)))
        for variant,gate in variants:
            fn=jax.jit(lambda x,p:block(x,p,variant,gate))
            jfn=jax.jit(jax.jacfwd(lambda x,p:block(x,p,variant,gate)[0],argnums=0))
            smallfn=jax.jit(lambda x,p:block(x,p,variant,gate)[0])
            x=initial
            xs=small_initial
            total=np.eye(xs.size)
            for layer in range(24):
                before_var=float(jnp.mean(x*x))
                before_cos=float(cosine(x))
                x,p=fn(x,params[layer])
                eig=np.linalg.eigvals(np.asarray(p))
                eig=np.sort(np.abs(eig))[::-1]
                forward.append({"seed":seed,"variant":variant,"gate":gate,"layer":layer+1,
                                "variance_in":before_var,"variance_out":float(jnp.mean(x*x)),
                                "cosine_in":before_cos,"cosine_out":float(cosine(x)),
                                "spectral_gap":float(1-eig[1]),
                                "entropy":float(jnp.mean(-jnp.sum(p*jnp.log(jnp.maximum(p,1e-300)),-1))/math.log(p.shape[-1]))})
                jac=np.asarray(jfn(xs,small_params[layer])).reshape(xs.size,xs.size)
                # Singular values of J and J^T agree; this is the exact
                # backward operator spectrum for the declared prototype.
                sv=np.linalg.svd(jac,compute_uv=False)
                total=jac@total
                xs=smallfn(xs,small_params[layer])
                row={"seed":seed,"variant":variant,"gate":gate,"layer":layer+1,
                     "singular_values":sv.tolist(),"min":float(sv.min()),"max":float(sv.max()),
                     "mean_squared_singular_value":float(np.mean(sv*sv))}
                if layer+1 in [12,24]:
                    total_sv=np.linalg.svd(total,compute_uv=False)
                    row["stack_singular_values"]=total_sv.tolist()
                    row["lyapunov_exponents"]=(np.log(np.maximum(total_sv,1e-300))/(layer+1)).tolist()
                spectral.append(row)
            print(f"propagation seed={seed} {variant}/{gate}",flush=True)
    return {"forward":forward,"spectral":spectral,
            "scope":{"depths":[12,24],"forward_shape":[64,64],"forward_head_dim":64,
                     "exact_spectral_shape":[4,16],"jacobian_dimension":64,
                     "residual_branch_scale":0.5,"initialization":"paired Gaussian weights, pre-LayerNorm residual stack"},
            "interpretation":"Empirical F(q), G(c) trajectories and exact small-stack spectra; neither an infinite-width theorem nor a full-size ViT-B/L claim."}
