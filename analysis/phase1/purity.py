"""CPU research-primitive HLO audit, not the Phase 2 full TPU model audit."""

import re

import jax
import jax.numpy as jnp
import numpy as np

from . import references as ref

FORBIDDEN={"exponential","log","log-plus-one","sine","cosine","tanh","logistic","erf","power","sqrt"}


def scan(text):
    # Match opcodes, not source names, comments or e.g. exponential in a name.
    counts={}
    for line in text.splitlines():
        match=re.search(r"=\s+(?:[A-Za-z0-9]+\[[^\]]*\](?:\{[^}]*\})?|\([^=]*\))\s+([a-z][a-z0-9-]*)\(",line)
        if match:
            op=match.group(1)
            counts[op]=counts.get(op,0)+1
    if not counts:
        raise AssertionError("HLO scanner recognized no instructions")
    return counts


def run():
    q=jnp.ones((4,8),jnp.float32)*0.1
    h=jnp.ones((4,8),jnp.float32)*0.1
    z=jnp.array([[0.2,-0.3,0.8,0.1]],jnp.float32)
    y=jnp.array([[0.925,0.025,0.025,0.025]],jnp.float32)
    funcs={"v2_projected_attention":(lambda q:jnp.sum(ref.attention(q,q,"v2",1.1,2.)**2),q),
           "v2_balanced_attention":(lambda q:jnp.sum(ref.attention(q,q,"v2_balanced",1.1,2.)**2),q),
           "v2_rational_tail_attention":(lambda q:jnp.sum(ref.attention(q,q,"v2_tail")**2),q),
           "projective_positions":(lambda q:jnp.sum(ref.bicayley(q,2)**2),q),
           "lorentz_pool":(lambda h:jnp.sum(ref.lorentz_pool(h)**2),h),
           "inverse_energy_pool":(lambda h:jnp.sum(ref.robust_pool(h)**2),h),
           "power_head":(lambda z:jnp.sum(ref.head_loss(z,y)),z),
           "brier_head":(lambda z:jnp.sum(ref.head_loss(z,y,"brier")),z),
           "rgelu":(lambda h:jnp.sum(ref.rgelu(h)),h),
           "alu":(lambda h:jnp.sum(ref.coupled_gate(h)),h)}
    records={}
    for name,(f,arg) in funcs.items():
        hlo=jax.jit(jax.value_and_grad(f)).lower(arg).compiler_ir("hlo").as_hlo_text()
        counts=scan(hlo)
        forbidden={op:n for op,n in counts.items() if op in FORBIDDEN}
        if forbidden:
            raise AssertionError((name,forbidden))
        records[name]={"instruction_counts":counts,"forbidden":forbidden}
    # Complete two-layer synthetic loss, backward and SGD update. This stays
    # within Phase 1; it is not a full-scale model or TPU compiler audit.
    from .convergence import logits, objective
    from .propagation import init_blocks
    rng=np.random.default_rng(42)
    params={"embed":jnp.asarray(rng.normal(size=(8,32)),jnp.float32),
            "posemb":jnp.zeros((4,32),jnp.float32),
            "blocks":jax.tree.map(lambda a:a.astype(jnp.float32),init_blocks(rng,2,32)),
            "head":jnp.asarray(rng.normal(size=(32,4)),jnp.float32)}
    x=jnp.asarray(rng.normal(size=(2,4,8)),jnp.float32)
    labels=jnp.array([0,1])
    for variant in ["algebraic_v1","v2_proposed","v2_rational_tail"]:
        def step(p):
            loss=lambda pp:objective(logits(pp,x,variant,"alu" if variant=="v2_rational_tail" else "rgelu","inverse_energy" if variant=="v2_rational_tail" else "lorentz"),labels,variant)
            value,grad=jax.value_and_grad(loss)(p)
            return value,jax.tree.map(lambda a,g:a-0.001*g,p,grad)
        counts=scan(jax.jit(step).lower(params).compiler_ir("hlo").as_hlo_text())
        forbidden={op:n for op,n in counts.items() if op in FORBIDDEN}
        if forbidden:
            raise AssertionError((variant,forbidden))
        records[f"two_layer_step_{variant}"]={"instruction_counts":counts,"forbidden":forbidden}
    ce=lambda z:-jnp.sum(y*jnp.log(ref.head_probabilities(z)))
    control=scan(jax.jit(jax.value_and_grad(ce)).lower(z).compiler_ir("hlo").as_hlo_text())
    if control.get("log",0)==0:
        raise AssertionError("negative control did not detect cross-entropy log")
    return {"records":records,"exact_mle_forward_negative_control":control,
            "scope":"unoptimized CPU HLO of research primitives and complete two-layer synthetic loss/backward/SGD steps; not a full-size training-step or TPU purity certificate",
            "elementwise_counts":[{"n":n,"d_h":64,"v1_attention_forward_rsqrt":n*n,"v2_attention_forward_rsqrt":2*n,
                                   "v2_balanced_forward_rsqrt":3*n,"rational_tail_forward_rsqrt":0,
                                   "v1_weight_squarings":3*n*n,"v2_weight_squarings":3*n*n,
                                   "shared_qk_and_pv_matmul_flops":4*n*n*64,
                                   "v2_ball_lift_extra_flops_approx":8*n*64,
                                   "note":"scalar operation elements per head; graph instruction counts differ after vectorization/fusion; balancing also requires an N² pass"} for n in [197,256]],
            "position_roots":{"v1_per_table_axis":14*32,"v2":0},
            "head_nested_roots":{"v1_power_63_over_64":6,"v2_power_7_over_8":3,"v2_brier":0}}
