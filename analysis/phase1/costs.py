"""Phase 1 changed-component forward/backward cost study.

No production models or training launchers are constructed. Components are
lowered at B/L tensor shapes without executing a model or reserving a TPU.
CPU XLA estimates are reported separately from integer source-program counts.
The latter count every fixed scan trip and every learned-parameter VJP.
Neither measure is TPU wall-clock evidence.
"""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

import jax
import jax.numpy as jnp

from algebraic_vision.attention import radical_attention
from algebraic_vision.losses import power_score,smoothed_targets
from algebraic_vision.model import layer_norm,mixed_rotary_tables,ViTConfig
from algebraic_vision.primitives import radical_gelu,alu,rotate_pairs
from . import references as ref,revisions as rev
from .purity import scan,FORBIDDEN
from .operation_count import forward_backward

ROOT=Path(__file__).resolve().parents[2]


def abstract(shape,dtype=jnp.float32):
    return jax.ShapeDtypeStruct(shape,dtype)


def measure(fn,arguments,output_shapes):
    """Include the primal and every input's VJP; the cotangent is runtime data."""
    def both(*args):
        inputs,cot=args[:-1],args[-1]
        out,pull=jax.vjp(fn,*inputs)
        return out,pull(cot)
    lowered=jax.jit(both).lower(*arguments,output_shapes)
    ops=scan(lowered.compiler_ir("hlo").as_hlo_text())
    forbidden={op:n for op,n in ops.items() if op in FORBIDDEN}
    if forbidden:
        raise AssertionError(forbidden)
    # Lowered analysis retains algorithm-level arithmetic before target code
    # generation. Compiled analysis is a separate, potentially fused estimate.
    before=lowered.cost_analysis()
    compiled=lowered.compile()
    after=compiled.cost_analysis()
    if not isinstance(before,dict) or not isinstance(after,dict):
        raise RuntimeError("pinned CPU backend did not supply scalar cost dictionaries")
    return {"lowered":{k:float(v) for k,v in before.items()},
            "compiled":{k:float(v) for k,v in after.items()},
            "static":forward_backward(fn,arguments,output_shapes),
            "hlo_instruction_counts":ops,"forbidden":forbidden}


def combine(p,v):
    return jnp.einsum("...ij,...jd->...id",p.astype(v.dtype),v,
                      preferred_element_type=jnp.float32).astype(v.dtype)


def attention_candidate(name):
    if name in {"algebraic_v1","retained_e8"}:
        return lambda q,k,v:radical_attention(q,k,v,jnp.zeros((q.shape[1],),jnp.float32))
    def f(q,k,v,*parameters):
        if name=="projected16":
            margin,gain_q,gain_k=parameters
            p=rev.projected(q,k,ref.constrained_margin(margin),0.0625+gain_q*gain_q,
                            0.0625+gain_k*gain_k,order=16,mxu_round=True)
        elif name=="projected8":
            margin,gain_q,gain_k=parameters
            p=rev.projected(q,k,ref.constrained_margin(margin),0.0625+gain_q*gain_q,
                            0.0625+gain_k*gain_k,order=8,mxu_round=True)
        elif name=="rational_tail":
            p=rev.tail_rows(rev.scores(q,k))
        elif name=="pade":
            p=rev.pade_rows(rev.scores(q,k))
        elif name=="pade_newton2":
            p=rev.pade_rows(rev.scores(q,k),newton_steps=2)
        elif name=="pade_newton1":
            p=rev.pade_rows(rev.scores(q,k),newton_steps=1)
        else:
            raise ValueError(name)
        return combine(p,v)
    return f


def selected_static_budget(batch,n,grid,depth,width,heads,hidden):
    """Full changed-component counts, including all learned-position VJPs."""
    a=abstract((batch,1,n,64),jnp.bfloat16)
    common=forward_backward(attention_candidate("algebraic_v1"),[a,a,a],a)
    activation=abstract((batch,n,hidden))
    old_gate=forward_backward(radical_gelu,[activation],activation)
    new_gate=forward_backward(rev.rational_gate,[activation],activation)
    residual=abstract((batch,n,width))
    norm={"scale":abstract((width,)),"bias":abstract((width,))}
    old_pool=forward_backward(lambda x,p:jnp.mean(layer_norm(x,p),-2),[residual,norm],abstract((batch,width)))
    new_pool=forward_backward(lambda x,p:layer_norm(ref.robust_pool(x),p),[residual,norm],abstract((batch,width)))
    q=abstract((batch,heads,n,64),jnp.bfloat16)
    freq=abstract((heads,16))
    new_pos=forward_backward(lambda q,k,rx,ry:rev.learned_bicayley(q,k,rx,ry,grid),[q,q,freq,freq],(q,q))
    cfg=ViTConfig(image_size=16*grid,patch_size=16,width=width,heads=heads,
                  pool="cls" if n==197 else "gap",rotary="cayley2d_mixed")
    def old_positions(q,k,wx,wy):
        tables=mixed_rotary_tables(cfg,{"wx":wx,"wy":wy})
        return rotate_pairs(q,*tables),rotate_pairs(k,*tables)
    old_freq=abstract((heads,32))
    old_pos=forward_backward(old_positions,[q,q,old_freq,old_freq],(q,q))
    z=abstract((batch,1000))
    label=jnp.zeros((batch,),jnp.int32)
    target=smoothed_targets(label,1000,0.1)
    old_head=forward_backward(lambda z:power_score(z,label,alpha_exponent=-6,order=64,smoothing=0.1,reduction="none"),[z],abstract((batch,)))
    new_head=forward_backward(lambda z:ref.head_loss(z,target),[z],abstract((batch,)))
    def total(gate,pos,pool,head):
        return {metric:depth*(heads*common[metric]+gate[metric]+pos[metric])+pool[metric]+head[metric] for metric in common}
    return {"batch":batch,"baseline":total(old_gate,old_pos,old_pool,old_head),"candidate":total(new_gate,new_pos,new_pool,new_head)}


def run():
    # Calibrate the pinned compiler's cost categories rather than assuming
    # its "transcendentals" label means a forbidden exponential/logarithm.
    calibrations={}
    for name,f in [("rsqrt",jax.lax.rsqrt),("divide",lambda x:1/x),("multiply_add",lambda x:x*x+1)]:
        info=jax.jit(f).lower(abstract((64,))).cost_analysis()
        calibrations[name]={k:float(v) for k,v in info.items()}
    if calibrations["rsqrt"].get("transcendentals")!=64 or calibrations["multiply_add"].get("flops")!=128:
        raise RuntimeError("cost calibration differs from the pinned audit convention")
    cache={}
    records=[]
    names=["algebraic_v1","retained_e8","projected8","projected16","rational_tail","pade","pade_newton1","pade_newton2"]
    for shape,depth,width,heads,hidden in [("B/16",12,768,12,3072),("L/16",24,1024,16,4096)]:
        for n,grid in [(197,14),(256,16)]:
            row={"shape":shape,"depth":depth,"width":width,"heads":heads,"tokens":n,"grid":grid,"batch":1,"components":{}}
            # Count one head, then scale by the number of heads and layers.
            key=("attention",n)
            if key not in cache:
                args=[abstract((1,1,n,64),jnp.bfloat16)]*3
                cache[key]={name:measure(attention_candidate(name),args+([abstract((1,1,1))]*3 if name.startswith("projected") else []),args[0]) for name in names}
            row["components"]["attention_per_head"]=cache[key]
            activation=abstract((1,n,hidden))
            row["components"]["rgelu"]=measure(radical_gelu,[activation],activation)
            row["components"]["alu"]=measure(alu,[activation],activation)
            row["components"]["rational_gate"]=measure(rev.rational_gate,[activation],activation)
            residual=abstract((1,n,width))
            norm= {"scale":abstract((width,)),"bias":abstract((width,))}
            def pool_v1(x,p):
                return jnp.mean(layer_norm(x,p),-2)
            def pool_energy(x,p):
                return layer_norm(ref.robust_pool(x),p)
            row["components"]["pool_v1"]=measure(pool_v1,[residual,norm],abstract((1,width)))
            row["components"]["pool_energy"]=measure(pool_energy,[residual,norm],abstract((1,width)))
            # Every grid step is explicitly unrolled in this research cost
            # reference. Charges include all position-frequency gradients.
            q=abstract((1,heads,n,64),jnp.bfloat16)
            freq=abstract((heads,16))
            pos=lambda q,k,rx,ry:rev.learned_bicayley(q,k,rx,ry,grid)
            row["components"]["positions_v2"]=measure(pos,[q,q,freq,freq],(q,q))
            config=ViTConfig(image_size=16*grid,patch_size=16,width=width,heads=heads,
                             pool="cls" if n==197 else "gap",rotary="cayley2d_mixed")
            def positions_v1(q,k,wx,wy):
                tables=mixed_rotary_tables(config,{"wx":wx,"wy":wy})
                return rotate_pairs(q,*tables),rotate_pairs(k,*tables)
            old_freq=abstract((heads,32))
            row["components"]["positions_v1"]=measure(positions_v1,[q,q,old_freq,old_freq],(q,q))
            logits=abstract((1,1000))
            label=jnp.zeros((1,),jnp.int32)
            target=smoothed_targets(label,1000,0.1)
            if "heads" not in cache:
                cache["heads"]={
                    "v1":measure(lambda z:power_score(z,label,alpha_exponent=-6,order=64,smoothing=0.1,reduction="none"),[logits],abstract((1,))),
                    "v2_power":measure(lambda z:ref.head_loss(z,target),[logits],abstract((1,)))}
            row["components"]["loss_heads"]=cache["heads"]
            comparisons={}
            for name in names[1:]:
                budget={}
                for analysis in ["lowered","compiled","static"]:
                    for metric in (["flops","transcendentals"] if analysis!="static" else ["arithmetic","matmul","rsqrt","minmax","comparisons","selects"]):
                        get=lambda component:component[analysis].get(metric,0.)
                        c=row["components"]
                        old_position=c["positions_v1"][analysis].get(metric,0.)
                        old=depth*(heads*get(c["attention_per_head"]["algebraic_v1"])+get(c["rgelu"])+old_position)+get(c["pool_v1"])+get(c["loss_heads"]["v1"])
                        new=depth*(heads*get(c["attention_per_head"][name])+get(c["rational_gate"])+get(c["positions_v2"]))+get(c["pool_energy"])+get(c["loss_heads"]["v2_power"])
                        budget[f"{analysis}_{metric}"]={"baseline_changed_component_lower_budget":old,"candidate_changed_component_budget":new,"candidate_minus_baseline":new-old,"non_regression":new<=old}
                comparisons[name]=budget
            row["comparisons"]=comparisons
            row["optimizer_parameter_delta"]=depth*(width//2-width)
            row["shared_matmul_train_flops"]=3*depth*(24*n*width*width+4*n*n*width)
            row["assumption"]="Shared linear/norm/embedding/head arithmetic cancels. All learned frequencies are differentiated; the shared retained-E8 attention has zero sink and fixed temperature, as in canonical v1. Static counts include every v1 scan trip; compiler estimates are separate. The selected candidate loses D/2 frequency parameters per layer, so any identical elementwise optimizer costs no more. Optional projected variants separately charge three learned margin/gain controls per head."
            batch_budgets=[selected_static_budget(b,n,grid,depth,width,heads,hidden) for b in [1,2,3]]
            affine={}
            categories=["arithmetic","matmul","minmax","comparisons","selects"]
            for metric in ["floating_work_including_selections","rsqrt"]:
                keys=categories if metric!="rsqrt" else ["rsqrt"]
                deltas=[sum(b["candidate"][k]-b["baseline"][k] for k in keys) for b in batch_budgets]
                slope=deltas[1]-deltas[0]
                affine[metric]={"delta_at_batch_one":deltas[0],"delta_per_extra_example":slope,
                    "batch_three_delta":deltas[2],"affine_crosscheck":deltas[2]==deltas[0]+2*slope,
                    "all_positive_batches_nonregressing":deltas[0]<=0 and slope<=0 and deltas[2]==deltas[0]+2*slope}
            row["batch_budgets"]=batch_budgets
            row["batch_affine_certificate"]=affine
            row["affine_justification"]="Every shape is batch-independent or B times a constant, no product contracts two independent batch axes, and all scan lengths and Python branches are batch-independent. Pointwise counts are linear; reductions across B cost B-1 per fixed group; other reductions are linear. Thus the complete source-program count is affine in B>=1. B=3 is an additional regression crosscheck. Shared optimizer arithmetic is non-increasing with the smaller parameter count. No extra rematerialization, sparsity or data-dependent loop is assumed."
            records.append(row)
            print("COST",shape,n,{name:round(b["compiled_flops"]["candidate_minus_baseline"]) for name,b in comparisons.items()},flush=True)
    paths=[Path(__file__),ROOT/"analysis/phase1/operation_count.py",ROOT/"analysis/phase1/revisions.py",ROOT/"analysis/phase1/references.py",ROOT/"analysis/phase1/purity.py"]
    paths+=list((ROOT/"algebraic_vision").glob("*.py"))
    return {"records":records,"calibration":calibrations,
            "provenance":{"source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},"jax":jax.__version__,"backend":jax.default_backend()},
            "cost_convention":"Integer source-program counts: scalar arithmetic/division each 1, matrix products 2*M*N*K, roots separate, every scan trip included. Min/max, comparisons and selects are also reported separately and can be included conservatively. CPU XLA lowered/compiled estimates are additional evidence; their rsqrt appears in a bucket named transcendentals. HLO scanner independently rejects forbidden mathematical operations.",
            "source":"https://docs.jax.dev/en/latest/aot.html",
            "limitation":"Compiler-specific theoretical estimates, not a mathematical proof of TPU compiler arithmetic or latency. V2 positions are unrolled completely; positive-length v1 scan bodies are costed at least once, so their underestimated trip counts make the comparison harder for v2."}


if __name__=="__main__":
    result=run()
    output=ROOT/"results/phase1/costs.json"
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
