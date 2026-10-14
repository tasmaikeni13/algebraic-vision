"""Numerical and latency validation on every chip, at B/16 and L/16 loads."""

import argparse
import json
import time
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from algebraic_vision.attention import radical_attention,softmax_attention
from algebraic_vision.kernels import fused_attention
from algebraic_vision.flash import flash_attention
from algebraic_vision.rational_kernel import ball_lift,fused_rational,reference as rational_reference
from algebraic_vision.production import atomic_json,source_hash
from scripts.audit_production import scan_jaxpr,scan_hlo


def milliseconds(fn,args,reps):
    jax.block_until_ready(fn(*args))
    samples=[]
    for _ in range(reps):
        start=time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter()-start)*1000)
    return float(np.median(samples))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed",action="store_true")
    parser.add_argument("--reps",type=int,default=20)
    parser.add_argument("--output",type=Path,default=Path("results/verification/kernels_tpu.json"))
    args=parser.parse_args()
    jax.config.update("jax_compilation_cache_dir", os.environ.get("JAX_COMPILATION_CACHE_DIR", "/dev/shm/algebraic-jax-cache"))
    if args.distributed:
        jax.distributed.initialize()
    mesh=Mesh(np.asarray(jax.devices()),("data",))
    shard,rep=NamedSharding(mesh,P("data")),NamedSharding(mesh,P())
    record={"backend":jax.default_backend(),"device_kind":jax.devices()[0].device_kind,
             "devices":jax.device_count(),"processes":jax.process_count(),"source_sha256":source_hash(),"records":[]}
    for scale,batch,heads in [("vit-b16",1024,12),("vit-l16",512,16)]:
        for dtype in [jnp.float32,jnp.bfloat16]:
            jax.config.update("jax_default_matmul_precision", "float32" if dtype==jnp.float32 else "bfloat16")
            shape=(batch,heads,196,64)
            local=(batch//jax.process_count(),)+shape[1:]
            rng=np.random.default_rng([42,jax.process_index()])
            arrays=[jax.make_array_from_process_local_data(shard,rng.normal(size=local).astype(np.float32),shape).astype(dtype) for _ in range(4)]
            q,k,v,g=arrays
            for arm in ("standard","algebraic","algebraic_v2","flash_standard","rational_ablation"):
                rational=arm=="rational_ablation"
                kind="softmax" if arm in ("standard","flash_standard") else "radical"
                marg=jax.device_put(jnp.full((heads,),1.1 if rational else 0.0,jnp.float32),rep)
                qq,kk=(ball_lift(q),ball_lift(k)) if rational else (q,k)
                def ref(q,k,v,m):
                    return rational_reference(q,k,v,m) if rational else (softmax_attention(q,k,v) if kind=="softmax" else radical_attention(q,k,v,m))
                def kernel(q,k,v,m):
                    if arm=="flash_standard":
                        return flash_attention(q,k,v)
                    return fused_rational(q,k,v,m) if rational else fused_attention(q,k,v,kind,m)
                def mapped(fn,backward=False):
                    if backward:
                        def local_fn(q,k,v,m,g):
                            gradients=jax.grad(lambda q,k,v,m:jnp.sum(fn(q,k,v,m).astype(jnp.float32)*g),argnums=(0,1,2,3))(q,k,v,m)
                            return (*gradients[:3],jax.lax.psum(gradients[3],"data"))
                        specs=(P("data"),P("data"),P("data"),P(),P("data"))
                        outputs=(P("data"),P("data"),P("data"),P())
                    else:
                        local_fn=fn
                        specs=(P("data"),P("data"),P("data"),P())
                        outputs=P("data")
                    return jax.jit(shard_map(local_fn,mesh=mesh,in_specs=specs,out_specs=outputs,check_rep=False))
                forward,backward=mapped(kernel),mapped(kernel,True)
                fargs=(qq,kk,v,marg)
                got,want=forward(*fargs),mapped(ref)(*fargs)
                grads,expected=backward(*fargs,g),mapped(ref,True)(*fargs,g)
                error=float(jnp.max(jnp.abs(got.astype(jnp.float32)-want.astype(jnp.float32))))
                gf,wf=got.astype(jnp.float32),want.astype(jnp.float32)
                forward_relative=float(jnp.sqrt(jnp.sum((gf-wf)*(gf-wf))/jnp.maximum(jnp.sum(wf*wf),1e-30)))
                reference_peak=float(jnp.max(jnp.abs(wf)))
                gradient_errors=[]
                for a,b in zip(grads,expected):
                    a,b=a.astype(jnp.float32),b.astype(jnp.float32)
                    gradient_errors.append({"max_abs":float(jnp.max(jnp.abs(a-b))),
                            "relative_l2":float(jnp.sqrt(jnp.sum((a-b)*(a-b))/jnp.maximum(jnp.sum(b*b),1e-30)))})
                tolerance=0.03 if dtype==jnp.bfloat16 else 0.001
                forward_bound=0.02*reference_peak if dtype==jnp.bfloat16 else 5e-5
                relative_bound=0.01 if dtype==jnp.bfloat16 else 0.0001
                if error>forward_bound or forward_relative>relative_bound or any(e["relative_l2"]>tolerance for e in gradient_errors):
                    raise AssertionError(f"kernel reference mismatch: {scale}/{arm}/{dtype}: {error}, {gradient_errors}")
                nested=scan_jaxpr(jax.make_jaxpr(backward)(*fargs,g))
                hlo=scan_hlo(backward.lower(*fargs,g).compile().as_text())
                if arm not in ("standard","flash_standard") and (not nested["pure"] or not hlo["pure"]):
                    raise AssertionError("impure algebraic kernel")
                if rational and nested["rsqrt"]:
                    raise AssertionError("root in rational inner kernel")
                row={"scale":scale,"arm":arm,"dtype":str(jnp.dtype(dtype)),"shape":list(shape),
                     "matmul_precision":"float32" if dtype==jnp.float32 else "bfloat16",
                     "forward_ms":milliseconds(forward,fargs,args.reps),
                     "backward_ms":milliseconds(backward,(*fargs,g),args.reps),
                     "forward_max_abs_error":error,"gradient_errors_q_k_v_margin":gradient_errors,
                     "forward_relative_l2_error":forward_relative,"forward_abs_bound":forward_bound,
                     "forward_relative_bound":relative_bound,"gradient_relative_bound":tolerance,
                     "nested_programs":nested,"hlo":hlo,"passed":True}
                record["records"].append(row)
                if jax.process_index()==0:
                    print(f"{scale}/{arm}/{row['dtype']}: forward={row['forward_ms']:.3f}ms, backward={row['backward_ms']:.3f}ms, error={error:.2g}",flush=True)
                    atomic_json(args.output,record)
    if jax.process_index()==0:
        for row in record["records"]:
            for reference_arm in ("standard","flash_standard"):
                standard=next(r for r in record["records"] if r["scale"]==row["scale"] and r["dtype"]==row["dtype"] and r["arm"]==reference_arm)
                row[f"latency_ratio_to_{reference_arm}"]=(row["forward_ms"]+row["backward_ms"])/(standard["forward_ms"]+standard["backward_ms"])
        atomic_json(args.output,record)


if __name__=="__main__":
    main()
