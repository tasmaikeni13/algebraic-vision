"""Exhaustive floating-point primitive audit, separate from integer indexing.

This traces the complete update and all Pallas forward/VJP bodies at both
full model scales. Optimized TPU HLO is checked separately by the smoke run.
"""

from collections import Counter
import hashlib
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,PartitionSpec as P,NamedSharding

from algebraic_vision.production import atomic_json,build_steps,source_hash
from algebraic_vision.train import build_model_config
from algebraic_vision.model import init_params
from experiments.production import base

ARITHMETIC={"add","add_any","sub","mul","div","neg","rsqrt","max","min",
            "reduce_sum","reduce_max","reduce_min","select_n","dot_general","conv_general_dilated","integer_pow"}
STRUCTURAL={"broadcast_in_dim","reshape","transpose","concatenate","split","slice","squeeze",
            "gather","scatter-add","scatter","pad","convert_element_type","bitcast_convert_type","iota","device_put","psum","pmean",
            "get","swap","pjit","shard_map","custom_jvp_call","custom_vjp_call","pallas_call","closed_call","remat2","scan"}


def inspect(closed):
    floats,indices,bad=Counter(),[],[]
    seen=set()
    def visit(obj):
        if hasattr(obj,"jaxpr"):visit(obj.jaxpr)
        elif hasattr(obj,"eqns"):
            if id(obj) in seen:return
            seen.add(id(obj))
            for eq in obj.eqns:
                name=eq.primitive.name
                dtypes=[getattr(v.aval,"dtype",None) for v in eq.outvars]
                is_float=any(t is not None and jnp.issubdtype(t,jnp.floating) for t in dtypes)
                if is_float:
                    floats[name]+=1
                    if name not in ARITHMETIC|STRUCTURAL:bad.append({"primitive":name,"dtypes":[str(t) for t in dtypes]})
                if name=="integer_pow" and not isinstance(eq.params["y"],int):bad.append({"primitive":"noninteger_pow"})
                if name in {"sign","rem"}:indices.append({"primitive":name,"dtypes":[str(t) for t in dtypes],"floating":is_float})
                for value in eq.params.values():visit(value)
        elif isinstance(obj,dict):
            for value in obj.values():visit(value)
        elif isinstance(obj,(list,tuple)):
            for value in obj:visit(value)
    visit(closed)
    return {"pure":not bad,"unexpected_float_operations":bad,"floating_primitives":dict(floats),"integer_index_operations":indices}


def main():
    mesh=Mesh(np.asarray(jax.devices()[:1]),("data",))
    rep,shard=NamedSharding(mesh,P()),NamedSharding(mesh,P("data"))
    records=[]
    for scale in ("vit-b16","vit-l16"):
        for arm in ("algebraic","algebraic_v2"):
            cfg=base(scale,stage="smoke",data_root="")
            cfg.arm,cfg.loss_backend=arm,"pallas"
            cfg.model["attention_backend"]="pallas"
            mc=build_model_config(cfg)
            abstract=jax.eval_shape(lambda k:init_params(mc,k),jax.random.PRNGKey(42))
            params=jax.tree.map(lambda a:jax.ShapeDtypeStruct(a.shape,a.dtype,sharding=rep),abstract)
            state={"m":params,"v":params}
            x=jax.ShapeDtypeStruct((cfg.batch_size,224,224,3),jnp.float32,sharding=shard)
            labels=jax.ShapeDtypeStruct((cfg.batch_size,),jnp.int32,sharding=shard)
            targets=jax.ShapeDtypeStruct((cfg.batch_size,1000),jnp.float32,sharding=shard)
            scalars=jax.ShapeDtypeStruct((3,),jnp.float32,sharding=rep)
            key=jax.ShapeDtypeStruct((2,),jnp.uint32,sharding=rep)
            train,_,_=build_steps(cfg,mc,mesh,donate=False)
            result=inspect(jax.make_jaxpr(train)(params,state,x,targets,labels,scalars,key))
            records.append(dict(result,arm=arm,scale=scale))
            print(scale,arm,result["pure"],result["unexpected_float_operations"],flush=True)
    atomic_json("results/verification/typed_purity.json",{"records":records,"source_sha256":source_hash(),
                "producer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope":"Full B/L model and optimizer, recursive Pallas programs; one abstract CPU shard for tracing only. Real TPU execution/HLO evidence comes from smoke audits."})
    if not all(r["pure"] for r in records):raise SystemExit(1)


if __name__=="__main__":main()
