"""2/4-layer synthetic patch classification, LR grid, NTK and head diagnostics."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from algebraic_vision.losses import power_score

from . import references as ref
from .propagation import block, init_blocks
from .simulations import layer_norm


def make_data(seed):
    rng=np.random.default_rng(seed+21000)
    patterns=rng.normal(size=(4,4,8))
    # Four patches with eight channels, plus randomized background noise.
    # Training and held-out examples share class patterns but not noise.
    y=np.arange(128)%4
    x=patterns[y]+0.6*rng.normal(size=(128,4,8))
    order=rng.permutation(128)
    x,y=x[order],y[order]
    return jnp.asarray(x[:64]),jnp.asarray(y[:64]),jnp.asarray(x[64:]),jnp.asarray(y[64:])


def logits(params,x,variant,gate="rgelu",pool="lorentz"):
    x=x@params["embed"]+params["posemb"]
    for p in params["blocks"]:
        x=jax.vmap(lambda a:block(a,p,variant,gate)[0])(x)
    if variant.startswith("v2"):
        h=ref.robust_pool(x) if pool=="inverse_energy" else ref.lorentz_pool(x)
        h=layer_norm(h)
    else:
        h=jnp.mean(layer_norm(x),-2)
    return h@params["head"]


def objective(z,y,variant,head="power"):
    target=jax.nn.one_hot(y,4)*0.9+0.025
    if variant=="standard":
        return -jnp.mean(jnp.sum(target*jax.nn.log_softmax(z),-1))
    if variant=="algebraic_v1":
        return power_score(z,y,alpha_exponent=-6,order=64,smoothing=0.1)
    return jnp.mean(ref.head_loss(z,target,head))


def run(seeds,steps=200):
    records,ntks=[],[]
    variants=[("standard","rgelu","power","lorentz"),
              ("algebraic_v1","rgelu","power","lorentz"),
              ("v2_proposed","rgelu","power","lorentz"),
              ("v2_sharp","rgelu","power","lorentz"),
              ("v2_sharp","alu","power","inverse_energy"),
              ("v2_rational_tail","alu","power","inverse_energy"),
              ("v2_sharp","alu","brier","inverse_energy")]
    lrs=np.geomspace(1e-4,1e-2,5)
    for seed in seeds:
        x,y,xval,yval=make_data(seed)
        for depth in [2,4]:
            rng=np.random.default_rng(seed+depth*3000)
            params={"embed":jnp.asarray(rng.normal(size=(8,32))/math.sqrt(8)),
                    "posemb":jnp.asarray(rng.normal(size=(4,32))*0.05),
                    "blocks":init_blocks(rng,depth,32),
                    "head":jnp.asarray(rng.normal(size=(32,4))*0.05)}
            for variant,gate,head,pool in variants:
                name=f"{variant}/{gate}/{head}/{pool}"
                model=lambda p,a:logits(p,a,variant,gate,pool)
                loss=lambda p:objective(model(p,x),y,variant,head)
                val_grad=jax.value_and_grad(loss)

                def train(initial,lr):
                    velocity=jax.tree.map(jnp.zeros_like,initial)
                    def step(state,_):
                        p,velocity=state
                        value,grad=val_grad(p)
                        velocity=jax.tree.map(lambda m,g:0.9*m+g,velocity,grad)
                        p=jax.tree.map(lambda w,g:w-lr*g,p,velocity)
                        return (p,velocity),value
                    (p,_),history=jax.lax.scan(step,(initial,velocity),None,length=steps)
                    return history,model(p,xval),model(p,x)

                fn=jax.jit(train)
                init_loss=float(loss(params))
                for lr in lrs:
                    history,pred,ptrain=fn(params,jnp.asarray(lr))
                    h=np.asarray(history)
                    finite=bool(np.all(np.isfinite(h)) and np.all(np.isfinite(np.asarray(pred))))
                    stable=bool(finite and h[-1]<init_loss and h.max()<=init_loss+max(1.,abs(init_loss)))
                    threshold=init_loss-0.5*(init_loss-h[-1]) if finite else -math.inf
                    hits=np.where(h<=threshold)[0] if finite else np.array([])
                    records.append({"seed":seed,"depth":depth,"variant":name,"lr":float(lr),"finite":finite,"stable":stable,
                                    "initial_loss":init_loss,"final_loss":float(h[-1]) if finite else None,
                                    "heldout_accuracy":float(jnp.mean(jnp.argmax(pred,-1)==yval)) if finite else None,
                                    "train_accuracy":float(jnp.mean(jnp.argmax(ptrain,-1)==y)) if finite else None,
                                    "half_improvement_step":int(hits[0]) if hits.size else None,
                                    "loss_history":[float(a) if np.isfinite(a) else None for a in h]})
                # Empirical multi-output NTK at the matched initialization.
                jac=jax.jacrev(lambda p:model(p,x[:8]).ravel())(params)
                flat=np.concatenate([np.asarray(a).reshape(32,-1) for a in jax.tree.leaves(jac)],1)
                gram=flat@flat.T
                eig=np.linalg.eigvalsh(gram)
                threshold=max(eig.max()*1e-10,1e-14)
                pos=eig[eig>threshold]
                ntks.append({"seed":seed,"depth":depth,"variant":name,"eigenvalues":eig.tolist(),
                             "rank":len(pos),"dimension":32,"threshold":float(threshold),
                             "condition_positive_subspace":float(pos.max()/pos.min()) if len(pos) else None,
                             "full_rank":bool(len(pos)==32)})
                print(f"convergence seed={seed} depth={depth} {name}",flush=True)
    # Head-only exact/quarter VJP optimization diagnostics. These deliberately
    # use a transcendental objective for observation; they are not pure losses.
    head_records=[]
    for seed in seeds:
        rng=np.random.default_rng(seed+44000)
        a=jnp.asarray(rng.normal(size=(32,8)))
        labels=jnp.asarray(rng.integers(0,4,size=32))
        target=jax.nn.one_hot(labels,4)*0.9+0.025
        init=jnp.asarray(rng.normal(size=(8,4))*0.1)
        for tau in [2.,4.]:
            for exponent in [0.5,0.25]:
                for lr in lrs:
                    def step(w,_):
                        z=a@w
                        p=ref.head_probabilities(z,tau)
                        ce=-jnp.mean(jnp.sum(target*jnp.log(p),-1))
                        g=a.T@ref.mle_gradient(z,target,tau,exponent)/len(a)
                        return w-lr*g,ce
                    _,hist=jax.jit(lambda w:jax.lax.scan(step,w,None,length=steps))(init)
                    head_records.append({"seed":seed,"tau":tau,"exponent":exponent,"lr":float(lr),
                                         "initial_ce":float(hist[0]),"final_ce":float(hist[-1]),
                                         "finite":bool(jnp.all(jnp.isfinite(hist)))})
    return {"records":records,"ntk":ntks,"head_gradient_diagnostics":head_records,
            "scope":{"depths":[2,4],"width":32,"patches":4,"train_examples":64,"heldout_examples":64,
                     "classes":4,"steps":steps,"lr_grid":lrs.tolist(),"optimizer":"SGD with momentum 0.9, full batch"},
            "mslr_definition":"largest tested LR with finite decreasing loss and no excursion > initial+max(1,|initial|); right-censored if 0.01 is stable",
            "limitations":"Synthetic prototypes cannot establish ImageNet convergence, full-size ViT-L stability, or a maximum beyond the tested LR range."}
