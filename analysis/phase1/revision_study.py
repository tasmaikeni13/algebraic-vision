"""Final Phase 1 revision, isolated from all production architecture code.

Retain E8 attention exactly; select learned bounded Bi-Cayley positions,
inverse-energy pooling, a genuine 7/8 power score and a rational softsign gate.
No claim that this removes the E8 attention roots or its attenuation is made.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("JAX_ENABLE_X64","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import sympy as sp

from . import references as ref,revisions as rev
from .convergence import make_data,objective
from .propagation import block as original_block,cosine,init_blocks
from .purity import scan,FORBIDDEN
from .simulations import layer_norm

ROOT=Path(__file__).resolve().parents[2]
SELECTED="v2_retained_e8"
SECTIONS=["symbolic","attention","positions","pool_precision","propagation","convergence","purity"]


def attention(q,k):
    # Reuse the exact baseline expression, including its precision policy.
    return ref.attention(q,k,"v1")


def revision_block(x,p,variant=SELECTED):
    if variant!=SELECTED:
        return original_block(x,p,variant)
    norm=layer_norm(x)
    q,k,v=[norm@p[name] for name in ["q","k","v"]]
    q,k=rev.learned_bicayley(q[None,None],k[None,None],p["raw_x"],p["raw_y"],math.isqrt(x.shape[-2]))
    probability=attention(q[0,0],k[0,0])
    x=x+0.5*((probability@v)@p["o"])
    hidden=rev.rational_gate(layer_norm(x)@p["up"])
    return x+0.5*(hidden@p["down"]),probability


def add_frequencies(blocks):
    return [dict(p,raw_x=jnp.linspace(0.6,1.4,p["q"].shape[0]//4)[None],
                 raw_y=jnp.linspace(0.8,1.6,p["q"].shape[0]//4)[None]) for p in blocks]


def logits(p,x,variant):
    x=x@p["embed"]+p["posemb"]
    for b in p["blocks"]:
        x=jax.vmap(lambda a:revision_block(a,b,variant)[0])(x)
    h=layer_norm(ref.robust_pool(x)) if variant==SELECTED else jnp.mean(layer_norm(x),-2)
    return h@p["head"]


def symbolic():
    a=sp.symbols("a",nonnegative=True)
    x,c=sp.symbols("x c",real=True,positive=True)
    gplus=x-x/(2*(1+c*x))
    gminus=x/(2*(1-c*x))
    d=4+6*a+5*a*a+2*a**3
    r=a+(4+2*a+a*a)/d
    witness=a**4*(53*a-80)**2+225*a**4+212*a**8+1060*a**7+6572*a**3+4876*a*a+2544*a+848
    checks={
        "positive_gate_derivative":sp.factor(sp.diff(gplus,x)-(1-1/(2*(1+c*x)**2)))==0,
        "negative_gate_derivative":sp.factor(sp.diff(gminus,x)-1/(2*(1-c*x)**2))==0,
        "gate_reflection":sp.factor(gplus-gminus.subs(x,-x)-x)==0,
        "gate_value_matches_zero":gplus.subs(x,0)==gminus.subs(x,0)==0,
        "gate_first_derivative_matches_zero":sp.diff(gplus,x).subs(x,0)==sp.diff(gminus,x).subs(x,0)==sp.Rational(1,2),
        "gate_second_derivative_matches_zero":sp.diff(gplus,x,2).subs(x,0)==sp.diff(gminus,x,2).subs(x,0)==c,
        "pade_residual":sp.factor(r*r-(1+a*a)+2*a**5/d**2)==0,
        "pade_bound_polynomial_witness":sp.expand(53*(d*d*(1+a*a)-256*a**5)-witness)==0,
        "selected_attention_is_baseline_expression":True,
    }
    if not all(checks.values()):raise AssertionError(checks)
    values=[]
    with mp.workdps(50):
        for s in [mp.mpf(0),mp.mpf("1e-12"),mp.mpf(1),mp.mpf(94),mp.mpf(10000)]:
            t=s/8
            den=4+6*t+5*t*t+2*t**3
            rr=t+(4+2*t+t*t)/den
            root=mp.sqrt(1+t*t)
            values.append({"score":str(s),"pade0_relative_radius_error":str(rr/root-1),
                           "pade1_relative_radius_error":str((rr+(1+t*t)/rr)/(2*root)-1)})
    return {"checks":checks,"mpmath_dps":50,"exploratory_pade_oracle":values,
            "selected_equations":{"attention":"exact frozen E8 attention; no ball lift, no column balancing",
                "position_frequency":"(1/G)*(1/4+3/(4*(1+raw^2))) in (1/(4G),1/G] for finite raw",
                "positions":"two planes, phases c*theta_x+r*theta_y and c*theta_x-r*theta_y",
                "pool":"sum a_i*h_i / sum a_i, a_i=d/(d+||h_i||^2), then final LayerNorm",
                "head":"strictly proper power score a=7/8, E8(z/4) link; ordinary exact AD",
                "gate":"g(x)=x/2*(1+c*x/(1+c*|x|)), c=0.851"},
            "position_proof":"0<theta_x,theta_y<=2/G. Every grid-offset phase has absolute value <4<2*pi. Both planes being identity therefore forces both phases to zero, whence dc=dr=0. This holds for all finite raw parameters, not only sampled initializations.",
            "gate_proof":"Both one-sided value/first/second derivatives agree at zero: 0,1/2,c. For x>=0, g'=1-1/(2*(1+c*x)^2); for x<0, g'=1/(2*(1-c*x)^2). Thus 0<g'<1 for finite x and c>0. Negative tail tends to -1/(2c).",
            "attention_limit":"The gate matches v1 attention exactly; g8(94) is still about 0.085. There is no assertion of eliminating this attenuation.",
            "equivalence_policy":"User authorized a derived numerical equivalence tolerance. The selected retained-E8 attention needs no such allowance: it uses the identical expression and precision policy."}


def attention_audit(seeds):
    records=[]
    baseline_grad=jax.jit(jax.value_and_grad(lambda q,k,cot:jnp.sum(ref.attention(q,k,"v1")*cot),argnums=(0,1)))
    candidate_grad=jax.jit(jax.value_and_grad(lambda q,k,cot:jnp.sum(attention(q,k)*cot),argnums=(0,1)))
    pf=jax.jit(attention)
    for distribution in ["gaussian","student_t3"]:
        for seed in seeds:
            rng=np.random.default_rng(seed)
            draw=rng.normal if distribution=="gaussian" else lambda size:rng.standard_t(3,size=size)/math.sqrt(3)
            q0,k0=draw(size=(197,64)),draw(size=(197,64))
            cot=jnp.asarray(rng.normal(size=(197,197)))
            for alpha in [1,2,4,8,10,16,32,64,100,128,256]:
                q,k=jnp.asarray(alpha*q0),jnp.asarray(alpha*k0)
                value,(dq,dk)=candidate_grad(q,k,cot)
                bv,(bq,bk)=baseline_grad(q,k,cot)
                p=pf(q,k)
                scores=q@k.T/8
                slope=jax.lax.rsqrt(1+(scores/8)**2)
                b=slope[...,None]*k[None]/8
                jac=p[...,None]*(b-jnp.sum(p[...,None]*b,1,keepdims=True))
                fq=jnp.sqrt(jnp.sum(jac*jac,(1,2)))
                pp=jnp.sum(p*p,-1,keepdims=True)
                fs=jnp.sqrt(jnp.sum(slope*slope*p*p*((1-p)**2+jnp.maximum(pp-p*p,0)),-1))
                records.append({"distribution":distribution,"seed":seed,"alpha":alpha,
                    "gradient_ratio":float(jnp.linalg.norm(dq)/jnp.linalg.norm(bq)),
                    "value_difference":float(jnp.abs(value-bv)),
                    "q_gradient_max_difference":float(jnp.max(jnp.abs(dq-bq))),"k_gradient_max_difference":float(jnp.max(jnp.abs(dk-bk))),
                    "q_gradient_norm":float(jnp.linalg.norm(dq)),"entropy":float(jnp.mean(-jnp.sum(p*jnp.log(jnp.maximum(p,1e-300)),-1))/math.log(197)),
                    "q_saturated_rows":float(jnp.mean(fq<1e-4)),"score_saturated_rows":float(jnp.mean(fs<1e-4)),
                    "q_jacobian_fro":float(jnp.mean(fq)),"score_jacobian_fro":float(jnp.mean(fs)),
                    "finite":bool(jnp.all(jnp.isfinite(dq))&jnp.all(jnp.isfinite(dk))&jnp.all(jnp.isfinite(p)))})
            print(f"revision attention {distribution} seed={seed}",flush=True)
    precision=[]
    for n in [32,197,256]:
        for seed in seeds:
            rng=np.random.default_rng(seed)
            q0,k0=rng.normal(size=(n,64)),rng.normal(size=(n,64))
            cot=jnp.asarray(rng.normal(size=(n,n)))
            for scale in [1,3,10,30,100]:
                q,k=jnp.asarray(q0*scale),jnp.asarray(k0*scale)
                p64,pull=jax.vjp(attention,q,k)
                g64=pull(cot)[0]
                for dtype in [jnp.float32,jnp.bfloat16]:
                    qd,kd=q.astype(dtype),k.astype(dtype)
                    p,back=jax.vjp(attention,qd,kd)
                    g=back(cot.astype(p.dtype))[0]
                    bp,bback=jax.vjp(lambda q,k:ref.attention(q,k,"v1"),qd,kd)
                    bg=bback(cot.astype(bp.dtype))[0]
                    precision.append({"n":n,"seed":seed,"scale":scale,"dtype":str(jnp.dtype(dtype)),
                        "tv_max":float(jnp.max(0.5*jnp.sum(jnp.abs(p-p64),-1))),
                        "backward_relative_error":float(jnp.linalg.norm(g.astype(jnp.float64)-g64)/jnp.linalg.norm(g64)),
                        "baseline_forward_max_difference":float(jnp.max(jnp.abs(p-bp))),
                        "baseline_backward_max_difference":float(jnp.max(jnp.abs(g-bg))),
                        "finite":bool(jnp.all(jnp.isfinite(p))&jnp.all(jnp.isfinite(g)))})
    radial=[]
    for scale in [10,32,100,256,1024]:
        q=jnp.array([[float(scale),0.,0.,0.]])
        k=jnp.array([[1.,0.,0.,0.],[2.,0.,0.,0.]])
        cot=jnp.array([[1.,0.]])
        _,(g,_)=candidate_grad(q,k,cot)
        _,(b,_)=baseline_grad(q,k,cot)
        radial.append({"alpha":scale,"gradient_ratio":float(g[0,0]/b[0,0]),"absolute_difference":float(abs(g[0,0]-b[0,0]))})
    return {"records":records,"precision":precision,"radial":radial,
            "identity_argument":"Both call the same unchanged E8 reference with the same dtypes. The finite-input comparisons are regression checks of this structural identity, not a statistical proof of two different kernels being equivalent.",
            "scope":"Two distributions, eight seeds, alpha 1..256; FP32/BF16 at N=32/197/256. Original boundary tests and column-balancing ablations remain in precision.json/crowding.json."}


def positions():
    records=[]
    for grid in [14,16,32,64]:
        # Include both limiting frequencies, mixed extremes and initialization.
        for rx,ry in [(0.,0.),(1.,1.),(0.6,1.6),(1e4,1e4),(0.,1e4)]:
            w=np.asarray([rev.frequency(jnp.array(rx),grid),rev.frequency(jnp.array(ry),grid,1)])
            theta=2*np.arctan(w)
            dr,dc=np.meshgrid(np.arange(1-grid,grid),np.arange(1-grid,grid))
            phases=np.stack([dc*theta[0]+dr*theta[1],dc*theta[0]-dr*theta[1]],-1)
            distance=np.sqrt(np.sum(8*np.sin(phases/2)**2,-1))
            mask=(dr!=0)|(dc!=0)
            for dtype in [jnp.float32,jnp.bfloat16]:
                # Nonlinear positional tables are FP32 for BF16 model inputs.
                c,s=rev.projective_tables(jnp.asarray(w,jnp.float32),2*grid)
                err=0.
                for index in range(grid):
                    err=max(err,float(jnp.max(jnp.abs(c[index]*c[:grid]-s[index]*s[:grid]-c[index:index+grid]))),
                            float(jnp.max(jnp.abs(s[index]*c[:grid]+c[index]*s[:grid]-s[index:index+grid]))))
                basis=jnp.tile(jnp.eye(4,dtype=dtype)[:,None,None,:],(1,1,grid*grid,1))
                rotated,_=rev.learned_bicayley(basis,basis,jnp.array([[rx]],jnp.float32),jnp.array([[ry]],jnp.float32),grid)
                matrices=rotated[:,0].transpose(1,2,0).astype(jnp.float32)
                round_error=float(jnp.max(jnp.abs(matrices.transpose(0,2,1)@matrices-jnp.eye(4))))
                unique=len(np.unique(np.asarray(matrices).reshape(grid*grid,-1),axis=0))
                records.append({"grid":grid,"dtype":str(jnp.dtype(dtype)),"raw_x":rx,"raw_y":ry,
                    "table_orthogonality_error":float(jnp.max(jnp.abs(c*c+s*s-1))),"power_addition_error":err,
                    "rounded_matrix_orthogonality_error":round_error,"min_offset_distance":float(distance[mask].min()),
                    "rank_determinant":float(-2*theta[0]*theta[1]),"maximum_phase":float(np.max(abs(phases))),
                    "unique_rounded_grid_matrices":unique})
    return {"records":records,"nonlinear_policy":"FP32 recurrence and arithmetic; BF16 Q/K at MXU boundaries. Rounding an orthogonal matrix is not exact orthogonality.",
            "proof":"All finite raw frequencies lie in (1/(4G),1/G]; hence nonzero local rank and both offset phases strictly in (-4,4), a subset of (-2*pi,2*pi). Only the zero offset is identity."}


def pool_precision():
    rows=[]
    for scale in [1,10,100,1e3,1e4,1e6]:
        initial=np.tile(np.array([1.,-1.,1.,-1.])*scale,(197,1))
        h64=jnp.asarray(initial)
        p64,pull=jax.vjp(ref.robust_pool,h64)
        g64=pull(jnp.ones_like(p64))[0]
        for dtype in [jnp.float32,jnp.bfloat16]:
            h=jnp.asarray(initial,dtype)
            p,pull=jax.vjp(ref.robust_pool,h)
            g=pull(jnp.ones_like(p))[0]
            rows.append({"scale":scale,"dtype":str(jnp.dtype(dtype)),
                "value_relative_error":float(jnp.linalg.norm(p.astype(jnp.float64)-p64)/jnp.linalg.norm(p64)),
                "gradient_relative_error":float(jnp.linalg.norm(g.astype(jnp.float64)-g64)/jnp.linalg.norm(g64)),
                "finite":bool(jnp.all(jnp.isfinite(p))&jnp.all(jnp.isfinite(g)))})
    return {"records":rows,"outlier_evidence":"outliers.json: v2_inverse_energy is the identical selected pool, tested against the actual LN-before-GAP baseline at all original sizes/counts/seeds/scales. This unchanged component's previous results are retained, not relabeled as new independent samples."}


def propagation(seeds):
    forward,spectral=[],[]
    for seed in seeds:
        rng=np.random.default_rng(seed+13000)
        blocks=add_frequencies(init_blocks(rng,24,64))
        initial=jnp.asarray(rng.normal(size=(64,64)))
        small_blocks=add_frequencies(init_blocks(rng,24,16))
        small_initial=jnp.asarray(rng.normal(size=(4,16)))
        for variant in ["standard","algebraic_v1",SELECTED]:
            fn=jax.jit(lambda x,p:revision_block(x,p,variant))
            jfn=jax.jit(jax.jacfwd(lambda x,p:revision_block(x,p,variant)[0],argnums=0))
            x,xs=initial,small_initial
            total=np.eye(xs.size)
            for index in range(24):
                vin,cin=float(jnp.mean(x*x)),float(cosine(x))
                x,p=fn(x,blocks[index])
                eig=np.sort(np.abs(np.linalg.eigvals(np.asarray(p))))[::-1]
                forward.append({"seed":seed,"variant":variant,"layer":index+1,
                    "variance_in":vin,"variance_out":float(jnp.mean(x*x)),"cosine_in":cin,"cosine_out":float(cosine(x)),
                    "spectral_gap":float(1-eig[1]),"entropy":float(jnp.mean(-jnp.sum(p*jnp.log(jnp.maximum(p,1e-300)),-1))/math.log(len(p)))})
                jac=np.asarray(jfn(xs,small_blocks[index])).reshape(xs.size,xs.size)
                sv=np.linalg.svd(jac,compute_uv=False)
                total=jac@total
                xs=fn(xs,small_blocks[index])[0]
                row={"seed":seed,"variant":variant,"layer":index+1,"singular_values":sv.tolist(),
                    "min":float(sv.min()),"max":float(sv.max()),"mean_squared_singular_value":float(np.mean(sv*sv))}
                if index+1 in [12,24]:
                    full=np.linalg.svd(total,compute_uv=False)
                    row["stack_singular_values"]=full.tolist()
                    row["lyapunov_exponents"]=(np.log(np.maximum(full,1e-300))/(index+1)).tolist()
                spectral.append(row)
            print(f"revision propagation seed={seed} {variant}",flush=True)
    return {"forward":forward,"spectral":spectral,
        "scope":{"depths":[12,24],"forward_shape":[64,64],"exact_spectral_shape":[4,16],"jacobian_dimension":64,
                 "residual_branch_scale":0.5,"learned_frequency_parameters_included":True},
        "limitation":"Exact spectra of a declared small state, matched layer counts and 64D forward heads. These are not full-width ViT-B/L spectra or an ImageNet training result."}


def convergence(seeds,steps):
    rows,ntks=[],[]
    lrs=np.geomspace(1e-4,1e-2,5)
    for seed in seeds:
        x,y,xval,yval=make_data(seed)
        for depth in [2,4]:
            rng=np.random.default_rng(seed+depth*3000)
            initial={"embed":jnp.asarray(rng.normal(size=(8,32))/math.sqrt(8)),
                     "posemb":jnp.asarray(rng.normal(size=(4,32))*0.05),
                     "blocks":init_blocks(rng,depth,32),"head":jnp.asarray(rng.normal(size=(32,4))*0.05)}
            for variant in ["standard","algebraic_v1",SELECTED]:
                params=dict(initial,blocks=add_frequencies(initial["blocks"])) if variant==SELECTED else initial
                model=lambda p,a:logits(p,a,variant)
                loss=lambda p:objective(model(p,x),y,variant)
                def loss_aux(p):
                    z=model(p,x)
                    return objective(z,y,variant),jnp.mean(jnp.argmax(z,-1)==y)
                value_grad=jax.value_and_grad(loss_aux,has_aux=True)
                def train(initial,lr):
                    def step(state,_):
                        p,velocity=state
                        (value,accuracy),g=value_grad(p)
                        velocity=jax.tree.map(lambda m,g:0.9*m+g,velocity,g)
                        p=jax.tree.map(lambda p,g:p-lr*g,p,velocity)
                        return (p,velocity),(value,accuracy)
                    (p,_),(hist,accuracies)=jax.lax.scan(step,(initial,jax.tree.map(jnp.zeros_like,initial)),None,length=steps)
                    return hist,accuracies,model(p,xval),model(p,x)
                fn=jax.jit(train)
                init_loss=float(loss(params))
                for lr in lrs:
                    hist,accuracies,pred,ptrain=fn(params,jnp.asarray(lr))
                    h=np.asarray(hist)
                    finite=bool(np.isfinite(h).all() and np.isfinite(np.asarray(pred)).all())
                    stable=bool(finite and h[-1]<init_loss and h.max()<=init_loss+max(1.,abs(init_loss)))
                    hit=np.where(h<=init_loss-0.5*(init_loss-h[-1]))[0] if finite else []
                    accuracy_hits=np.where(np.asarray(accuracies)>=0.95)[0] if finite else []
                    rows.append({"seed":seed,"depth":depth,"variant":variant,"lr":float(lr),"finite":finite,"stable":stable,
                        "initial_loss":init_loss,"final_loss":float(h[-1]) if finite else None,
                        "heldout_accuracy":float(jnp.mean(jnp.argmax(pred,-1)==yval)) if finite else None,
                        "train_accuracy":float(jnp.mean(jnp.argmax(ptrain,-1)==y)) if finite else None,
                        "half_improvement_step":int(hit[0]) if len(hit) else None,
                        "first_train_accuracy_95_step":int(accuracy_hits[0]) if len(accuracy_hits) else None,
                        "train_accuracy_history":[float(a) for a in accuracies],
                        "loss_history":[float(a) if np.isfinite(a) else None for a in h]})
                jac=jax.jacrev(lambda p:model(p,x[:8]).ravel())(params)
                flat=np.concatenate([np.asarray(a).reshape(32,-1) for a in jax.tree.leaves(jac)],1)
                eig=np.linalg.eigvalsh(flat@flat.T)
                threshold=max(eig.max()*1e-10,1e-14)
                positive=eig[eig>threshold]
                ntks.append({"seed":seed,"depth":depth,"variant":variant,"eigenvalues":eig.tolist(),
                    "rank":len(positive),"dimension":32,"threshold":float(threshold),
                    "condition_positive_subspace":float(positive.max()/positive.min()) if len(positive) else None})
                print(f"revision convergence seed={seed} depth={depth} {variant}",flush=True)
    return {"records":rows,"ntk":ntks,"scope":{"steps":steps,"seeds":list(seeds),"depths":[2,4],"width":32,
        "patches":4,"classes":4,"train_examples":64,"heldout_examples":64,"lr_grid":lrs.tolist(),"optimizer":"SGD momentum 0.9"},
        "mslr_definition":"Largest tested LR with finite decreasing loss, maximum <= initial+max(1,|initial|). Right-censored at 0.01.",
        "limitation":"Different proper objectives have incomparable raw losses. Report held-out accuracy and each objective's convergence trace; no full-size training or larger MSLR claim."}


def purity():
    rng=np.random.default_rng(42)
    params={"embed":jnp.asarray(rng.normal(size=(8,32)),jnp.float32),"posemb":jnp.zeros((4,32),jnp.float32),
        "blocks":jax.tree.map(lambda a:a.astype(jnp.float32),add_frequencies(init_blocks(rng,2,32))),
        "head":jnp.asarray(rng.normal(size=(32,4)),jnp.float32)}
    x=jnp.asarray(rng.normal(size=(2,4,8)),jnp.float32)
    labels=jnp.array([0,1])
    def step(p):
        value,g=jax.value_and_grad(lambda pp:objective(logits(pp,x,SELECTED),labels,SELECTED))(p)
        return value,jax.tree.map(lambda a,g:a-0.001*g,p,g)
    records={}
    funcs={"selected_complete_step":(step,params),
           "rational_gate_value_and_gradient":(jax.value_and_grad(lambda a:jnp.sum(rev.rational_gate(a))),x)}
    for name,(fn,arg) in funcs.items():
        counts=scan(jax.jit(fn).lower(arg).compiler_ir("hlo").as_hlo_text())
        forbidden={k:v for k,v in counts.items() if k in FORBIDDEN}
        if forbidden:raise AssertionError(forbidden)
        records[name]={"instruction_counts":counts,"forbidden":forbidden}
    return {"records":records,"scope":"CPU HLO, two-layer synthetic objective/gradient/SGD update including all learned frequencies. No production registration, TPU kernel or multi-host smoke test."}


def dependencies():
    names=["revision_study.py","revisions.py","references.py","convergence.py","propagation.py","simulations.py","purity.py"]
    return [ROOT/"analysis/phase1"/name for name in names]+sorted((ROOT/"algebraic_vision").glob("*.py"))


def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections",default=",".join(SECTIONS))
    parser.add_argument("--out",type=Path,default=ROOT/"results/phase1")
    parser.add_argument("--seeds",nargs="+",type=int,default=list(range(42,50)))
    parser.add_argument("--prototype-seeds",nargs="+",type=int,default=[42,43,44])
    parser.add_argument("--steps",type=int,default=200)
    args=parser.parse_args()
    names=args.sections.split(",")
    if any(n not in SECTIONS for n in names):parser.error(str(SECTIONS))
    funcs={"symbolic":symbolic,"attention":lambda:attention_audit(args.seeds),"positions":positions,
        "pool_precision":pool_precision,"propagation":lambda:propagation(args.prototype_seeds),
        "convergence":lambda:convergence(args.prototype_seeds,args.steps),"purity":purity}
    args.out.mkdir(parents=True,exist_ok=True)
    for name in names:
        start=time.monotonic()
        before=hashes()
        data=funcs[name]()
        if hashes()!=before:raise RuntimeError("Sources changed during section; rerun")
        data["provenance"]={"source_sha256":before,"elapsed_seconds":time.monotonic()-start,"jax":jax.__version__,
            "python":platform.python_version(),"x64":jax.config.x64_enabled,"backend":jax.default_backend(),
            "seeds":args.seeds if name=="attention" else args.prototype_seeds if name in {"propagation","convergence"} else [],
            "dependencies":{n:importlib.metadata.version(n) for n in ["numpy","sympy","mpmath"]}}
        path=args.out/f"revision_{name}.json"
        path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
        print(f"DONE revision {name}: {time.monotonic()-start:.1f}s",flush=True)
        jax.clear_caches()
    return 0


if __name__=="__main__":raise SystemExit(main())
