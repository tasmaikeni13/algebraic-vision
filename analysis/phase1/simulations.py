"""Paired Monte Carlo studies and Figure 5 stress diagnostics on CPU."""

import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np

from . import references as ref

VARIANTS = {
    "standard": ("standard", 2.0, 1.0),
    "algebraic_v1": ("v1", 2.0, 1.0),
    "v2_proposed": ("v2", 2.0, 1.0),
    "v2_sharp": ("v2", 1.1, 2.0),
    "v2_balanced": ("v2_balanced", 1.1, 2.0),
    "v2_rational_tail": ("v2_tail", 2.0, 1.0),
}


def row_attention(q, k, variant):
    return ref.attention(q, k, *VARIANTS[variant])


def summary(values):
    a = np.asarray(values, dtype=float)
    return {"mean":float(a.mean()), "std":float(a.std(ddof=1)) if len(a)>1 else 0.0,
            "se":float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else 0.0,
            "min":float(a.min()), "max":float(a.max()), "samples":len(a)}


def stress(seeds):
    alphas = [1, 2, 4, 8, 10, 16, 32, 64, 100, 128, 256]
    records = []
    n, d = 197, 64
    for distribution in ("gaussian", "student_t3"):
        for seed in seeds:
            rng = np.random.default_rng(seed)
            draw = (lambda shape: rng.normal(size=shape)) if distribution == "gaussian" else (lambda shape: rng.standard_t(3, size=shape)/math.sqrt(3))
            q0, k0 = draw((n,d)), draw((n,d))
            cot = jnp.asarray(rng.normal(size=(n,n)))
            for name, (arm, kappa, gain) in VARIANTS.items():
                # Score units differ between kernels; Q derivatives are the
                # comparable diagnostic. Column balancing couples query rows.
                fn = jax.jit(jax.value_and_grad(lambda q,k: jnp.sum(ref.attention(q,k,arm,kappa,gain)*cot), argnums=0))
                for alpha in alphas:
                    q, k = jnp.asarray(alpha*q0), jnp.asarray(alpha*k0)
                    p = ref.attention(q,k,arm,kappa,gain)
                    _, dq = fn(q,k)
                    if arm.startswith("v2") and arm != "v2_tail":
                        qg, kg = q*gain, k*gain
                        r = jnp.sqrt(d+jnp.sum(qg*qg,-1,keepdims=True))
                        kt = ref.ball_lift(kg)
                        dot = qg@kt.T
                        u = dot/r
                        slope = 16*kappa/(kappa*kappa-u*u)
                        b = gain*slope[...,None]*(kt[None,:,:]/r[:,None,:]-qg[:,None,:]*dot[...,None]/r[:,None,:]**3)
                    else:
                        scores = q@k.T/math.sqrt(d)
                        if arm == "standard":
                            slope = jnp.ones_like(scores)
                        elif arm == "v2_tail":
                            t = jnp.abs(scores/8)
                            slope = (2*t*t+8*t+4)/((t+2)*(2*t*t+3*t+2))
                        else:
                            slope = jax.lax.rsqrt(1+(scores/8)**2)
                        b = slope[...,None]*k[None,:,:]/math.sqrt(d)
                    # Exact row Jacobian for row-local variants. Balanced Q
                    # Jacobians are sampled with VJPs below instead of claiming
                    # the row-local formula covers their cross-row coupling.
                    jpq = p[...,None]*(b-jnp.sum(p[...,None]*b,1,keepdims=True))
                    fro_q = jnp.sqrt(jnp.sum(jpq*jpq, (1,2)))
                    pp = jnp.sum(p*p,-1,keepdims=True)
                    fro_score = jnp.sqrt(jnp.sum(slope*slope*p*p*((1-p)**2+jnp.maximum(pp-p*p,0)), -1))
                    row = {"distribution":distribution,"seed":seed,"alpha":alpha,"variant":name,
                           "entropy":float(jnp.mean(-jnp.sum(p*jnp.log(jnp.maximum(p,1e-300)),-1))/math.log(n)),
                           "q_gradient_norm":float(jnp.linalg.norm(dq)),
                           "median_abs_score":float(jnp.median(jnp.abs(q@k.T/math.sqrt(d)))),
                           "score_jacobian_fro":float(jnp.mean(fro_score)) if arm!="v2_balanced" else None,
                           "score_saturated_rows":float(jnp.mean(fro_score<1e-4)) if arm!="v2_balanced" else None,
                           "q_jacobian_fro":float(jnp.mean(fro_q)) if arm!="v2_balanced" else None,
                           "q_saturated_rows":float(jnp.mean(fro_q<1e-4)) if arm!="v2_balanced" else None}
                    records.append(row)
                print(f"stress {distribution} seed={seed} {name}", flush=True)
    # Exact radial counterexample: two collinear keys produce nonzero relative
    # scores but all ball-lift radial derivatives decay as alpha^-3.
    radial = []
    for alpha in [10,32,100,256,1024]:
        q = jnp.array([[float(alpha),0,0,0]])
        k = jnp.array([[1.,0,0,0],[2.,0,0,0]])
        row = {"alpha":alpha}
        for name in ["algebraic_v1","v2_proposed","v2_sharp","v2_rational_tail"]:
            g = jax.grad(lambda a: row_attention(a,k,name)[0,0])(q)
            row[name] = float(jnp.abs(g[0,0]))
        radial.append(row)
    return {"records":records,"radial_counterexample":radial,
            "jacobian_units":"score Jacobian uses s for standard/v1/tail, u for projected v2; Q Jacobian uses the same pre-projection Q in all arms",
            "balanced_coupling":"Q VJP includes all query rows; row-local Jacobian statistics intentionally null"}


def crowding(seeds):
    records = []
    for n in [197,256]:
        for foreground in [1,3,5]:
            for scenario in ["mixed_queries","all_signal_queries"]:
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    fg, bg = rng.normal(size=(2,64))
                    fg = fg/np.linalg.norm(fg)*8
                    bg = bg-bg.dot(fg)/fg.dot(fg)*fg
                    bg = bg/np.linalg.norm(bg)*8
                    q = np.repeat(bg[None,:],n,0) if scenario=="mixed_queries" else np.repeat(fg[None,:],n,0)
                    q[:max(1,n//8)] = fg
                    q = q+0.25*rng.normal(size=(n,64))
                    k = np.repeat(bg[None,:],n,0)+0.25*rng.normal(size=(n,64))
                    k[:foreground] = fg+0.25*rng.normal(size=(foreground,64))
                    q,k = jnp.asarray(q),jnp.asarray(k)
                    for name in VARIANTS:
                        p = row_attention(q,k,name)
                        fn = lambda keys: jnp.sum(row_attention(q,keys,name)[0,:foreground])
                        dk = jax.grad(fn)(k)
                        signal = float(jnp.linalg.norm(jnp.sum(dk[:foreground],0)))
                        noise = float(jnp.sqrt(jnp.sum(dk[foreground:]*dk[foreground:])))
                        records.append({"n":n,"foreground":foreground,"scenario":scenario,"seed":seed,"variant":name,
                                        "foreground_mass":float(jnp.sum(p[0,:foreground])),
                                        "gradient_snr":signal/max(noise,1e-30),
                                        "foreground_gradient_norm":signal,"background_gradient_norm":noise})
    return {"records":records,"snr_definition":"||sum foreground d(mass)/dK|| / sqrt(sum background ||d(mass)/dK||^2); same foreground-mass objective across arms"}


def layer_norm(x):
    x = x-jnp.mean(x,-1,keepdims=True)
    return x*jax.lax.rsqrt(jnp.mean(x*x,-1,keepdims=True)+1e-6)


def angle(a,b):
    a,b = np.asarray(a),np.asarray(b)
    return float(np.degrees(np.arccos(np.clip(a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b)), -1,1))))


def outliers(seeds):
    records=[]
    pools={"raw_gap_diagnostic":lambda h:jnp.mean(h,0),
           "standard_v1_actual_ln_gap":lambda h:jnp.mean(layer_norm(h),0),
           "v2_lorentz":lambda h:layer_norm(ref.lorentz_pool(h)),
           "v2_inverse_energy":lambda h:layer_norm(ref.robust_pool(h))}
    for n,d in [(197,768),(256,1024)]:
        for seed in seeds:
            rng=np.random.default_rng(seed)
            signal=rng.normal(size=d)
            signal-=signal.mean()
            signal=signal/np.linalg.norm(signal)*math.sqrt(d)
            clean=0.4*signal+0.8*rng.normal(size=(n,d))
            direction=rng.normal(size=d)
            direction=direction/np.linalg.norm(direction)
            for count in [1,3,5]:
                for scale in [5,10,25,50,100]:
                    bad=clean.copy()
                    bad[:count]=scale*math.sqrt(d)*direction
                    for name,fn in pools.items():
                        records.append({"n":n,"d":d,"seed":seed,"outliers":count,"scale_sqrt_d":scale,"pool":name,
                                        "angle_degrees":angle(fn(jnp.asarray(clean)),fn(jnp.asarray(bad)))})
    return {"records":records,"baseline_note":"Frozen model applies per-token final LayerNorm before GAP. Raw GAP is a separate diagnostic and is not the baseline."}


def precision(seeds):
    records=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        q0,k0=rng.normal(size=(32,64)),rng.normal(size=(32,64))
        cot=jnp.asarray(rng.normal(size=(32,32)))
        for scale in [1,3,10,30,100]:
            q,k=jnp.asarray(q0*scale),jnp.asarray(k0*scale)
            for name in VARIANTS:
                fn=lambda a,b:row_attention(a,b,name)
                p64,pull64=jax.vjp(fn,q,k)
                g64=pull64(cot)[0]
                for dtype in [jnp.float32,jnp.bfloat16]:
                    pp,pull=jax.vjp(fn,q.astype(dtype),k.astype(dtype))
                    gg=pull(cot.astype(pp.dtype))[0]
                    records.append({"seed":seed,"scale":scale,"variant":name,"dtype":str(jnp.dtype(dtype)),
                                    "tv_max":float(jnp.max(0.5*jnp.sum(jnp.abs(pp-p64),-1))),
                                    "backward_relative_error":float(jnp.linalg.norm(gg.astype(jnp.float64)-g64)/jnp.maximum(jnp.linalg.norm(g64),1e-30)),
                                    "finite":bool(jnp.all(jnp.isfinite(pp))&jnp.all(jnp.isfinite(gg)))})
    boundary=[]
    with mp.workdps(50):
        for kap in ["2","1.1","1.0500001"]:
            uu=[mp.mpf(x) for x in ["-0.9999999","-0.9","0","0.9","0.9999999"]]
            kk=mp.mpf(kap)
            ww=[((kk+a)/(kk-a))**8 for a in uu]
            p_ref=np.array([float(a/sum(ww)) for a in ww])
            dd=[p_ref[i]*(1-p_ref[i])*float(16*kk/(kk*kk-uu[i]**2)) for i in range(len(uu))]
            for dtype in [jnp.float32,jnp.bfloat16]:
                # Native low-precision u/kappa is a deliberately unsafe
                # ablation; mixed mode keeps the rational loop in float32.
                for mode in ["native","fp32_accumulation"]:
                    u=jnp.array([[float(a) for a in uu]],dtype)
                    kp=jnp.array(float(kk),dtype)
                    if mode=="fp32_accumulation":
                        u,kp=u.astype(jnp.float32),jnp.float32(float(kk))
                    pp=ref.rational_rows(u,kp)
                    jac=jax.jacrev(lambda x:ref.rational_rows(x,kp))(u)[0,:,0,:]
                    diag=np.asarray(jnp.diag(jac),dtype=float)
                    boundary.append({"kappa":kap,"dtype":str(jnp.dtype(dtype)),"mode":mode,
                                     "tv":float(0.5*np.sum(np.abs(np.asarray(pp[0],dtype=float)-p_ref))),
                                     "backward_diagonal_relative_error":float(np.linalg.norm(diag-dd)/max(np.linalg.norm(dd),1e-30)),
                                     "finite":bool(jnp.all(jnp.isfinite(pp))&jnp.all(jnp.isfinite(jac)))})
    pool=[]
    for scale in [1,10,100,1e3,1e4,1e6]:
        # Every token identical: exact fixed point gives a high-precision
        # reference even where the naive denominator rounds to zero.
        h0=np.tile(np.array([1.,-1.,1.,-1.])*scale,(197,1))
        for dtype in [jnp.float32,jnp.bfloat16]:
            for stable in [False,True]:
                h=jnp.asarray(h0,dtype)
                p=ref.lorentz_pool(h,stable)
                ref_value=jnp.asarray(h0[0])
                # VJP compared to float64 evaluation at the SAME input,
                # including the derivative, not just a fixed-point value.
                cot=jnp.ones_like(p)
                g=jax.vjp(lambda a:ref.lorentz_pool(a,stable),h)[1](cot)[0]
                gref=jax.vjp(lambda a:ref.lorentz_pool(a,True),jnp.asarray(h0))[1](cot.astype(jnp.float64))[0]
                finite=bool(jnp.all(jnp.isfinite(p))&jnp.all(jnp.isfinite(g)))
                pool.append({"scale":scale,"dtype":str(jnp.dtype(dtype)),"stable":stable,"finite":finite,
                             "relative_value_error":float(jnp.linalg.norm(p-ref_value)/jnp.linalg.norm(ref_value)) if finite else None,
                             "relative_gradient_error":float(jnp.linalg.norm(g.astype(jnp.float64)-gref)/jnp.linalg.norm(gref)) if finite else None})
    return {"records":records,"kernel_boundary_50_digit":boundary,"pool_boundary":pool,
            "dtype_policy":"BF16 inputs/VJP outputs, FP32 reductions and nonlinearities; native BF16 rational loop is tested separately and rejected"}


def positions():
    records=[]
    for grid in [14,16,32,64]:
        # Check every distinct offset, and all pairs of positions through
        # the group law, rather than sampling grid cells only.
        w=np.array([1/(4*grid),1/(5*grid)])
        theta=2*np.arctan(w)
        distances=[]
        for dr in range(-(grid-1),grid):
            for dc in range(-(grid-1),grid):
                if dr==dc==0:
                    continue
                phases=np.array([dc*theta[0]+dr*theta[1],dc*theta[0]-dr*theta[1]])
                distances.append(float(np.sqrt(np.sum(4-4*np.cos(phases)))))
        for dtype in [jnp.float32,jnp.bfloat16]:
            c,s=ref.projective_tables(jnp.asarray(w,dtype),grid)
            unit_error=float(jnp.max(jnp.abs(c*c+s*s-1)))
            cp,sp=ref.projective_tables(jnp.asarray(w,jnp.float64),2*grid)
            group_error=0.
            for a in range(grid):
                # All nonnegative powers whose sum is within the table.
                group_error=max(group_error,float(jnp.max(jnp.abs(c[a]*c-s[a]*s-cp[a:a+grid]))),
                                float(jnp.max(jnp.abs(s[a]*c+c[a]*s-sp[a:a+grid]))))
            records.append({"grid":grid,"dtype":str(jnp.dtype(dtype)),"orthogonality_error":unit_error,
                            "power_addition_error":group_error,"min_offset_distance":min(distances),
                            "rank_determinant":float(-2*theta[0]*theta[1])})
    return {"records":records,"zero_frequency_counterexample":{"determinant":0,"all_offsets_identity":True},
            "v1_plane_null_direction":"(dr,dc)=(atan(wx),-atan(wy)); individual planes have rank one. Multiple independent v1 planes can jointly have rank two.",
            "note":"SO(4) by itself does not ensure global injectivity. These tied positive grid-dependent frequencies prevent angle wrapping."}
