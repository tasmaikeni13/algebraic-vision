"""Phase 1 selection probe; temporary, held-out corpus is not used for tuning."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_ENABLE_X64','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import itertools,json,math,time
import jax
import jax.numpy as jnp
import numpy as np
from analysis.phase1 import references as ref

parameters=np.array(list(itertools.product([0.125,0.25,0.5,1.,2.,4.],[0.5,0.75,1.,1.25])))
rows=[]

def prob(q,k,param,m):
    b,temperature=param
    s=(ref.work(q)@ref.work(k).T)*q.shape[-1]**-0.5
    t=s*temperature/8
    a=jnp.maximum(t,-t)
    if m==3:
        b=b+1
        positive=1+t+a*a/(b+a)
        negative=1/(1+a+a*a/(b+a))
    else:
        aa=a*a
        root=a+(1+0.5*a+b*aa)/(1+1.5*a+(1+b)*aa+2*b*aa*a)
        for _ in range(m):
            root=0.5*(root+(1+t*t)/root)
        positive=t+root
        negative=1/(a+root)
    base=jnp.where(t>=0,positive,negative)
    base=base/jnp.max(base,-1,keepdims=True)
    w=ref.square_power(base,3)
    return w/jnp.sum(w,-1,keepdims=True)

for m in [0,1,2,3]:
    started=time.monotonic()
    outs=[{'family':m,'b':float(k),'temperature':float(g), 'gradient_ratios':[], 'radial_ratios':[], 'bf16_tv_ratios':[], 'bf16_backward_ratios':[], 'f32_tv_ratios':[]} for k,g in parameters]
    def evaluate(q,k,cot,param):
        p=prob(q,k,param,m)
        g=jax.grad(lambda a:jnp.sum(prob(a,k,param,m)*cot))(q)
        return p,g
    batched=jax.jit(jax.vmap(evaluate,in_axes=(None,None,None,0)))
    basefun=jax.jit(lambda q,k,cot:(ref.attention(q,k,'v1'),jax.grad(lambda a:jnp.sum(ref.attention(a,k,'v1')*cot))(q)))
    for seed in [0,1]:
        rng=np.random.default_rng(seed)
        for dist in ['gaussian','student_t3']:
            q0,k0=(rng.normal(size=(2,197,64)) if dist=='gaussian' else rng.standard_t(3,size=(2,197,64))/math.sqrt(3))
            cot=jnp.asarray(rng.normal(size=(197,197)))
            for alpha in [10,100]:
                q,k=jnp.asarray(q0*alpha),jnp.asarray(k0*alpha)
                _,gbase=basefun(q,k,cot)
                _,gg=batched(q,k,cot,jnp.asarray(parameters))
                ratio=np.asarray(jnp.linalg.norm(gg,axis=(1,2))/jnp.linalg.norm(gbase))
                for out,val in zip(outs,ratio):out['gradient_ratios'].append(float(val))
    for alpha in [10,32,100]:
        q,k=jnp.array([[float(alpha),0,0,0]]),jnp.array([[1.,0,0,0],[2.,0,0,0]])
        cot=jnp.array([[1.,0.]])
        _,gbase=basefun(q,k,cot)
        _,gg=batched(q,k,cot,jnp.asarray(parameters))
        ratio=np.asarray(jnp.abs(gg[:,0,0]/gbase[0,0]))
        for out,val in zip(outs,ratio):out['radial_ratios'].append(float(val))
    prec={}
    for seed in [0,1]:
        rng=np.random.default_rng(seed+1000)
        q0,k0=rng.normal(size=(2,32,64))
        cot=jnp.asarray(rng.normal(size=(32,32)))
        for scale in [1,3,10,30,100]:
            q,k=jnp.asarray(q0*scale),jnp.asarray(k0*scale)
            pbase,gbase=basefun(q,k,cot)
            pbasebf,gbasebf=basefun(q.astype(jnp.bfloat16),k.astype(jnp.bfloat16),cot.astype(jnp.float32))
            btv=float(jnp.max(0.5*jnp.sum(jnp.abs(pbasebf-pbase),-1)))
            bge=float(jnp.linalg.norm(gbasebf.astype(jnp.float64)-gbase)/jnp.linalg.norm(gbase))
            btv32=float(jnp.max(0.5*jnp.sum(jnp.abs(basefun(q.astype(jnp.float32),k.astype(jnp.float32),cot.astype(jnp.float32))[0]-pbase),-1)))
            pp,gg=batched(q,k,cot,jnp.asarray(parameters))
            ppbf,ggbf=batched(q.astype(jnp.bfloat16),k.astype(jnp.bfloat16),cot.astype(jnp.float32),jnp.asarray(parameters,jnp.float32))
            pp32,_=batched(q.astype(jnp.float32),k.astype(jnp.float32),cot.astype(jnp.float32),jnp.asarray(parameters,jnp.float32))
            tv=np.asarray(jnp.max(0.5*jnp.sum(jnp.abs(ppbf.astype(jnp.float64)-pp),-1),-1))
            ge=np.asarray(jnp.linalg.norm(ggbf.astype(jnp.float64)-gg,axis=(1,2))/jnp.linalg.norm(gg,axis=(1,2)))
            tv32=np.asarray(jnp.max(0.5*jnp.sum(jnp.abs(pp32.astype(jnp.float64)-pp),-1),-1))
            prec.setdefault(scale,[]).append((tv,ge,tv32,btv,bge,btv32))
    for scale,vals in prec.items():
        for i,out in enumerate(outs):
            out['bf16_tv_ratios'].append(float(np.mean([v[0][i] for v in vals])/np.mean([v[3] for v in vals])))
            out['bf16_backward_ratios'].append(float(np.mean([v[1][i] for v in vals])/np.mean([v[4] for v in vals])))
            out['f32_tv_ratios'].append(float(np.mean([v[2][i] for v in vals])/np.mean([v[5] for v in vals])))
    for out in outs:
        out['min_gradient']=min(out['gradient_ratios'])
        out['min_radial']=min(out['radial_ratios'])
        out['max_bf16']=max(out['bf16_tv_ratios']+out['bf16_backward_ratios'])
        out['violation']=max(1/out['min_gradient'],1/out['min_radial'],out['max_bf16'])
    rows+=outs
    print('FAMILY',m,'seconds',round(time.monotonic()-started,2),flush=True)
    for out in sorted(outs,key=lambda x:x['violation'])[:5]:
        print({key:out[key] for key in ['family','b','temperature','min_gradient','min_radial','max_bf16','violation']},flush=True)
    jax.clear_caches()
json.dump({'selection_seeds':[0,1], 'precision_seeds':[1000,1001], 'parameters':rows},open('/tmp/vision_rational_probe.json','w'),indent=2,allow_nan=False)
