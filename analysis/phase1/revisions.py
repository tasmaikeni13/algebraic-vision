"""Further Phase 1 equations; no production architecture registration.

Derivatives use ordinary AD or exact analytic differentials of the stated
rational functions. No surrogate gradients are used. Keeping the families
separate preserves the previous experiment definitions and evidence.
"""

from functools import partial

import jax
import jax.numpy as jnp

from . import references as ref


@partial(jax.custom_jvp,nondiff_argnums=(2,))
def cayley_rows(u,kappa,order):
    """Normalized rational kernel with its exact, conservative differential."""
    base=(kappa+u)/(kappa-u)
    if order>16:
        base=base/jnp.max(base,-1,keepdims=True)
    weights=ref.square_power(base,order.bit_length()-1)
    return weights/jnp.sum(weights,-1,keepdims=True)


@cayley_rows.defjvp
def _cayley_jvp(order,primals,tangents):
    u,kappa=primals
    du,dk=tangents
    p=cayley_rows(u,kappa,order)
    # Differentiating an unnecessary common row scale creates a large graph.
    # Its exact derivative cancels. This is the full differential in BOTH
    # scores and margins, not a modified backward rule.
    dlog=(2*order)*(kappa*du-u*dk)/(kappa*kappa-u*u)
    return p,p*(dlog-jnp.sum(p*dlog,-1,keepdims=True))


def projected(q, k, kappa=1.1, gain_q=0.25, gain_k=None, order=16,
              column_squarings=None, mxu_round=False):
    """Higher-order Cayley kernel, optionally retaining key radial energy.

    A column factor (1+||k||²/d)^(2^r) compensates some information lost by
    the key ball lift. It is a content prior, not column balancing. Common
    row/column scales cancel over the reals and prevent large-order overflow.
    """
    if order < 1 or order & (order-1):
        raise ValueError("order must be a positive power of two")
    gain_k=gain_q if gain_k is None else gain_k
    original_k=ref.work(k)
    qt,kt=ref.ball_lift(q,gain_q),ref.ball_lift(k,gain_k)
    if mxu_round:
        qt=qt.astype(jnp.bfloat16)
        kt=kt.astype(jnp.bfloat16)
    dtype=jnp.promote_types(jnp.result_type(q,k),jnp.float32)
    u=jnp.einsum("...id,...jd->...ij",qt,kt,preferred_element_type=dtype)
    weights=cayley_rows(u,kappa,order)
    if column_squarings is None:
        return weights
    energy=1+jnp.sum(original_k*original_k,-1)/k.shape[-1]
    energy=energy/jnp.max(energy,-1,keepdims=True)
    weights=weights*ref.square_power(energy,column_squarings)[...,None,:]
    return weights/jnp.sum(weights,-1,keepdims=True)


@partial(jax.custom_jvp,nondiff_argnums=(1,))
def projective_tables(w,grid):
    """Unrolled bounded-frequency projective recurrence, no square roots."""
    v,u=jnp.ones_like(w),jnp.zeros_like(w)
    cs,ss=[],[]
    for _ in range(grid):
        vv,uu=v*v,u*u
        den=vv+uu
        cs.append((vv-uu)/den)
        ss.append(2*u*v/den)
        v,u=v-w*u,u+w*v
    return jnp.stack(cs),jnp.stack(ss)


@projective_tables.defjvp
def _projective_jvp(grid,primals,tangents):
    (w,),(dw,)=primals,tangents
    c,s=projective_tables(w,grid)
    index=jnp.arange(grid,dtype=w.dtype).reshape((grid,)+(1,)*w.ndim)
    dtheta=2*index*dw/(1+w*w)
    return (c,s),(-s*dtheta,c*dtheta)


def pade_base(t,b=0.25,newton_steps=0):
    """Positive reciprocal base, Padé correction to the radical base.

    For a=|t|, R0=a+(1+a/2+b*a²)/(1+3a/2+(1+b)*a²+2b*a³).
    At b=1/4, R0 approximates sqrt(1+a²) through fourth order. Optional
    Newton steps are finite rational equations, not exact real square roots.
    """
    if newton_steps < 0:
        raise ValueError("newton_steps must be nonnegative")
    a=jnp.maximum(t,-t)
    aa=a*a
    root=a+(1+0.5*a+b*aa)/(1+1.5*a+(1+b)*aa+2*b*aa*a)
    for _ in range(newton_steps):
        root=0.5*(root+(1+t*t)/root)
    # R0'(0)=0, so AD through abs has the correct zero derivative here.
    return jnp.where(t>=0,t+root,1/(a+root))


@partial(jax.custom_jvp,nondiff_argnums=(3,))
def pade_rows(scores,b=0.25,temperature=1.,newton_steps=0):
    base=pade_base(ref.work(scores)*temperature/8,b,newton_steps)
    base=base/jnp.max(base,-1,keepdims=True)
    weights=ref.square_power(base)
    return weights/jnp.sum(weights,-1,keepdims=True)


@pade_rows.defjvp
def _pade_jvp(newton_steps,primals,tangents):
    scores,b,temperature=primals
    ds,db,dt=tangents
    s=ref.work(scores)
    t=s*temperature/8
    a=jnp.maximum(t,-t)
    aa=a*a
    numerator=1+0.5*a+b*aa
    denominator=1+1.5*a+(1+b)*aa+2*b*aa*a
    inv=1/denominator
    root=a+numerator*inv
    root_a=1+(0.5+2*b*a)*inv-numerator*(1.5+2*(1+b)*a+6*b*aa)*inv*inv
    root_b=-aa*a*inv*inv
    for _ in range(newton_steps):
        inverse=1/root
        ratio=(1+aa)*inverse*inverse
        root_a=0.5*((1-ratio)*root_a+2*a*inverse)
        root_b=0.5*(1-ratio)*root_b
        root=0.5*(root+(1+aa)*inverse)
    f=a+root
    base=jnp.where(t>=0,t+root,1/f)
    base=base/jnp.max(base,-1,keepdims=True)
    weights=ref.square_power(base)
    p=weights/jnp.sum(weights,-1,keepdims=True)
    slope=(1+root_a)/f
    sign=jnp.where(t>=0,1.,-1.)
    dlog=slope*(temperature*ref.work(ds)+s*dt)+8*sign*root_b/f*db
    return p,p*(dlog-jnp.sum(p*dlog,-1,keepdims=True))


def _rational_gate_parts(x,c):
    y=c*ref.work(x)
    # The inactive branch is clamped away from its pole. The active branch
    # is exactly 1/(1+c*|x|), including its correct first two derivatives at 0.
    denominator=jnp.where(y>=0,1+jnp.maximum(y,-0.5),1-jnp.minimum(y,0.5))
    r=1/denominator
    half=0.5*ref.work(x)*r
    return jnp.where(y>=0,ref.work(x)-half,half),r,y


@partial(jax.custom_jvp,nondiff_argnums=(1,))
def rational_gate(x,c=0.851):
    """Root-free coupled softsign gate, C², derivative in (0,1).

    g(x)=x/2*(1+c*x/(1+c*|x|)), c>0. It matches rgelu's value, slope and
    curvature at zero for c=1.702/2. Its negative tail is a bounded constant.
    """
    x=jnp.asarray(x)
    return _rational_gate_parts(x,c)[0].astype(x.dtype)


@rational_gate.defjvp
def _rational_gate_jvp(c,primals,tangents):
    (x,),(dx,)=primals,tangents
    x=jnp.asarray(x)
    value,r,y=_rational_gate_parts(x,c)
    half=0.5*r*r
    derivative=jnp.where(y>=0,1-half,half)
    return value.astype(x.dtype),(derivative*ref.work(dx)).astype(x.dtype)


def _fixed_pade_parts(scores):
    """One rational Newton correction with fixed b=1/4, temperature=1."""
    t=ref.work(scores)/8
    a=jnp.maximum(t,-t)
    a2=a*a
    denominator=1+a*(1.5+a*(1.25+0.5*a))
    inverse=1/denominator
    numerator=1+a*(0.5+0.25*a)
    r0=a+numerator*inverse
    ir=1/r0
    root=0.5*(r0+(1+a2)*ir)
    f=a+root
    base=jnp.where(t>=0,t+root,1/f)
    base=base/jnp.max(base,-1,keepdims=True)
    weight=ref.square_power(base)
    p=weight/jnp.sum(weight,-1,keepdims=True)
    return p,a,a2,inverse,numerator,r0,ir,f


@jax.custom_jvp
def fixed_pade_rows(scores):
    """Fixed, positive rational approximation; exact conservative derivative."""
    return _fixed_pade_parts(scores)[0]


@fixed_pade_rows.defjvp
def _fixed_pade_jvp(primals,tangents):
    (scores,),(ds,)=primals,tangents
    p,a,a2,inv,num,r0,ir,f=_fixed_pade_parts(scores)
    r0a=1+(0.5+0.5*a)*inv-num*(1.5+a*(2.5+1.5*a))*inv*inv
    ra=0.5*((1-(1+a2)*ir*ir)*r0a+2*a*ir)
    slope=(1+ra)/f
    dlog=slope*ref.work(ds)
    return p,p*(dlog-jnp.sum(p*dlog,-1,keepdims=True))


@jax.custom_jvp
def tail_rows(scores,denominator=2.,temperature=1.):
    """Generalized earlier rational tail; matches the origin slope for a>0."""
    t=ref.work(scores)*temperature/8
    a=jnp.maximum(t,-t)
    correction=a*a/(denominator+a)
    base=jnp.where(t>=0,1+t+correction,1/(1+a+correction))
    base=base/jnp.max(base,-1,keepdims=True)
    weights=ref.square_power(base)
    return weights/jnp.sum(weights,-1,keepdims=True)


@tail_rows.defjvp
def _tail_jvp(primals,tangents):
    scores,denominator,temperature=primals
    ds,da,db=tangents
    s=ref.work(scores)
    p=tail_rows(scores,denominator,temperature)
    t=s*temperature/8
    a=jnp.maximum(t,-t)
    aa=a*a
    poly=2*aa+(denominator+1)*a+denominator
    common=(denominator+a)*poly
    slope=(2*aa+4*denominator*a+denominator*denominator)/common
    sign=jnp.where(t>=0,1.,-1.)
    dlog=slope*(temperature*ref.work(ds)+s*db)-8*sign*aa/common*da
    return p,p*(dlog-jnp.sum(p*dlog,-1,keepdims=True))


def scores(q,k):
    dtype=jnp.promote_types(jnp.result_type(q,k),jnp.float32)
    return jnp.einsum("...id,...jd->...ij",q,k,preferred_element_type=dtype)*q.shape[-1]**-0.5


def frequency(raw,grid,axis=0):
    """Strictly positive, bounded learned frequency, no scalar rsqrt.

    Initialize raw away from zero. The two planes share each axis frequency;
    determinant nondegeneracy is maintained throughout finite parameter updates.
    """
    # Each phase over an offset has magnitude <4<2*pi for either plane,
    # which is sufficient to exclude all nonzero wraps on a G by G grid.
    cap=1/grid
    return cap*(0.25+0.75/(1+raw*raw))


def learned_bicayley(q,k,raw_x,raw_y,grid):
    """Per-head version used to count all position-parameter gradients."""
    wx,wy=frequency(raw_x,grid),frequency(raw_y,grid,1)
    # Explicit unrolling makes the static operation budget include every grid
    # recurrence step instead of a cost estimator's unknown loop trip count.
    cx,sx=projective_tables(wx,grid)
    cy,sy=projective_tables(wy,grid)
    row,col=jnp.arange(grid*grid)//grid,jnp.arange(grid*grid)%grid
    cp,sp=cx[col]*cy[row]-sx[col]*sy[row],sx[col]*cy[row]+cx[col]*sy[row]
    cm,sm=cx[col]*cy[row]+sx[col]*sy[row],sx[col]*cy[row]-cx[col]*sy[row]
    cp,sp,cm,sm=[a.transpose(1,0,2) for a in [cp,sp,cm,sm]]
    extra=q.shape[-2]-grid*grid
    if extra:
        ones=jnp.ones((raw_x.shape[0],extra,raw_x.shape[1]),q.dtype)
        zeros=jnp.zeros_like(ones)
        cp,sp=jnp.concatenate([cp,ones],1),jnp.concatenate([sp,zeros],1)
        cm,sm=jnp.concatenate([cm,ones],1),jnp.concatenate([sm,zeros],1)
    def rotate(x):
        a,b,c,d=jnp.split(ref.work(x),4,-1)
        return jnp.concatenate([cp*a-sp*b,sp*a+cp*b,cm*c-sm*d,sm*c+cm*d],-1).astype(x.dtype)
    return rotate(q),rotate(k)
