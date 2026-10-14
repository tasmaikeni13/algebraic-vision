"""Research regression tests: exact identities, actual derivatives, counterexamples."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic_vision.attention import radical_weights
from algebraic_vision.primitives import cayley_tables, radical_exp
from analysis.phase1 import references as ref
from analysis.phase1.purity import scan


def test_rational_identity_and_frozen_radical_reference():
    u=jnp.array([-0.999,-0.5,0.,0.3,0.999])
    k=1.05
    scores=16*k*u/(k*k-u*u)
    w=ref.square_power((k+u)/(k-u))
    np.testing.assert_allclose(radical_exp(scores),w,rtol=5e-6,atol=1e-30)
    s=jnp.arange(21,dtype=jnp.float32).reshape(3,7)-10
    np.testing.assert_allclose(ref.radical_rows(s),radical_weights(s,jnp.zeros((3,1)),8,1.),rtol=3e-6)


@pytest.mark.parametrize("scale",[1.,10.,100.])
def test_baseline_reference_vjp_matches_frozen_implementation(scale):
    s=jnp.linspace(-2,2,21,dtype=jnp.float32).reshape(3,7)*scale
    cot=jnp.linspace(-1,1,21,dtype=jnp.float32).reshape(3,7)
    grad1=jax.grad(lambda a:jnp.sum(ref.radical_rows(a)*cot))(s)
    grad2=jax.grad(lambda a:jnp.sum(radical_weights(a,jnp.zeros((3,1)),8,1.)*cot))(s)
    np.testing.assert_allclose(grad1,grad2,rtol=3e-4,atol=5e-8)


def test_margin_has_correct_domain():
    t=jnp.array([-1e5,-10.,-1.,0.,1.,10.,1e5])
    value=ref.constrained_margin(t)
    assert bool(jnp.all((value>=1.05)&(value<=2.)))
    assert float(ref.constrained_margin(0.))==2.
    # The prompt's original parameterization violates the advertised range.
    assert float(1+1/ref.rho(-10.))>2
    assert float(1+1/ref.rho(100.))<1.05


def test_projective_matches_existing_power_and_has_frequency_derivative():
    w=jnp.array([0.02,0.17,0.5],jnp.float32)
    c,s=ref.projective_tables(w,64)
    c1,s1=cayley_tables(64,w)
    np.testing.assert_allclose(c,c1,atol=3e-6)
    np.testing.assert_allclose(s,s1,atol=3e-6)
    np.testing.assert_allclose(c*c+s*s,1,atol=4e-7)
    g=jax.grad(lambda a:ref.projective_tables(a,64)[1][-1].sum())(w)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.linalg.norm(g))>1


def test_bicayley_relative_offsets_and_norms():
    grid=4
    basis=jnp.tile(jnp.eye(4)[None,:,:],(grid*grid,1,1)).transpose(1,0,2)
    matrices=jax.vmap(lambda a:ref.bicayley(a,grid))(basis).transpose(1,2,0)
    for r1,c1,r2,c2 in [(0,1,1,2),(1,2,2,3),(0,0,3,1)]:
        relative=matrices[r1*grid+c1].T@matrices[r2*grid+c2]
        expected=matrices[(r2-r1)*grid+(c2-c1)]
        np.testing.assert_allclose(relative,expected,atol=1e-6)
    for m in matrices:
        np.testing.assert_allclose(m.T@m,jnp.eye(4),atol=1e-6)
        assert float(jnp.linalg.det(m))==pytest.approx(1,abs=1e-6)
    assert min(float(jnp.linalg.norm(matrices[i]-matrices[j])) for i in range(16) for j in range(i))>0.05


@pytest.mark.parametrize("n",[1,3,197])
def test_pool_fixed_points_and_stable_denominator(n):
    h=jnp.array([1.,-2.,3.,-4.],jnp.float64)
    batch=jnp.tile(h,(n,1))
    np.testing.assert_allclose(ref.lorentz_pool(batch),h,rtol=2e-14)
    np.testing.assert_allclose(ref.robust_pool(batch),h,rtol=2e-14)
    rng=np.random.default_rng(5)
    x=jnp.asarray(rng.normal(size=(n,4)))
    np.testing.assert_allclose(ref.lorentz_pool(x,True),ref.lorentz_pool(x,False),rtol=1e-12,atol=1e-12)


def test_stable_pool_and_vjp_near_light_cone():
    h=jnp.tile(jnp.array([1e6,-1e6,1e6,-1e6],jnp.float32),(197,1))
    stable=ref.lorentz_pool(h)
    assert bool(jnp.all(jnp.isfinite(stable)))
    np.testing.assert_allclose(stable,h[0],rtol=3e-6)
    g=jax.grad(lambda x:jnp.sum(ref.lorentz_pool(x)))(h)
    assert bool(jnp.all(jnp.isfinite(g)))
    np.testing.assert_allclose(g,jnp.ones_like(g)/197,rtol=1e-4,atol=1e-6)


def test_outlier_derivative_needs_more_than_one_token():
    h=jnp.array([[10.,-10.,10.,-10.]])
    jac=jax.jacfwd(ref.lorentz_pool)(h).reshape(4,4)
    np.testing.assert_allclose(jac,jnp.eye(4),atol=1e-10)
    rng=np.random.default_rng(8)
    clean=jnp.asarray(rng.normal(size=(8,4)))
    def derivative(scale):
        x=clean.at[0].set(jnp.array([scale,0.,0.,0.]))
        return float(jnp.linalg.norm(jax.jacfwd(lambda t:ref.lorentz_pool(x.at[0].set(t)))(x[0])))
    assert derivative(100)<0.15*derivative(10)


def test_pool_exact_strict_lower_bound():
    x=jnp.asarray(np.random.default_rng(9).normal(size=(19,16)))*30
    b=ref.ball_lift(x)
    denominator=19**2-jnp.sum(jnp.sum(b,0)**2)
    lower=19**2*16/(16+jnp.max(jnp.sum(x*x,-1)))
    assert float(denominator)>=float(lower)>0


def test_mle_gradient_is_conservative_quarter_is_not():
    z=jnp.array([-1.3,0.4,2.1])
    y=jnp.array([0.8,0.1,0.1])
    ce=lambda a:-jnp.sum(y*jnp.log(ref.head_probabilities(a)))
    np.testing.assert_allclose(jax.grad(ce)(z),ref.mle_gradient(z,y),atol=1e-14)
    exact=jax.jacfwd(lambda a:ref.mle_gradient(a,y))(z)
    quarter=jax.jacfwd(lambda a:ref.mle_gradient(a,y,exponent=0.25))(z)
    np.testing.assert_allclose(exact,exact.T,atol=1e-14)
    assert float(jnp.max(jnp.abs(quarter-quarter.T)))>1e-3


@pytest.mark.parametrize("kind",["power","brier"])
def test_algebraic_losses_minimized_at_target_with_true_derivatives(kind):
    y=jnp.array([0.2,0.3,0.5])
    base=y
    for _ in range(3):
        base=jnp.sqrt(base)
    z=2*(base-1/base)  # tau=4, inverse rho(z/tau).
    np.testing.assert_allclose(ref.head_probabilities(z),y,atol=1e-14)
    np.testing.assert_allclose(jax.grad(lambda a:ref.head_loss(a,y,kind))(z),0,atol=1e-13)
    for offset in [jnp.array([1.,0.,0.]),jnp.array([0.,-2.,0.])]:
        assert float(ref.head_loss(z+offset,y,kind))>float(ref.head_loss(z,y,kind))
    direction=jnp.array([0.3,-0.2,0.1])
    f=lambda a:ref.head_loss(a,y,kind)
    point=z+jnp.array([0.4,0.2,-0.1])
    finite_diff=(f(point+1e-5*direction)-f(point-1e-5*direction))/2e-5
    np.testing.assert_allclose(finite_diff,jax.grad(f)(point).dot(direction),atol=2e-9)


def test_hlo_scanner_finds_negative_control():
    text=jax.jit(lambda x:jnp.exp(x)+jnp.log(x)).lower(jnp.ones((3,))).compiler_ir("hlo").as_hlo_text()
    counts=scan(text)
    assert counts.get("exponential",0)>0
    assert counts.get("log",0)>0
    with pytest.raises(AssertionError):
        scan("this is not HLO")


def test_missing_phase_evidence_cannot_pass_gate(tmp_path):
    from analysis.phase1.report import build
    decision=build(tmp_path)
    assert decision["decision"]=="NO-GO"
    assert "convergence" in decision["missing_evidence"]


def test_stale_source_rejected_by_evidence_audit():
    from analysis.phase1.report import evidence_integrity
    assert evidence_integrity({"symbolic":{"provenance":{"source_sha256":{}}}})


def test_rational_tail_has_the_actual_claimed_derivative():
    s=jnp.array([-100.,-3.,0.,2.,100.])
    t=jnp.abs(s/8)
    slope=(2*t*t+8*t+4)/((t+2)*(2*t*t+3*t+2))
    p=ref.rational_tail_rows(s)
    expected=(jnp.diag(p)-p[:,None]*p[None,:])*slope[None,:]
    np.testing.assert_allclose(jax.jacfwd(ref.rational_tail_rows)(s),expected,atol=1e-13)
