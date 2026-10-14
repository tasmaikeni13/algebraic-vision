"""Real derivatives and operation-budget checks for Phase 1 corrections."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from analysis.phase1 import references as ref,revisions as rev
from analysis.phase1.operation_count import count,forward_backward


@pytest.mark.parametrize("order",[8,16,32])
def test_cayley_differential_includes_score_and_margin(order):
    u=jnp.array([[-0.9,-0.4,0.,0.3,0.8]])
    k=jnp.array([[1.1]])
    du=jnp.array([[0.2,-0.1,0.3,0.2,0.4]])
    dk=jnp.array([[0.07]])
    def raw(u,k):
        w=ref.square_power((k+u)/(k-u),order.bit_length()-1)
        return w/jnp.sum(w,-1,keepdims=True)
    y,dy=jax.jvp(lambda a,b:rev.cayley_rows(a,b,order),(u,k),(du,dk))
    expected,gradient=jax.jvp(raw,(u,k),(du,dk))
    np.testing.assert_allclose(y,expected,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(dy,gradient,rtol=1e-12,atol=1e-13)
    cot=jnp.array([[0.1,0.2,-0.3,0.4,-0.2]])
    params=jnp.concatenate([u.ravel(),k.ravel()])
    f=lambda x:jnp.sum(rev.cayley_rows(x[:-1].reshape(u.shape),x[-1:].reshape(k.shape),order)*cot)
    hessian=jax.jacfwd(jax.grad(f))(params)
    np.testing.assert_allclose(hessian,hessian.T,rtol=1e-10,atol=1e-11)


def test_tail_differential_includes_temperature_and_denominator():
    s=jnp.array([-4.,-0.3,0.,0.8,3.])
    a,b=jnp.array(2.5),jnp.array(0.75)
    ds=jnp.array([0.1,-0.2,0.3,0.4,-0.5])
    def raw(s,a,b):
        t=s*b/8
        u=jnp.maximum(t,-t)
        r=u*u/(a+u)
        base=jnp.where(t>=0,1+t+r,1/(1+u+r))
        w=ref.square_power(base)
        return w/jnp.sum(w)
    y,dy=jax.jvp(rev.tail_rows,(s,a,b),(ds,jnp.array(0.1),jnp.array(-0.2)))
    yy,ddy=jax.jvp(raw,(s,a,b),(ds,jnp.array(0.1),jnp.array(-0.2)))
    np.testing.assert_allclose(y,yy,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(dy,ddy,rtol=1e-12,atol=1e-13)


@pytest.mark.parametrize("grid",[14,16])
def test_projective_differential_matches_recurrence(grid):
    w=jnp.array([[0.02,0.03],[0.01,0.04]])
    dw=jnp.array([[0.1,-0.2],[0.3,0.4]])
    actual=jax.jvp(lambda a:rev.projective_tables(a,grid),(w,),(dw,))
    expected=jax.jvp(lambda a:ref.projective_tables(a,grid),(w,),(dw,))
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)


def test_learned_frequencies_stay_nonzero_and_have_a_gradient():
    raw=jnp.array([-1e4,-2.,-1.,0.,1.,2.,1e4])
    for axis in [0,1]:
        cap=1/16
        value=rev.frequency(raw,16,axis)
        assert bool(jnp.all((value>cap*0.25)&(value<=cap)))
        assert abs(float(jax.grad(lambda t:rev.frequency(t,16,axis))(1.)))>1e-4


def test_learned_so4_relative_offsets():
    grid=4
    basis=jnp.tile(jnp.eye(4)[:,None,None,:],(1,1,grid*grid,1))
    rawx,rawy=jnp.array([[1.]]),jnp.array([[0.7]])
    q,_=rev.learned_bicayley(basis,basis,rawx,rawy,grid)
    matrices=q[:,0].transpose(1,2,0)
    for r1,c1,r2,c2 in [(0,0,2,3),(1,1,2,2),(0,1,3,2)]:
        relative=matrices[r1*grid+c1].T@matrices[r2*grid+c2]
        np.testing.assert_allclose(relative,matrices[(r2-r1)*grid+c2-c1],atol=1e-12)
    np.testing.assert_allclose(matrices.transpose(0,2,1)@matrices,np.broadcast_to(np.eye(4),matrices.shape),atol=1e-12)


def test_pade_residual_identity_and_newton_error_bound():
    t=sp.symbols("t",nonnegative=True)
    denominator=1+sp.Rational(3,2)*t+sp.Rational(5,4)*t*t+sp.Rational(1,2)*t**3
    root=t+(1+t/2+t*t/4)/denominator
    assert sp.factor(root*root-(1+t*t)+t**5/(8*denominator**2))==0
    a=np.geomspace(1e-9,1e9,101)
    r0=a+(1+0.5*a+0.25*a*a)/(1+1.5*a+1.25*a*a+0.5*a*a*a)
    r1=0.5*(r0+(1+a*a)/r0)
    r2=0.5*(r1+(1+a*a)/r1)
    truth=np.sqrt(1+a*a)
    np.testing.assert_array_less(np.abs(r0*r0/(1+a*a)-1),1/128+1e-14)
    assert np.max(np.abs(r2/truth-1))<1/(8*65024**2)+1e-14
    for steps in [0,1,2]:
        x=jnp.array([-100.,-1.,0.,1.,100.])
        np.testing.assert_allclose(rev.pade_base(x,newton_steps=steps)*rev.pade_base(-x,newton_steps=steps),1,rtol=1e-13)
        assert float(jax.grad(lambda t:rev.pade_base(t,newton_steps=steps))(0.))==pytest.approx(1,abs=1e-13)


def test_static_counter_counts_all_scan_trips_and_matmuls():
    def recurrent(x):
        def body(c,_):return jax.lax.rsqrt(1+c),None
        return jax.lax.scan(body,x,None,length=7)[0]
    x=jax.ShapeDtypeStruct((4,),jnp.float32)
    result=forward_backward(recurrent,[x],x)
    assert result["rsqrt"]==28
    a=jax.ShapeDtypeStruct((3,5),jnp.float32)
    b=jax.ShapeDtypeStruct((5,7),jnp.float32)
    stats=count(jax.make_jaxpr(lambda a,b:a@b)(a,b))
    assert stats["matmul"]==2*3*5*7
    with pytest.raises(ValueError,match="unsupported primitive"):
        count(jax.make_jaxpr(jnp.exp)(x))


def test_pade_analytic_differential_includes_all_parameters():
    s=jnp.array([-20.,-2.,0.,0.3,5.])
    b,t=jnp.array(0.25),jnp.array(0.8)
    ds=jnp.array([0.2,0.4,-0.3,0.1,-0.2])
    for steps in [0,1,2]:
        def raw(s,b,t):
            base=rev.pade_base(s*t/8,b,steps)
            weight=ref.square_power(base)
            return weight/jnp.sum(weight)
        expected=jax.jvp(raw,(s,b,t),(ds,jnp.array(0.1),jnp.array(-0.07)))
        actual=jax.jvp(lambda s,b,t:rev.pade_rows(s,b,t,steps),(s,b,t),(ds,jnp.array(0.1),jnp.array(-0.07)))
        for a,e in zip(actual,expected):np.testing.assert_allclose(a,e,atol=1e-12,rtol=1e-11)


def test_rational_gate_is_c2_at_zero_with_bounded_derivative():
    x=jnp.array([-1e5,-10.,-1.,0.,1.,10.,1e5])
    slope=jax.vmap(jax.grad(rev.rational_gate))(x)
    assert bool(jnp.all((slope>=0)&(slope<=1)))
    assert float(rev.rational_gate(jnp.array(0.)))==0
    assert float(jax.grad(rev.rational_gate)(0.))==pytest.approx(0.5)
    assert float(jax.grad(jax.grad(rev.rational_gate))(0.))==pytest.approx(0.851)
    np.testing.assert_allclose(rev.rational_gate(x)-rev.rational_gate(-x),x,atol=1e-10)


def test_fixed_pade_rule_is_the_derivative_of_its_actual_value():
    s=jnp.array([-20.,-2.,0.,0.3,5.])
    direction=jnp.array([0.2,-0.1,0.3,-0.4,0.1])
    expected=jax.jvp(lambda x:rev.pade_rows(x,newton_steps=1),(s,),(direction,))
    actual=jax.jvp(rev.fixed_pade_rows,(s,),(direction,))
    for a,e in zip(actual,expected):np.testing.assert_allclose(a,e,atol=1e-12,rtol=1e-11)


def test_gate_rejects_missing_cells_and_real_attention_regressions():
    import copy
    import json
    from pathlib import Path
    from analysis.phase1 import revision_report as report
    root=Path(__file__).resolve().parents[1]/"results/phase1"
    data={n:json.loads((root/f"revision_{n}.json").read_text()) for n in report.SECTIONS}
    costs=json.loads((root/"costs.json").read_text())
    initial={"outliers":json.loads((root/"outliers.json").read_text())}
    assert not report.audit(root,data,costs)[1]
    assert report.evaluate(data,costs,initial)["gradient_pass"]
    incomplete=copy.deepcopy(data)
    incomplete["attention"]["precision"].pop()
    assert report.audit(root,incomplete,costs)[1]
    regressed=copy.deepcopy(data)
    regressed["attention"]["records"][0]["gradient_ratio"]=0.999999
    assert not report.evaluate(regressed,costs,initial)["gradient_pass"]
    bad_costs=copy.deepcopy(costs)
    bad_costs["records"][0]["batch_affine_certificate"]["rsqrt"]["all_positive_batches_nonregressing"]=False
    assert not report.evaluate(data,bad_costs,initial)["training_operation_count_pass"]
