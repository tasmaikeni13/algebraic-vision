import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic_vision.rational_kernel import ball_lift,fused_rational,reference


@pytest.mark.parametrize("tokens,dtype",[(196,jnp.bfloat16),(197,jnp.float32)])
def test_rational_values_and_all_four_vjps(tokens,dtype):
    keys=jax.random.split(jax.random.PRNGKey(88),4)
    q,k,v,g=(jax.random.normal(key,(1,2,tokens,64),dtype=dtype) for key in keys)
    q,k=ball_lift(q),ball_lift(k)
    margins=jnp.array([1.05,2.0],jnp.float32)
    args=(q,k,v,margins)
    kernel=lambda *a:fused_rational(*a,interpret=True)
    np.testing.assert_allclose(kernel(*args).astype(jnp.float32),reference(*args).astype(jnp.float32),atol=2e-3,rtol=0.02)
    def gradients(fn):
        return jax.jit(jax.grad(lambda *a:jnp.sum(fn(*a).astype(jnp.float32)*g),argnums=(0,1,2,3)))(*args)
    for a,b in zip(gradients(kernel),gradients(reference)):
        a,b=a.astype(jnp.float32),b.astype(jnp.float32)
        if dtype==jnp.bfloat16:
            # Exact quotient and normalized-log-derivative formulas round
            # differently near cancellation; compare the complete VJP norm.
            assert float(jnp.linalg.norm(a-b)/jnp.maximum(jnp.linalg.norm(b),1e-20)) < 0.01
        else:
            np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-4)
    from scripts.audit_production import scan_jaxpr
    result=scan_jaxpr(jax.make_jaxpr(lambda *a:(kernel(*a),gradients(kernel)))(*args))
    assert result["pure"] and result["rsqrt"]==0
