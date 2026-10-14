import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic_vision.flash import flash_attention
from algebraic_vision.attention import softmax_attention


@pytest.mark.parametrize("dtype",[jnp.float32,jnp.bfloat16])
def test_online_flash_values_and_vjp(dtype):
    keys=jax.random.split(jax.random.PRNGKey(63),4)
    q,k,v,g=(jax.random.normal(key,(1,2,197,64),dtype=dtype) for key in keys)
    got=flash_attention(q,k,v,interpret=True)
    want=softmax_attention(q,k,v)
    np.testing.assert_allclose(got.astype(jnp.float32),want.astype(jnp.float32),rtol=0.02,atol=0.004 if dtype==jnp.bfloat16 else 1e-6)
    def gradients(fn):
        return jax.grad(lambda q,k,v:jnp.sum(fn(q,k,v).astype(jnp.float32)*g),argnums=(0,1,2))(q,k,v)
    for a,b in zip(gradients(lambda q,k,v:flash_attention(q,k,v,interpret=True)),gradients(softmax_attention)):
        np.testing.assert_allclose(a.astype(jnp.float32),b.astype(jnp.float32),rtol=0.03 if dtype==jnp.bfloat16 else 2e-4,atol=0.004 if dtype==jnp.bfloat16 else 2e-6)
