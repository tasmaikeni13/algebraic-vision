"""Pallas attention kernels against the XLA references (interpret mode)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic_vision.attention import (radical_attention, radical_weights,
                                        softmax_attention)
from algebraic_vision.kernels import _radical_rho, fused_attention

jax.config.update("jax_enable_x64", True)


def _inputs(shape=(2, 3, 20, 16), scale=2.0):
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    q, k, v = (jax.random.normal(kk, shape, jnp.float32) * scale
               for kk in keys[:3])
    g = jax.random.normal(keys[3], shape, jnp.float32)
    return q, k, v, g


@pytest.mark.parametrize("kind", ["softmax", "radical"])
def test_kernel_matches_reference(kind):
    q, k, v, g = _inputs()
    sink = jnp.array([0.5, 1.0, 2.0], jnp.float32)
    if kind == "softmax":
        def ref(q_, k_, v_, s_):
            return softmax_attention(q_, k_, v_)
    else:
        def ref(q_, k_, v_, s_):
            return radical_attention(q_, k_, v_, s_, 8, 1.5)

    def ker(q_, k_, v_, s_):
        return fused_attention(q_, k_, v_, kind, s_, 8, 1.5, interpret=True)

    np.testing.assert_allclose(ker(q, k, v, sink), ref(q, k, v, sink),
                               atol=5e-6)
    g1 = jax.grad(lambda *a: jnp.sum(ker(*a) * g), (0, 1, 2, 3))(q, k, v,
                                                                 sink)
    g2 = jax.grad(lambda *a: jnp.sum(ref(*a) * g), (0, 1, 2, 3))(q, k, v,
                                                                 sink)
    for a, b in zip(g1, g2):
        np.testing.assert_allclose(a, b, atol=2e-5)


def test_direct_radical_form_is_accurate_for_weights():
    # The kernel's division-free form loses relative accuracy only on
    # negligible weights; normalised weights stay within 1e-6.
    s = jnp.linspace(-1000.0, 30.0, 197, dtype=jnp.float32)[None, :]
    exact = radical_weights(s.astype(jnp.float64),
                            jnp.zeros((1, 1), jnp.float64), 8, 1.0)
    e, _ = _radical_rho(s / 8.0, 8, exact=False)
    direct = e / jnp.sum(e, -1, keepdims=True)
    assert float(jnp.max(jnp.abs(direct - exact))) < 1e-6
