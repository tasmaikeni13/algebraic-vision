"""Kernel/reference checks cover both directions and padded ViT/head shapes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic_vision.attention import radical_attention, softmax_attention
from algebraic_vision.kernels import fused_attention, fused_power_score
from algebraic_vision.losses import power_score


@pytest.mark.parametrize("kind", ["softmax", "radical"])
@pytest.mark.parametrize("tokens,dtype", [(16, jnp.float32), (196, jnp.bfloat16),
                                         (197, jnp.float32)])
def test_attention_values_and_all_gradients(kind, tokens, dtype):
    keys = jax.random.split(jax.random.PRNGKey(71), 4)
    q, k, v, g = (jax.random.normal(key, (1, 2, tokens, 32), dtype=dtype) for key in keys)
    sink = jnp.array([0.0, 0.7], jnp.float32)

    def reference(q, k, v, sink):
        return (softmax_attention(q, k, v) if kind == "softmax" else
                radical_attention(q, k, v, sink, 8, 1.5))

    def kernel(q, k, v, sink):
        return fused_attention(q, k, v, kind, sink, 8, 1.5, interpret=True)

    args = (q, k, v, sink)
    np.testing.assert_allclose(jax.jit(kernel)(*args).astype(jnp.float32),
                               reference(*args).astype(jnp.float32), rtol=0.02, atol=2e-3)
    def grad(fn):
        return jax.jit(jax.grad(lambda *xs: jnp.sum(fn(*xs).astype(jnp.float32) * g),
                                argnums=(0, 1, 2, 3)))(*args)
    got, want = grad(kernel), grad(reference)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a.astype(jnp.float32), b.astype(jnp.float32),
                                   rtol=0.025 if dtype == jnp.bfloat16 else 2e-4,
                                   atol=4e-3 if dtype == jnp.bfloat16 else 2e-6)


@pytest.mark.parametrize("exponent,order,classes", [(-3, 8, 11), (-6, 64, 1000), (1, 8, 100)])
@pytest.mark.parametrize("normalize", ["none", "rms"])
def test_power_value_and_gradient(exponent, order, classes, normalize):
    from algebraic_vision.losses import _normalize_logits
    z = jax.random.normal(jax.random.PRNGKey(17), (9, classes), dtype=jnp.float32) * 3
    labels = jnp.arange(9) % classes
    reference = lambda z: power_score(z, labels, exponent, order, 0.1, normalize)
    kernel = lambda z: fused_power_score(_normalize_logits(z, normalize, 1.0),
                                         labels, exponent, order, 0.1, interpret=True)
    a, ga = jax.jit(jax.value_and_grad(kernel))(z)
    b, gb = jax.value_and_grad(reference)(z)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=3e-5)
    np.testing.assert_allclose(ga, gb, rtol=5e-4, atol=3e-6)


def test_kernel_rejects_invalid_kind_and_order():
    q = jnp.zeros((1, 1, 16, 32), jnp.float32)
    with pytest.raises(ValueError, match="unknown attention"):
        fused_attention(q, q, q, "typo")
    with pytest.raises(ValueError, match="power of two"):
        fused_attention(q, q, q, "radical", order=3)
    with pytest.raises(ValueError, match="head"):
        fused_attention(q, q, q, sink=jnp.zeros(2))
