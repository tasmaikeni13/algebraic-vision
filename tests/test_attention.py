"""Radical attention weights: custom gradient, range, sink mass."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic_vision.attention import radical_weights
from algebraic_vision.model import forward, init_params, standard_config
from algebraic_vision.primitives import radical_exp

jax.config.update("jax_enable_x64", True)


def _naive_weights(s, sink, order):
    e = radical_exp(s, order)
    return e / (sink + jnp.sum(e, -1, keepdims=True))


@pytest.mark.parametrize("order", [8, 16])
def test_radical_weights_vjp_matches_autodiff(order):
    key = jax.random.PRNGKey(0)
    s = jax.random.normal(key, (4, 9)) * 3.0
    sink = jnp.full((4, 1), 0.7)
    g = jax.random.normal(jax.random.PRNGKey(1), (4, 9))

    def f_custom(s_, k_):
        return jnp.sum(radical_weights(s_, k_, order, 1.5) * g)

    def f_naive(s_, k_):
        return jnp.sum(_naive_weights(1.5 * s_, k_, order) * g)

    for i in (0, 1):
        a = jax.grad(f_custom, argnums=i)(s, sink)
        b = jax.grad(f_naive, argnums=i)(s, sink)
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14)


def test_radical_weights_large_scores_float32():
    # Autodiff of e / D would form D^2, which overflows for n = 16 at
    # |s| ~ 120; the custom gradient does not.
    s = jnp.linspace(-150.0, 150.0, 197, dtype=jnp.float32)[None]
    sink = jnp.ones((1, 1), jnp.float32)
    g = jnp.ones_like(s)
    grad = jax.grad(lambda v: jnp.sum(radical_weights(v, sink, 16, 1.0)
                                      * g * jnp.arange(197.0)))(s)
    assert bool(jnp.all(jnp.isfinite(grad)))
    ref = jax.grad(lambda v: jnp.sum(
        _naive_weights(v, 1.0, 16) * jnp.arange(197.0)))(
            s.astype(jnp.float64))
    np.testing.assert_allclose(grad, ref, rtol=1e-4, atol=1e-7)


def test_sink_mass():
    s = jax.random.normal(jax.random.PRNGKey(2), (3, 7))
    sink = jnp.array([[0.0], [1.0], [5.0]])
    p = radical_weights(s, sink, 8, 1.0)
    d = sink + jnp.sum(radical_exp(s, 8), -1, keepdims=True)
    np.testing.assert_allclose(jnp.sum(p, -1, keepdims=True), 1 - sink / d,
                               rtol=1e-12)


def test_radical_attention_is_softmax_like_at_init():
    cfg = standard_config(image_size=32, patch_size=8, width=64, depth=2,
                          heads=2, mlp_dim=128)
    rad = replace(cfg, attention="radical")
    params = init_params(cfg, jax.random.PRNGKey(5))
    x = jax.random.uniform(jax.random.PRNGKey(6), (2, 32, 32, 3),
                           minval=-1, maxval=1)
    params["head"]["w"] = jax.random.normal(jax.random.PRNGKey(7),
                                            params["head"]["w"].shape)
    a, b = forward(cfg, params, x), forward(rad, params, x)
    assert float(jnp.max(jnp.abs(a - b))) < 0.05 * float(jnp.max(jnp.abs(a)))
