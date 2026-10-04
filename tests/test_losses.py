"""Losses: cross-entropy, power scores and Fenchel-Young losses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic_vision.losses import (cross_entropy, entmax15_fy,
                                     entmax15_probabilities, power_score,
                                     radical_fy, radical_fy_probabilities,
                                     radical_probabilities, smoothed_targets,
                                     squareplus_fy, squareplus_probabilities)

jax.config.update("jax_enable_x64", True)


def test_cross_entropy_matches_definition():
    z = jax.random.normal(jax.random.PRNGKey(5), (8, 5))
    labels = jnp.arange(8) % 5
    y = jax.nn.one_hot(labels, 5) * 0.9 + 0.02
    ref = -jnp.mean(jnp.sum(y * jax.nn.log_softmax(z), -1))
    np.testing.assert_allclose(cross_entropy(z, labels, 0.1), ref,
                               rtol=1e-6)


def _power_reference(z, labels, a, order, smoothing):
    p = radical_probabilities(z, order)
    k = z.shape[-1]
    y = jax.nn.one_hot(labels, k) * (1 - smoothing) + smoothing / k
    return jnp.mean(jnp.sum(p ** a, -1) / a - jnp.sum(y * p ** (a - 1), -1)
                    / (a - 1) + jnp.sum(y ** a, -1) / (a * (a - 1)))


@pytest.mark.parametrize("exponent,order,smoothing", [(-3, 8, 0.0),
                                                      (-3, 16, 0.1),
                                                      (-2, 8, 0.1),
                                                      (1, 8, 0.1),
                                                      (-6, 64, 0.1)])
def test_power_score_matches_reference(exponent, order, smoothing):
    z = jax.random.normal(jax.random.PRNGKey(3), (32, 11)) * 3
    labels = jax.random.randint(jax.random.PRNGKey(4), (32,), 0, 11)
    a = 1 - 2.0 ** exponent if exponent < 0 else 1 + 2.0 ** -exponent
    got = power_score(z, labels, exponent, order, smoothing)
    ref = _power_reference(z, labels, a, order, smoothing)
    np.testing.assert_allclose(got, ref, rtol=1e-6)
    g1 = jax.grad(lambda v: power_score(v, labels, exponent, order,
                                        smoothing))(z)
    g2 = jax.grad(lambda v: _power_reference(v, labels, a, order,
                                             smoothing))(z)
    np.testing.assert_allclose(g1, g2, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("a", [7.0 / 8.0, 63.0 / 64.0])
def test_power_score_is_proper_on_soft_targets(a):
    # The expected score over labels drawn from q is minimised at p = q.
    q = jnp.array([0.5, 0.3, 0.2])

    def expected(p):
        return jnp.sum(p ** a) / a - jnp.sum(q * p ** (a - 1)) / (a - 1)

    best = expected(q)
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = jnp.asarray(rng.dirichlet(np.ones(3)))
        assert float(expected(p)) >= float(best) - 1e-12


PROBS = [squareplus_probabilities, radical_fy_probabilities,
         lambda z: radical_fy_probabilities(z, 16), entmax15_probabilities]


@pytest.mark.parametrize("prob", PROBS)
@pytest.mark.parametrize("scale", [0.5, 5.0, 40.0])
def test_fy_predictions_lie_on_simplex(prob, scale):
    z = jax.random.normal(jax.random.PRNGKey(0), (16, 1000)) * scale
    p = prob(z)
    assert float(jnp.min(p)) >= 0.0
    np.testing.assert_allclose(jnp.sum(p, -1), 1.0, atol=2e-5)
    # Monotone in the logits: larger logit, larger probability.
    order = jnp.argsort(z, -1)
    ps = jnp.take_along_axis(p, order, -1)
    assert float(jnp.min(jnp.diff(ps, axis=-1))) >= -1e-9


@pytest.mark.parametrize("loss,prob", [
    (squareplus_fy, squareplus_probabilities),
    (entmax15_fy, entmax15_probabilities),
])
def test_closed_form_fy_value_and_gradient(loss, prob):
    z = jax.random.normal(jax.random.PRNGKey(1), (8, 50)) * 3
    labels = jnp.arange(8) % 50
    values = loss(z, labels, 0.1, reduction="none")
    assert float(jnp.min(values)) >= -1e-9          # Fenchel-Young inequality
    grad = jax.grad(lambda v: loss(v, labels, 0.1))(z)
    y = smoothed_targets(labels, 50, 0.1)
    np.testing.assert_allclose(grad, (prob(z) - y) / 8, atol=1e-12)
    d = jax.random.normal(jax.random.PRNGKey(2), z.shape) * 1e-6
    fd = (loss(z + d, labels, 0.1) - loss(z - d, labels, 0.1)) / 2
    np.testing.assert_allclose(jnp.sum(grad * d), fd, rtol=1e-6)


def test_radical_fy_gradient_and_minimum():
    z = jax.random.normal(jax.random.PRNGKey(3), (8, 50)) * 3
    labels = jnp.arange(8) % 50
    grad = jax.grad(lambda v: radical_fy(v, labels, 8, 0.1))(z)
    y = smoothed_targets(labels, 50, 0.1)
    np.testing.assert_allclose(grad, (radical_fy_probabilities(z) - y) / 8,
                               atol=1e-12)
