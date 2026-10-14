"""Numerical tests of the algebraic primitives."""

import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from algebraic_vision.primitives import (alu, cayley_frequencies,
                                         cayley_tables, radical_exp,
                                         radical_exp_slope, radical_gelu,
                                         radical_sigmoid)

jax.config.update("jax_enable_x64", True)
mp.mp.dps = 40


@pytest.mark.parametrize("order", [2, 4, 8, 16])
def test_radical_exp_matches_reference(order):
    s = np.linspace(-60.0, 60.0, 241)
    got = np.asarray(radical_exp(jnp.asarray(s), order))
    ref = np.array([float((mp.mpf(v) / order
                           + mp.sqrt(1 + (mp.mpf(v) / order) ** 2)) ** order)
                    for v in s])
    np.testing.assert_allclose(got, ref, rtol=1e-13)


@pytest.mark.parametrize("order", [4, 8, 16])
def test_radical_exp_derivative(order):
    s = jnp.linspace(-40.0, 40.0, 161)
    grad = jax.vmap(jax.grad(lambda v: radical_exp(v, order)))(s)
    expected = radical_exp(s, order) * radical_exp_slope(s, order)
    np.testing.assert_allclose(grad, expected, rtol=1e-12)


def test_radical_exp_reciprocal_and_taylor():
    s = jnp.linspace(-30.0, 30.0, 121)
    np.testing.assert_allclose(radical_exp(s, 8) * radical_exp(-s, 8), 1.0,
                               rtol=1e-13)
    small = jnp.array([1e-3, -2e-3])
    np.testing.assert_allclose(radical_exp(small, 8), jnp.exp(small),
                               rtol=1e-8)


def test_radical_exp_rejects_bad_order():
    with pytest.raises(ValueError):
        radical_exp(jnp.ones(3), 6)


def test_alu_matches_formula_and_derivative_range():
    c = math.sqrt(2.0 / math.pi)
    x = jnp.linspace(-80.0, 80.0, 4001)
    ref = x * (1 + c * x / jnp.sqrt(1 + (c * x) ** 2)) / 2
    np.testing.assert_allclose(alu(x, c), ref, rtol=1e-12, atol=1e-15)
    d = jax.vmap(jax.grad(lambda v: alu(v, c)))(x)
    assert float(d.min()) >= 0.5 - 2 * math.sqrt(6) / 9 - 1e-12
    assert float(d.max()) <= 0.5 + 2 * math.sqrt(6) / 9 + 1e-12
    eps = 1e-6
    fd = (alu(x + eps, c) - alu(x - eps, c)) / (2 * eps)
    np.testing.assert_allclose(d, fd, atol=1e-8)


def test_alu_keeps_negative_tail_in_float32():
    x = jnp.array([-1e3, -1e4], jnp.float32)
    c = 1.0
    out = np.asarray(alu(x, c), np.float64)
    ref = 1 / (4 * c * c * np.asarray(x, np.float64))
    np.testing.assert_allclose(out, ref, rtol=1e-5)


def test_radical_sigmoid_properties():
    y = jnp.linspace(-60.0, 60.0, 2401)
    s = radical_sigmoid(y, 8)
    np.testing.assert_allclose(s + radical_sigmoid(-y, 8), 1.0, atol=1e-14)
    small = jnp.linspace(-0.05, 0.05, 11)
    np.testing.assert_allclose(radical_sigmoid(small, 8),
                               jax.nn.sigmoid(small), atol=1e-7)
    d = jax.vmap(jax.grad(lambda v: radical_sigmoid(v, 8)))(y)
    assert float(d.min()) >= 0.0 and float(d.max()) <= 0.25 + 1e-12


def test_radical_gelu_close_to_gelu_and_stable():
    x = jnp.linspace(-4.0, 4.0, 81)
    gap = jnp.max(jnp.abs(radical_gelu(x)
                          - jax.nn.gelu(x, approximate=False)))
    assert float(gap) < 0.03
    big = jnp.array([-1e5, -300.0, 300.0, 1e5], jnp.float32)
    g = jax.vmap(jax.grad(radical_gelu))(big)
    assert bool(jnp.all(jnp.isfinite(radical_gelu(big))))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_cayley_tables_are_exact_rotations():
    w = cayley_frequencies(8)
    cos_like, sin_like = cayley_tables(16, w)
    theta = 2 * np.arctan(np.asarray(w, np.float64))
    pos = np.arange(16)[:, None]
    np.testing.assert_allclose(cos_like, np.cos(pos * theta), atol=2e-6)
    np.testing.assert_allclose(sin_like, np.sin(pos * theta), atol=2e-6)
    np.testing.assert_allclose(cos_like ** 2 + sin_like ** 2, 1.0, atol=1e-6)
