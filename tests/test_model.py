"""Model structure, position rotations and the purity of the graph."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic_vision.model import (algebraic_config, count_params, forward,
                                    init_params, mixed_rotary_tables,
                                    rotary_tables, standard_config)
from algebraic_vision.primitives import rotate_pairs

SMALL = dict(image_size=32, patch_size=8, width=64, depth=2, heads=2,
             mlp_dim=128)


def _tok(r, c, grid=8):
    return grid * r + c


def test_parameter_parity():
    std = init_params(standard_config(**SMALL), jax.random.PRNGKey(0))
    alg_cfg = algebraic_config(**SMALL)
    alg = init_params(alg_cfg, jax.random.PRNGKey(0))
    # The only extra parameters are the learned Cayley frequencies
    # (w_x, w_y per head and feature pair).
    extra = alg_cfg.depth * alg_cfg.heads * alg_cfg.head_dim
    assert count_params(alg) == count_params(std) + extra
    assert extra / count_params(std) < 0.001
    # Shared tensors start from identical values for the same seed.
    np.testing.assert_array_equal(std["blocks"][1]["qkv"]["w"],
                                  alg["blocks"][1]["qkv"]["w"])


def test_forward_shapes_and_finiteness():
    for cfg in (standard_config(**SMALL), algebraic_config(**SMALL),
                algebraic_config(**SMALL, pool="cls", glu=True)):
        params = init_params(cfg, jax.random.PRNGKey(1))
        x = jax.random.uniform(jax.random.PRNGKey(2), (3, 32, 32, 3),
                               minval=-1, maxval=1)
        logits = forward(cfg, params, x)
        assert logits.shape == (3, cfg.num_classes)
        assert bool(jnp.all(jnp.isfinite(logits)))


@pytest.mark.parametrize("rotary", ["cayley2d", "rope2d"])
def test_fixed_rotations_depend_on_relative_offset(rotary):
    cfg = algebraic_config(image_size=64, patch_size=8, width=64, heads=1,
                           rotary=rotary)
    cos, sin = rotary_tables(cfg)
    q = jax.random.normal(jax.random.PRNGKey(3), (64,))
    k = jax.random.normal(jax.random.PRNGKey(4), (64,))
    qs = rotate_pairs(jnp.broadcast_to(q, (64, 64)), cos, sin)
    ks = rotate_pairs(jnp.broadcast_to(k, (64, 64)), cos, sin)
    logits = qs @ ks.T
    # Offsets (2, 3) and (-3, -4) at two different absolute positions.
    np.testing.assert_allclose(logits[_tok(0, 0), _tok(2, 3)],
                               logits[_tok(4, 1), _tok(6, 4)], rtol=2e-5)
    np.testing.assert_allclose(logits[_tok(5, 5), _tok(2, 1)],
                               logits[_tok(7, 6), _tok(4, 2)], rtol=2e-5)


def test_learned_rotations_depend_on_relative_offset():
    cfg = algebraic_config(image_size=64, patch_size=8, width=64, heads=2)
    params = init_params(cfg, jax.random.PRNGKey(4))
    cos, sin = mixed_rotary_tables(cfg, params["blocks"][0]["rot"])
    q = jax.random.normal(jax.random.PRNGKey(5), (32,))
    k = jax.random.normal(jax.random.PRNGKey(6), (32,))
    qs = rotate_pairs(jnp.broadcast_to(q, (64, 32)), cos[1], sin[1])
    ks = rotate_pairs(jnp.broadcast_to(k, (64, 32)), cos[1], sin[1])
    logits = qs @ ks.T
    np.testing.assert_allclose(logits[_tok(0, 0), _tok(3, 2)],
                               logits[_tok(4, 5), _tok(7, 7)], rtol=1e-5)
    np.testing.assert_allclose(logits[_tok(6, 1), _tok(2, 4)],
                               logits[_tok(5, 2), _tok(1, 5)], rtol=1e-5)


def test_algebraic_graph_has_no_transcendental_ops():
    from scripts.audit_purity import audit
    result = audit("algebraic", 32, 8, 64, 2, 2, 128)
    assert result["transcendental_ops"] == {}
    standard = audit("standard", 32, 8, 64, 2, 2, 128)
    assert standard["transcendental_ops"]
