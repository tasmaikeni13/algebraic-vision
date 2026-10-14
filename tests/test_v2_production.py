"""Check production v2 against the accepted study and frozen baselines."""

import dataclasses
import subprocess
import types

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from algebraic_vision import model, v2
from algebraic_vision.production import ProductionConfig, host_schedule, optimizer_init, optimizer_update, target_loss
from algebraic_vision.train import build_model_config, loss_and_probs
from experiments.production import jobs
from analysis.production_fairness import check_matrix

SMALL = dict(image_size=32, patch_size=8, width=64, depth=2, heads=2, mlp_dim=128, num_classes=7, compute_dtype="float32")


def test_baseline_initialization_forward_and_vjp_are_unchanged():
    frozen = types.ModuleType("frozen_model")
    source = subprocess.check_output(["git", "show", "e265e04:algebraic_vision/model.py"], text=True)
    exec(compile(source, "frozen_model.py", "exec"), frozen.__dict__)
    images = jax.random.uniform(jax.random.PRNGKey(19), (2, 32, 32, 3))
    for make in ("standard_config", "algebraic_config"):
        old_cfg, new_cfg = getattr(frozen, make)(**SMALL), getattr(model, make)(**SMALL)
        old = frozen.init_params(old_cfg, jax.random.PRNGKey(42))
        new = model.init_params(new_cfg, jax.random.PRNGKey(42))
        for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
            np.testing.assert_array_equal(a, b)
        # A zero-initialized head would hide any backbone regression.
        head = jax.random.normal(jax.random.PRNGKey(5), old["head"]["w"].shape) * 0.1
        old["head"]["w"], new["head"]["w"] = head, head
        np.testing.assert_array_equal(frozen.forward(old_cfg, old, images), model.forward(new_cfg, new, images))
        ga = jax.grad(lambda p: jnp.sum(frozen.forward(old_cfg, p, images)))(old)
        gb = jax.grad(lambda p: jnp.sum(model.forward(new_cfg, p, images)))(new)
        for a, b in zip(jax.tree.leaves(ga), jax.tree.leaves(gb)):
            np.testing.assert_array_equal(a, b)


def test_v2_matches_the_selected_research_equations_and_shared_weights():
    from analysis.phase1 import references as ref, revisions as rev
    cfg = model.algebraic_v2_config(**SMALL)
    params = model.init_params(cfg, jax.random.PRNGKey(42))
    standard = model.init_params(model.standard_config(**SMALL), jax.random.PRNGKey(42))
    for name in ("patch", "head", "norm", "posemb"):
        for a, b in zip(jax.tree.leaves(params[name]), jax.tree.leaves(standard[name])):
            np.testing.assert_array_equal(a, b)
    for a, b in zip(params["blocks"], standard["blocks"]):
        for name in b:
            for aa, bb in zip(jax.tree.leaves(a[name]), jax.tree.leaves(b[name])):
                np.testing.assert_array_equal(aa, bb)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 2, 16, 32), dtype=jnp.float32)
    rot = params["blocks"][0]["rot"]
    expected = rev.learned_bicayley(x, x, rot["raw_x"], rot["raw_y"], 4)[0]
    np.testing.assert_allclose(v2.rotate(x, v2.rotary_tables(cfg, rot)), expected, atol=1e-6)
    np.testing.assert_allclose(v2.rational_gate(x), rev.rational_gate(x), atol=1e-6)
    np.testing.assert_allclose(v2.inverse_energy_pool(x), ref.robust_pool(x), atol=1e-6)
    y = jnp.array([1, 3])
    z = jax.random.normal(jax.random.PRNGKey(2), (2, 7), dtype=jnp.float32)
    from algebraic_vision.losses import power_score
    cfg_train = ProductionConfig(arm="algebraic_v2", loss_backend="xla")
    targets = jax.nn.one_hot(y, 7) * 0.9 + 0.1 / 7
    np.testing.assert_allclose(target_loss(cfg_train, z, targets), power_score(2*z, y, -3, 8, 0.1), atol=1e-5)
    train_loss, _ = loss_and_probs(cfg_train)
    np.testing.assert_allclose(jax.grad(train_loss)(z, y), jax.grad(lambda z: target_loss(cfg_train, z, targets))(z), atol=2e-6)


def test_pure_adamw_matches_optax_and_has_no_transcendentals():
    cfg = ProductionConfig(total_steps=10, warmup_steps=0, lr=0.001, weight_decay=0.1)
    params = {"w": jnp.ones((4, 4), jnp.float32), "b": jnp.ones(4, jnp.float32)}
    grads = {"w": jnp.arange(16, dtype=jnp.float32).reshape(4, 4) * 0.01, "b": jnp.full(4, -0.05)}
    state = optimizer_init(params)
    from algebraic_vision.train import decay_mask
    reference = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adamw(cfg.lr, weight_decay=0.1, mask=decay_mask))
    want, refstate = params, reference.init(params)
    for step in range(3):
        scalars = np.array([cfg.lr, 1 / (1-cfg.b1**(step+1)), 1 / (1-cfg.b2**(step+1))], np.float32)
        params, state, _ = optimizer_update(cfg, params, state, grads, scalars)
        updates, refstate = reference.update(grads, refstate, want)
        want = optax.apply_updates(want, updates)
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(want)):
            np.testing.assert_allclose(a, b, atol=2e-7)
    from scripts.audit_production import scan_jaxpr
    result = scan_jaxpr(jax.make_jaxpr(lambda p, s, g, c: optimizer_update(cfg, p, s, g, c))(params, state, grads, scalars))
    assert result["pure"]


def test_matrices_are_complete_and_fairness_fails_closed():
    matrix = jobs("sweep")
    assert len(matrix) == 108
    assert check_matrix(matrix)["pass"]
    smoke = jobs("smoke")
    assert len(smoke) == 18 and all(j["total_steps"] == 3 for j in smoke)
    check_matrix(smoke)
    with pytest.raises(ValueError, match="incomplete"):
        check_matrix(smoke[:-1])
    changed = [dict(j) for j in smoke]
    changed[0]["stochastic_depth"] = 0.5
    with pytest.raises(ValueError, match="unmatched"):
        check_matrix(changed)
    with pytest.raises(ValueError, match="minival"):
        jobs("main")
    template=jobs("main",template=True)
    assert len(template)==18 and not any(c["launch_ready"] for c in template)
    check_matrix(template)
    with pytest.raises(ValueError,match="template"):
        ProductionConfig(**template[0]).validate()


def test_scales_and_schedules():
    from experiments.production import base
    for scale, width, depth, heads, mlp in [("vit-b16",768,12,12,3072), ("vit-l16",1024,24,16,4096)]:
        cfg = base(scale, stage="main", data_root="unused")
        mc = build_model_config(cfg)
        assert (mc.width, mc.depth, mc.heads, mc.mlp_dim, mc.num_patches) == (width,depth,heads,mlp,196)
        assert cfg.total_images == 300 * 1_271_167
        assert np.all(np.isfinite(host_schedule(cfg, 0)))
        assert np.all(np.isfinite(host_schedule(cfg, cfg.steps()-1)))
