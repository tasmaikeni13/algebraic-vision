"""The kernel training region synchronizes updates and evaluation on devices."""

import os
from pathlib import Path
import subprocess
import sys


def test_two_device_training_matches_one_device():
    script = r'''
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from algebraic_vision.train import TrainConfig, build_model_config, build_steps, make_optimizer
from algebraic_vision.model import init_params
devices = jax.devices()
assert len(devices) == 2
images = np.random.default_rng(2).integers(0, 256, (4, 16, 16, 3), dtype=np.uint8)
labels = np.array([0, 2, 1, 3], np.int32)
valid = np.array([True, False, True, True])
for arm, backend in (('standard', 'xla'), ('algebraic', 'pallas')):
    cfg = TrainConfig(arm=arm, loss_backend=backend,
                      image_size=16, total_steps=3, warmup_steps=0, batch_size=4,
                      model={'width':32,'depth':1,'heads':1,'mlp_dim':64,'patch_size':8,
                             'num_classes':7,'compute_dtype':'float32','attention_backend':backend})
    model = build_model_config(cfg)
    original = init_params(model, jax.random.PRNGKey(17))
    outputs = []
    for count in (1, 2):
        mesh = Mesh(np.array(devices[:count]), ('data',))
        rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P('data'))
        params = jax.tree_util.tree_map(lambda x: jax.device_put(np.asarray(x).copy(), rep), original)
        tx, _ = make_optimizer(cfg)
        state = jax.device_put(tx.init(params), rep)
        train, evaluate, _ = build_steps(cfg, model, tx, mesh)
        x, y, mask = (jax.device_put(a, shard) for a in (images, labels, valid))
        for step in range(2):
            params, state, metrics = train(params, state, x, y, jnp.int32(step))
        evaluation = evaluate(params, x, y, mask)
        outputs.append(jax.tree_util.tree_map(np.asarray, (params, metrics, evaluation)))
    for a, b in zip(jax.tree_util.tree_leaves(outputs[0]), jax.tree_util.tree_leaves(outputs[1])):
        np.testing.assert_allclose(a, b, rtol=4e-4, atol=2e-5)
    assert float(outputs[1][2]['count']) == 3
print('Two-device XLA and interpreted-Pallas training/evaluation passed')
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="0",
               XLA_FLAGS="--xla_force_host_platform_device_count=2", OPENBLAS_NUM_THREADS="1")
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                   env=env, check=True, capture_output=True, text=True, timeout=120)
