"""Per-block attention diagnostics on valid held-out images."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from algebraic_vision import augment, data, model
from algebraic_vision.attention import attention_weights, _scores


def distributed_diagnostics(cfg, mc, params, source, indices, mesh, shard):
    def local(p, images, valid):
        x = model.embed(mc, p, augment.eval_preprocess(images, cfg.image_size))
        tables = model.shared_rotary_tables(mc)
        rows = []
        for block in p["blocks"]:
            q, k, v = model.block_qkv(mc, x, block, tables)
            weights = attention_weights(q, k, mc.attention, model.sink_mass(mc, block), mc.order, mc.temperature)
            entropy = -jnp.sum(weights * jnp.log(jnp.maximum(weights, 1e-30)), -1)
            scores = jnp.abs(_scores(q, k))
            # Per-example medians before the global mean; exact definition recorded.
            values = jnp.stack([jnp.mean(entropy, (1, 2)), jnp.mean(scores, (1, 2, 3)),
                                jnp.median(scores.reshape(scores.shape[0], -1), axis=-1)], -1)
            rows.append(jnp.sum(values * valid[:, None], 0))
            x = model.block_forward(mc, x, block, tables)
        return jax.lax.psum((jnp.stack(rows), jnp.sum(valid)), "data")
    evaluate = jax.jit(shard_map(local, mesh=mesh, in_specs=(P(), P("data"), P("data")),
                       out_specs=(P(), P()), check_rep=False),
                       in_shardings=(NamedSharding(mesh, P()), shard, shard))
    sums, count = None, 0
    for i, (images, _, valid) in enumerate(data.eval_batches(source, indices, cfg.eval_batch,
                                               jax.process_index(), jax.process_count())):
        if i >= cfg.diagnostics_batches:
            break
        x = jax.make_array_from_process_local_data(shard, images, (cfg.eval_batch,) + images.shape[1:])
        mask = jax.make_array_from_process_local_data(shard, valid.astype(np.float32), (cfg.eval_batch,))
        values, n = evaluate(params, x, mask)
        values, n = np.asarray(values.addressable_data(0)), float(n.addressable_data(0))
        sums, count = values if sums is None else sums + values, count + n
    return [{"block": i, "images": int(count), "attention_entropy": float(row[0] / count),
             "mean_abs_score": float(row[1] / count), "mean_example_median_abs_score": float(row[2] / count)}
            for i, row in enumerate(sums)]
