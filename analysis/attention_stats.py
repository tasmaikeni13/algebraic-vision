#!/usr/bin/env python3
"""Attention statistics of trained models.

Usage:
  python analysis/attention_stats.py OUT.json NAME=params.pkl:config.json ...

For each trained model, runs 256 validation images and records per block:
mean normalised attention entropy, fraction of query rows whose largest
weight exceeds 0.9 (near one-hot), mean largest weight, quantiles of |score|,
mean visible mass (1 - sink share), and the fraction of rows whose
score-Jacobian norm is below 1e-4 (saturated).
"""

import json
import math
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from algebraic_vision.attention import _scores
from algebraic_vision.model import (block_forward, block_qkv, embed,
                                    shared_rotary_tables, sink_mass)
from algebraic_vision.primitives import radical_exp, radical_exp_slope
from algebraic_vision.train import TrainConfig, build_model_config


def block_attention(cfg, x, block, tables):
    t = x.shape[1]
    q, k, _ = block_qkv(cfg, x, block, tables)
    s = _scores(q, k)
    if cfg.attention == "softmax":
        p = jax.nn.softmax(s, -1)
        g = jnp.ones_like(s)
    else:
        z = cfg.temperature * s
        e = radical_exp(z, cfg.order)
        omega = sink_mass(cfg, block).reshape(1, -1, 1, 1)
        p = e / (omega + jnp.sum(e, -1, keepdims=True))
        g = cfg.temperature * radical_exp_slope(z, cfg.order)
    mass = jnp.sum(p, -1)
    pn = p / mass[..., None]
    ent = -jnp.sum(pn * jnp.log(jnp.maximum(pn, 1e-30)), -1) / math.log(t)
    pp = jnp.sum(p * p, -1, keepdims=True)
    rest = jnp.maximum(pp - p * p, 0.0)
    jf = jnp.sqrt(jnp.sum(g * g * p * p * ((1 - p) ** 2 + rest), -1))
    pmax = jnp.max(p, -1)
    abs_s = jnp.abs(s).reshape(-1)
    return {"entropy": jnp.mean(ent),
            "entropy_per_head": jnp.mean(ent, (0, 2)),
            "onehot_rows": jnp.mean(pmax > 0.9), "max_weight": jnp.mean(pmax),
            "score_q50": jnp.quantile(abs_s, 0.5),
            "score_q99": jnp.quantile(abs_s, 0.99),
            "score_max": jnp.max(abs_s), "mass": jnp.mean(mass),
            "saturated_rows": jnp.mean(jf < 1e-4)}


def analyse(params, cfg, images):
    tables = shared_rotary_tables(cfg)

    def run(p, x):
        h = embed(cfg, p, x)
        rows = []
        for block in p["blocks"]:
            rows.append(block_attention(cfg, h, block, tables))
            h = block_forward(cfg, h, block, tables)
        return rows

    rows = jax.jit(run)(params, images)
    return [{k: np.asarray(v).tolist() for k, v in r.items()} for r in rows]


def validation_images(size, count=256):
    """First `count` validation images, preprocessed as in evaluation."""
    if size == 64:
        imgs = np.load("/dev/shm/data/in64/val_images.npy",
                       mmap_mode="r")[:count]
    else:
        from algebraic_vision import data
        source = data.open_source("/dev/shm/data/in256", "val", workers=8)
        imgs, _ = source.load(np.arange(count))
        source.close()
        off = (imgs.shape[1] - size) // 2
        imgs = imgs[:, off:off + size, off:off + size]
    return jnp.asarray(imgs, jnp.float32) / 127.5 - 1.0


def main():
    out = Path(sys.argv[1])
    record, images = {}, {}
    for item in sys.argv[2:]:
        name, paths = item.split("=", 1)
        params_path, config_path = paths.split(":")
        cfg = build_model_config(TrainConfig(**json.loads(
            Path(config_path).read_text())["config"]))
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        if cfg.image_size not in images:
            images[cfg.image_size] = validation_images(cfg.image_size)
        rows = analyse(params, cfg, images[cfg.image_size])
        record[name] = rows
        print(f"== {name}")
        for i, r in enumerate(rows):
            print(f"  block {i:2d} H={r['entropy']:.3f} "
                  f"onehot={r['onehot_rows']:.3f} "
                  f"pmax={r['max_weight']:.3f} |s|50={r['score_q50']:.2f} "
                  f"|s|99={r['score_q99']:.2f} max={r['score_max']:.1f} "
                  f"mass={r['mass']:.3f} sat={r['saturated_rows']:.3f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
