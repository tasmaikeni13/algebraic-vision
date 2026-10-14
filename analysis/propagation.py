#!/usr/bin/env python3
"""Signal propagation through randomly initialised ViT stacks.

For each configuration and several random initialisations, a batch of real
images is pushed through the blocks and we record, per block,

* residual-stream RMS,
* RMS of the gradient reaching the block input from a Gaussian cotangent on
  the final normalised tokens,
* mean attention entropy (of the visible weights renormalised to sum 1)
  divided by log(tokens), and mean visible attention mass,
* mean pairwise cosine similarity between tokens (rank-collapse indicator).

Writes results/analysis/propagation.json.
"""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from algebraic_vision.attention import attention_weights
from algebraic_vision.model import (algebraic_config, block_forward,
                                    block_qkv, embed, init_params, layer_norm,
                                    shared_rotary_tables, sink_mass,
                                    standard_config)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/propagation.json"


def block_stats(cfg, x, block, tables):
    t = x.shape[1]
    q, k, _ = block_qkv(cfg, x, block, tables)
    w = attention_weights(q, k, cfg.attention, sink_mass(cfg, block),
                          cfg.order, cfg.temperature)
    mass = jnp.sum(w, -1, keepdims=True)
    pn = w / mass
    ent = -jnp.sum(pn * jnp.log(jnp.maximum(pn, 1e-30)), -1) / math.log(t)
    return jnp.mean(ent), jnp.mean(mass)


def token_cosine(x):
    xn = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    gram = jnp.einsum("btd,bsd->bts", xn, xn)
    t = x.shape[1]
    return (jnp.sum(gram, (1, 2)) - t).mean() / (t * (t - 1))


def run(cfg, params, images, cotangent):
    tables = shared_rotary_tables(cfg)
    depth = len(params["blocks"])

    def body(taps):
        x = embed(cfg, params, images)
        rows = []
        for i, block in enumerate(params["blocks"]):
            x = x + taps[i]
            ent, mass = block_stats(cfg, x, block, tables)
            x = block_forward(cfg, x, block, tables)
            rows.append((jnp.sqrt(jnp.mean(x * x)), ent, mass,
                         token_cosine(x)))
        final = layer_norm(x, params["norm"])
        return jnp.sum(final * cotangent), rows

    x0 = embed(cfg, params, images)
    taps = [jnp.zeros_like(x0) for _ in range(depth)]
    (_, rows), grads = jax.value_and_grad(body, has_aux=True)(taps)
    grad_rms = [jnp.sqrt(jnp.mean(g * g)) for g in grads]
    return {
        "residual_rms": jnp.stack([r[0] for r in rows]),
        "entropy": jnp.stack([r[1] for r in rows]),
        "mass": jnp.stack([r[2] for r in rows]),
        "token_cosine": jnp.stack([r[3] for r in rows]),
        "grad_rms": jnp.stack(grad_rms),
    }


def configs(size):
    if size == "S/8@64":
        shape = dict(image_size=64, patch_size=8, width=384, heads=6,
                     mlp_dim=1536)
    else:
        shape = dict(image_size=224, patch_size=16)
    std = standard_config(**shape)
    alg = algebraic_config(**shape)
    return {
        "standard": std,
        "algebraic": alg,
        "std+radical": replace(std, attention="radical"),
        "std+radical+sink": replace(std, attention="radical", sink="learned"),
        "std+alu": replace(std, activation="alu"),
        "std+cayley2d": replace(std, rotary="cayley2d"),
        "alg_n16": replace(alg, order=16),
        "alg_temp4": replace(alg, temperature=4.0),
    }


def main():
    imgs = np.load("/dev/shm/data/in64/val_images.npy", mmap_mode="r")[:64]
    base = jnp.asarray(imgs, jnp.float32) / 127.5 - 1.0
    results = {}
    for size in ("S/8@64", "B/16@224"):
        images = base if size == "S/8@64" else jax.image.resize(
            base, (64, 224, 224, 3), "linear")
        for name, cfg in configs(size).items():
            fn = jax.jit(lambda p, x, g, cfg=cfg: run(cfg, p, x, g))
            seeds = []
            for seed in range(8):
                params = init_params(cfg, jax.random.PRNGKey(seed))
                tokens = cfg.num_patches
                g = jax.random.normal(jax.random.PRNGKey(100 + seed),
                                      (64, tokens, cfg.width))
                out = jax.device_get(fn(params, images, g))
                seeds.append({k: np.asarray(v).tolist()
                             for k, v in out.items()})
            agg = {k: np.mean([s[k] for s in seeds], 0).tolist()
                   for k in seeds[0]}
            agg["grad_ratio_first_last"] = float(
                np.mean([s["grad_rms"][0] / s["grad_rms"][-1] for s in seeds]))
            results.setdefault(size, {})[name] = agg
            print(f"{size} {name:18s} grad first/last "
                  f"{agg['grad_ratio_first_last']:.3f} final cos "
                  f"{agg['token_cosine'][-1]:.3f} entropy L1/L12 "
                  f"{agg['entropy'][0]:.3f}/{agg['entropy'][-1]:.3f} mass "
                  f"{agg['mass'][0]:.4f} rms last "
                  f"{agg['residual_rms'][-1]:.2f}", flush=True)
    # Criterion: the algebraic model propagates signal about as well as the
    # standard one (first/last gradient ratio within a factor of two, no
    # stronger collapse of the token representations).
    verdict = {}
    for size, table in results.items():
        s, a = table["standard"], table["algebraic"]
        ratio = a["grad_ratio_first_last"] / s["grad_ratio_first_last"]
        cos_std, cos_alg = s["token_cosine"][-1], a["token_cosine"][-1]
        verdict[size] = {
            "grad_ratio_rel": ratio, "cos_std": cos_std, "cos_alg": cos_alg,
            "comparable": bool(0.5 <= ratio <= 2.0
                               and cos_alg <= cos_std + 0.05),
        }
    results["criteria"] = verdict
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))
    return 0 if all(v["comparable"] for v in verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
