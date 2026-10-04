#!/usr/bin/env python3
"""Benchmark attention implementations on one TPU chip.

Times forward and forward+backward of the XLA and Pallas implementations of
softmax and radical attention at the shapes used by the pilot and the main
runs, checks the kernels against the XLA reference, and writes
results/kernels/attention_bench.json.
"""

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from algebraic_vision.attention import radical_attention, softmax_attention
from algebraic_vision.kernels import fused_attention

SHAPES = {"pilot ViT-S/4@64": (128, 6, 256, 64),
          "main ViT-B/16@224": (64, 12, 196, 64)}


def timeit(fn, *args, reps=30):
    out = fn(*args)
    jax.block_until_ready(out)
    t = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / reps * 1e3


def main():
    out_path = Path(sys.argv[1] if len(sys.argv) > 1
                    else "results/kernels/attention_bench.json")
    record = {"device": jax.devices()[0].device_kind, "shapes": {}}
    for label, (b, h, t, d) in SHAPES.items():
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        q, k, v = (jax.random.normal(kk, (b, h, t, d), jnp.bfloat16)
                   for kk in keys[:3])
        g = jax.random.normal(keys[3], (b, h, t, d), jnp.bfloat16)
        sink = jnp.ones((h,), jnp.float32)
        impls = {
            "softmax/xla": lambda q, k, v, s: softmax_attention(q, k, v),
            "softmax/pallas": lambda q, k, v, s: fused_attention(
                q, k, v, "softmax"),
            "radical/xla": lambda q, k, v, s: radical_attention(
                q, k, v, s, 8, 1.0),
            "radical/pallas": lambda q, k, v, s: fused_attention(
                q, k, v, "radical", s, 8, 1.0),
        }
        rows = {}
        for name, f in impls.items():
            fwd = jax.jit(f)
            fwdbwd = jax.jit(jax.grad(
                lambda q, k, v, s: jnp.sum(f(q, k, v, s).astype(jnp.float32)
                                           * g), argnums=(0, 1, 2, 3)))
            rows[name] = {"fwd_ms": timeit(fwd, q, k, v, sink),
                          "fwd_bwd_ms": timeit(fwdbwd, q, k, v, sink)}
        for kind in ("softmax", "radical"):
            ref = jax.jit(impls[f"{kind}/xla"])(q, k, v, sink)
            got = jax.jit(impls[f"{kind}/pallas"])(q, k, v, sink)
            err = float(jnp.max(jnp.abs(ref.astype(jnp.float32)
                                        - got.astype(jnp.float32))))
            rows[f"{kind}/pallas"]["max_abs_err_vs_xla"] = err
        record["shapes"][label] = rows
        print(label)
        for name, r in rows.items():
            print(f"  {name:16s} fwd {r['fwd_ms']:7.3f} ms   fwd+bwd "
                  f"{r['fwd_bwd_ms']:7.3f} ms"
                  + (f"   err {r['max_abs_err_vs_xla']:.2e}"
                     if "max_abs_err_vs_xla" in r else ""), flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
