#!/usr/bin/env python3
"""Check the compiled training graph of each arm for transcendental ops.

The model forward pass, the loss and their gradient are lowered and compiled
for the current backend, and every HLO instruction is scanned. The algebraic
arm must contain none of the transcendental opcodes; the standard arm is
expected to contain exp (softmax), erf (GELU) and log (cross-entropy). The
optimizer is excluded: both arms share AdamW with a cosine schedule.

Usage: python scripts/audit_purity.py [--image-size 224 --patch-size 16]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from algebraic_vision.model import forward, init_params
from algebraic_vision.train import (TrainConfig, build_model_config,
                                    loss_and_probs)

TRANSCENDENTAL = ("exponential", "exponential-minus-one", "log",
                  "log-plus-one", "logistic", "tanh", "erf", "erf-inv",
                  "sine", "cosine", "tan", "atan2", "power", "cbrt",
                  "lgamma", "digamma", "igamma", "bessel")
OPCODE = re.compile(r"=\s*[\w\[\]{},:\s\(\)\.]*?\b([a-z][a-z0-9\-]*)\(")


def opcodes(hlo_text):
    found = {}
    for line in hlo_text.splitlines():
        line = line.strip()
        if "=" not in line or line.startswith(("HloModule", "ROOT", "//")):
            m = re.search(r"\s([a-z][a-z0-9\-]*)\(", line)
        else:
            m = re.search(r"=\s*\S+\s+([a-z][a-z0-9\-]*)\(", line)
        if m:
            found[m.group(1)] = found.get(m.group(1), 0) + 1
    return found


# Training loss of the algebraic model (power score a = 63/64, E_64 link).
ALGEBRAIC_LOSS = {"loss": "power", "power_alpha_exponent": -6,
                  "power_order": 64}


def audit(arm, image_size, patch_size, width, depth, heads, mlp_dim):
    extra = ALGEBRAIC_LOSS if arm == "algebraic" else {}
    cfg = TrainConfig(arm=arm, image_size=image_size,
                      model={"patch_size": patch_size, "width": width,
                             "depth": depth, "heads": heads,
                             "mlp_dim": mlp_dim}, **extra)
    model_cfg = build_model_config(cfg)
    train_loss, _ = loss_and_probs(cfg)
    params = init_params(model_cfg, jax.random.PRNGKey(0))
    x = jnp.zeros((8, image_size, image_size, 3), jnp.float32)
    y = jnp.zeros((8,), jnp.int32)

    def objective(p, images, labels):
        return train_loss(forward(model_cfg, p, images), labels)

    compiled = jax.jit(jax.value_and_grad(objective)).lower(
        params, x, y).compile()
    ops = opcodes(compiled.as_text())
    bad = {k: v for k, v in ops.items() if k in TRANSCENDENTAL}
    return {"arm": arm, "loss": cfg.resolved_loss(),
            "model": {k: v for k, v in vars(model_cfg).items()
                      if k in ("attention", "order", "sink", "activation",
                               "rotary")},
            "transcendental_ops": bad, "opcode_count": len(ops),
            "rsqrt": ops.get("rsqrt", 0), "sqrt": ops.get("sqrt", 0),
            "divide": ops.get("divide", 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--mlp-dim", type=int, default=3072)
    parser.add_argument("--output", type=Path,
                        default=Path("results/analysis/purity.json"))
    # results/analysis/purity_tpu_b16.json is this script's output for
    # ViT-B/16 compiled for TPU v4.
    args = parser.parse_args()
    shape = (args.image_size, args.patch_size, args.width, args.depth,
             args.heads, args.mlp_dim)
    std = audit("standard", *shape)
    alg = audit("algebraic", *shape)
    record = {"backend": jax.default_backend(),
              "device_kind": jax.devices()[0].device_kind,
              "standard": std, "algebraic": alg,
              "algebraic_pure": not alg["transcendental_ops"],
              "standard_uses_transcendentals": bool(std["transcendental_ops"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    return 0 if record["algebraic_pure"] else 1


if __name__ == "__main__":
    sys.exit(main())
