"""Audit the compiled pod update, including AdamW and nested Pallas bodies."""

import argparse
import dataclasses
import json
import re
import os
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from algebraic_vision.production import ProductionConfig, atomic_json, build_steps, host_schedule, optimizer_init, source_hash
from algebraic_vision.train import build_model_config
from algebraic_vision.model import init_params
from experiments.production import base
from scripts.audit_purity import TRANSCENDENTAL

# Structural, collective and integer RNG instructions are necessary to express
# tensors and sharding; the nonlinear real-valued instructions are restricted.
FORBIDDEN = set(TRANSCENDENTAL) | {"sqrt", "atan", "asin", "acos", "sinh", "cosh", "expm1", "log1p", "exp", "pow"}


def scan_hlo(text):
    ops = Counter()
    for line in text.splitlines():
        match = re.search(r"=\s+(?:[a-z0-9]+\[[^\]]*\](?:\{[^}]*\})?|\([^=]*\))\s+([a-z][a-z0-9-]*)\(", line)
        if match:
            ops[match.group(1)] += 1
    bad = {name: count for name, count in ops.items() if name in FORBIDDEN}
    if not ops:
        raise ValueError("no HLO instructions parsed")
    return {"pure": not bad, "forbidden": bad, "opcodes": dict(sorted(ops.items())),
             "rsqrt": ops.get("rsqrt", 0), "opaque_custom_calls": ops.get("custom-call", 0)}


def scan_jaxpr(closed):
    ops = Counter()
    seen = set()
    def visit(obj):
        if hasattr(obj, "jaxpr"):
            visit(obj.jaxpr)
        elif hasattr(obj, "eqns"):
            if id(obj) in seen:
                return
            seen.add(id(obj))
            for eqn in obj.eqns:
                ops[eqn.primitive.name] += 1
                # Static integer powers are arithmetic expansions in XLA;
                # a dynamic real pow remains forbidden in both scans.
                if eqn.primitive.name == "integer_pow" and not isinstance(eqn.params["y"], int):
                    raise ValueError("noninteger exponent in integer_pow")
                for value in eqn.params.values():
                    visit(value)
        elif isinstance(obj, dict):
            for value in obj.values():
                visit(value)
        elif isinstance(obj, (tuple, list)):
            for value in obj:
                visit(value)
    visit(closed)
    bad = {name: count for name, count in ops.items() if name in FORBIDDEN}
    return {"pure": not bad, "forbidden": bad, "primitives": dict(sorted(ops.items())),
             "rsqrt": ops.get("rsqrt", 0),
             "scope": "all nested JAX, shard_map, custom derivative and Pallas programs; static instruction sites, not scalar operations"}


def audit(cfg, mesh):
    mc = build_model_config(cfg)
    rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("data"))
    # Abstract init avoids allocating a model just to compile it.
    abstract = jax.eval_shape(lambda key: init_params(mc, key), jax.random.PRNGKey(cfg.seed))
    params = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=rep), abstract)
    state = {"m": params, "v": params}
    shape = (cfg.batch_size, cfg.image_size, cfg.image_size, 3)
    x = jax.ShapeDtypeStruct(shape, jnp.float32, sharding=shard)
    y = jax.ShapeDtypeStruct((cfg.batch_size,), jnp.int32, sharding=shard)
    targets = jax.ShapeDtypeStruct((cfg.batch_size, mc.num_classes), jnp.float32, sharding=shard)
    scalars = jax.ShapeDtypeStruct((3,), jnp.float32, sharding=rep)
    key = jax.ShapeDtypeStruct((2,), jnp.uint32, sharding=rep)
    train, _, _ = build_steps(cfg, mc, mesh, donate=False)
    args = (params, state, x, targets, y, scalars, key)
    traced = scan_jaxpr(jax.make_jaxpr(train)(*args))
    compiled = train.lower(*args).compile()
    hlo = scan_hlo(compiled.as_text())
    return {"arm": cfg.arm, "scale": cfg.scale, "backend": jax.default_backend(),
             "devices": jax.device_count(), "processes": jax.process_count(),
             "device_kind": jax.devices()[0].device_kind, "model_config": dataclasses.asdict(mc),
             "hlo": hlo, "nested_programs": traced, "pure": hlo["pure"] and traced["pure"],
             "source_sha256": source_hash()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", nargs="+", choices=("vit-b16", "vit-l16"), default=["vit-b16", "vit-l16"])
    parser.add_argument("--arms", nargs="+", default=["algebraic", "algebraic_v2"])
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/verification/purity.json"))
    args = parser.parse_args()
    jax.config.update("jax_compilation_cache_dir", os.environ.get("JAX_COMPILATION_CACHE_DIR", "/dev/shm/algebraic-jax-cache"))
    if args.distributed:
        jax.distributed.initialize()
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    records = []
    for scale in args.scales:
        for arm in args.arms:
            cfg = base(scale, stage="smoke", data_root="", smoke_steps=1)
            cfg.arm = arm
            record = audit(cfg, mesh)
            records.append(record)
            if jax.process_index() == 0:
                print(f"{scale}/{arm}: pure={record['pure']}, HLO roots={record['hlo']['rsqrt']}, nested roots={record['nested_programs']['rsqrt']}", flush=True)
                atomic_json(args.output, {"records": records})
    if any(not r["pure"] for r in records if r["arm"] != "standard"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
