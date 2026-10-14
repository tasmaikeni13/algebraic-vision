"""Matched three-arm pod training with a wholly algebraic device update.

The frozen legacy optimizer remains available in train.py. Here host-computed
Adam bias corrections and warmup/cosine rates are scalar inputs to the update;
no cosine, power or square-root opcode enters the algebraic training graph.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from algebraic_vision import augment, data, losses, model
from algebraic_vision.train import TrainConfig, build_model_config, decay_mask, loss_and_probs, run_eval, to_host


@dataclass
class ProductionConfig(TrainConfig):
    scale: str = "vit-b16"
    dataset: str = "imagenet1k"
    train_images: int = 1_271_167  # 10,000 minival images excluded
    stage: str = "main"
    synthetic: bool = False
    synthetic_examples: int = 64
    stochastic_depth: float = 0.0
    randaugment_layers: int = 0
    randaugment_magnitude: float = 0.3
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    decoder_workers: int = 32
    prefetch: int = 4
    resume: bool = True
    stop_after_steps: int = 0
    require_pod: bool = True
    diagnostics_batches: int = 1
    checkpoint_dir: str = "results/checkpoints"
    checkpoint_every: int = 1000
    save_params: str = ""
    keep_completed_checkpoint: bool = True
    launch_ready: bool = True

    def validate(self):
        if not self.launch_ready:
            raise ValueError("main template requires real minival selections before launch")
        self.resolved_loss()
        if self.steps() < 1 or self.batch_size < 1 or self.eval_batch < 1:
            raise ValueError("positive step count and batch sizes required")
        if self.stage == "smoke" and (self.steps() > 5 or not self.synthetic):
            raise ValueError("smoke runs are limited to 1–5 steps")
        if self.stage == "sweep" and self.evaluate_test:
            raise ValueError("sweeps must select exclusively on minival")
        if not 0 <= self.stochastic_depth < 1 or not 0 <= self.label_smoothing <= 1:
            raise ValueError("invalid regularization probability")
        if min(self.mixup_alpha, self.cutmix_alpha) < 0:
            raise ValueError("mixing strengths must be nonnegative")


def source_hash():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for directory in ("algebraic_vision", "experiments", "scripts", "analysis"):
        for path in sorted((root / directory).glob("*.py")):
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def config_hash(cfg):
    values = dataclasses.asdict(cfg)
    for name in ("output", "resume", "stop_after_steps", "log_every", "checkpoint_every"):
        values.pop(name)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def host_schedule(cfg, step):
    warmup = min(cfg.warmup_steps, max(cfg.steps() - 1, 0))
    if warmup and step < warmup:
        lr = cfg.lr * step / warmup
    else:
        progress = min(1.0, (step - warmup) / max(1, cfg.steps() - warmup))
        lr = cfg.end_lr + 0.5 * (cfg.lr - cfg.end_lr) * (1 + math.cos(math.pi * progress))
    return np.asarray([lr, 1.0 / (1.0 - cfg.b1 ** (step + 1)),
                       1.0 / (1.0 - cfg.b2 ** (step + 1))], np.float32)


def optimizer_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree.map(jnp.zeros_like, params)}


def optimizer_update(cfg, params, state, grads, scalars):
    square_norm = sum(jnp.sum(g.astype(jnp.float32) * g.astype(jnp.float32))
                      for g in jax.tree.leaves(grads))
    inverse = jax.lax.rsqrt(jnp.maximum(square_norm, 1e-30))
    clip = jnp.minimum(1.0, cfg.grad_clip * inverse)
    grads = jax.tree.map(lambda g: g * clip, grads)
    m = jax.tree.map(lambda a, g: cfg.b1 * a + (1 - cfg.b1) * g, state["m"], grads)
    v = jax.tree.map(lambda a, g: cfg.b2 * a + (1 - cfg.b2) * g * g, state["v"], grads)
    mask = decay_mask(params)
    def update(p, a, b, decay):
        variance = b * scalars[2]
        # sqrt(0) is exactly zero; rsqrt's input is kept strictly positive.
        root = variance * jax.lax.rsqrt(jnp.maximum(variance, 1e-30))
        direction = (a * scalars[1]) / (root + 1e-8)
        direction = direction + (cfg.weight_decay * p if decay else 0.0)
        return p - scalars[0] * direction
    params = jax.tree.map(update, params, m, v, mask)
    norm = square_norm * inverse
    return params, {"m": m, "v": v}, norm


def training_forward(cfg, model_cfg, params, images, key):
    if not cfg.stochastic_depth:
        return model.forward(model_cfg, params, images)
    x = model.embed(model_cfg, params, images)
    tables = model.shared_rotary_tables(model_cfg)
    keys = jax.random.split(key, model_cfg.depth)
    block_fn = lambda x, p: model.block_forward(model_cfg, x, p, tables)
    if model_cfg.remat:
        block_fn = jax.checkpoint(block_fn)
    for i, block in enumerate(params["blocks"]):
        survival = 1.0 - cfg.stochastic_depth * (i + 1) / model_cfg.depth
        update = block_fn(x, block) - x
        keep = jax.random.bernoulli(keys[i], survival, (x.shape[0], 1, 1))
        x = x + jnp.where(keep, update / survival, 0.0)
    if model_cfg.pool == "inverse_energy":
        from algebraic_vision.v2 import inverse_energy_pool
        pooled = model.layer_norm(inverse_energy_pool(x), params["norm"])
    else:
        x = model.layer_norm(x, params["norm"])
        pooled = x[:, 0] if model_cfg.pool == "cls" else jnp.mean(x, 1)
    return model.dense(pooled, params["head"], jnp.float32)


def target_loss(cfg, logits, targets):
    if cfg.resolved_loss() == "ce":
        return -jnp.mean(jnp.sum(targets * jax.nn.log_softmax(logits), -1))
    if cfg.resolved_loss() not in ("power", "v2_power"):
        raise ValueError("production supports the three selected head losses")
    from algebraic_vision.kernels import _fused_power, _power_parts
    is_v2 = cfg.arm == "algebraic_v2"
    exponent, order = (-3, 8) if is_v2 else (cfg.power_alpha_exponent, cfg.power_order)
    z = logits.astype(jnp.float32) * (2.0 if is_v2 else 1.0)
    z = losses._normalize_logits(z, cfg.power_normalize, cfg.power_eps)
    use_pallas = cfg.loss_backend == "pallas" or (cfg.loss_backend == "auto" and jax.default_backend() == "tpu")
    if use_pallas:
        values = _fused_power(z, targets, exponent, order, cfg.label_smoothing,
                              jax.default_backend() != "tpu")
    else:
        values, _ = _power_parts(z, targets, exponent=exponent, order=order,
                                 classes=z.shape[-1], smoothing=cfg.label_smoothing)
    return jnp.mean(values)


def build_steps(cfg, model_cfg, mesh, donate=True):
    rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("data"))
    _, probs = loss_and_probs(cfg)

    def local_update(params, state, x, targets, labels, scalars, key):
        key = jax.random.fold_in(key, jax.lax.axis_index("data"))
        def objective(p):
            z = training_forward(cfg, model_cfg, p, x, key)
            return target_loss(cfg, z, targets), z
        (value, z), grads = jax.value_and_grad(objective, has_aux=True)(params)
        grads = jax.lax.pmean(grads, "data")
        params, state, norm = optimizer_update(cfg, params, state, grads, scalars)
        metrics = {"loss": value, "grad_norm": norm,
                   "train_top1": jnp.mean(jnp.argmax(z, -1) == labels)}
        return params, state, jax.lax.pmean(metrics, "data")

    mapped = shard_map(local_update, mesh=mesh,
                       in_specs=(P(), P(), P("data"), P("data"), P("data"), P(), P()),
                       out_specs=(P(), P(), P()), check_rep=False)
    train = jax.jit(mapped, in_shardings=(rep, rep, shard, shard, shard, rep, rep),
                    out_shardings=(rep, rep, rep), donate_argnums=(0, 1) if donate else ())

    def local_eval(params, images, labels, valid):
        z = model.forward(model_cfg, params, augment.eval_preprocess(images, cfg.image_size))
        p = probs(z)
        pred, conf = jnp.argmax(z, -1), jnp.max(p, -1)
        nll = -jnp.log(jnp.maximum(jnp.take_along_axis(p, labels[:, None], -1)[:, 0], 1e-30))
        correct, v = (pred == labels).astype(jnp.float32), valid.astype(jnp.float32)
        bins = jnp.clip((conf * 15).astype(jnp.int32), 0, 14)
        hot = jax.nn.one_hot(bins, 15) * v[:, None]
        return jax.lax.psum({"count": jnp.sum(v), "correct": jnp.sum(correct * v),
                             "nll": jnp.sum(nll * v), "bin_count": jnp.sum(hot, 0),
                             "bin_conf": jnp.sum(hot * conf[:, None], 0),
                             "bin_acc": jnp.sum(hot * correct[:, None], 0)}, "data")
    evaluate = jax.jit(shard_map(local_eval, mesh=mesh,
                        in_specs=(P(), P("data"), P("data"), P("data")),
                        out_specs=P(), check_rep=False),
                       in_shardings=(rep, shard, shard, shard), out_shardings=rep)
    return train, evaluate, shard


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_checkpoint(path, cfg, params, state, step, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    saved = {"params": to_host(params), "opt_state": to_host(state), "step": step,
             "history": history, "config_hash": config_hash(cfg), "source_sha256": source_hash()}
    with tmp.open("wb") as stream:
        pickle.dump(saved, stream, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def restore_checkpoint(path, cfg):
    with Path(path).open("rb") as stream:
        saved = pickle.load(stream)
    if saved["config_hash"] != config_hash(cfg) or saved["source_sha256"] != source_hash():
        raise ValueError("checkpoint configuration or code differs from this run")
    if not 0 <= saved["step"] <= cfg.steps():
        raise ValueError("invalid checkpoint step")
    return saved


class SyntheticSource:
    def __init__(self, count, size, classes):
        self.count, self.image_shape, self.classes = count, (size, size, 3), classes
    def __len__(self):
        return self.count
    def load(self, indices):
        images = np.stack([np.random.default_rng(int(i)).integers(0, 256, self.image_shape,
                             dtype=np.uint8) for i in indices])
        return images, np.asarray(indices % self.classes, np.int32)
    def close(self):
        pass


def check_topology(cfg):
    devices = jax.devices()
    if cfg.require_pod and (jax.process_count() != 4 or len(devices) != 16 or
                            any("v4" not in d.device_kind.lower() for d in devices)):
        raise RuntimeError("this job requires all four hosts and 16 TPU v4 chips")
    if cfg.batch_size % len(devices) or cfg.eval_batch % len(devices):
        raise ValueError("both global batch sizes must divide over the device mesh")
    return devices


def train(cfg):
    cfg.validate()
    if cfg.distributed:
        jax.distributed.initialize()
    pi, pc = jax.process_index(), jax.process_count()
    devices = check_topology(cfg)
    mesh = Mesh(np.asarray(devices), ("data",))
    mc = build_model_config(cfg)
    rep = NamedSharding(mesh, P())
    checkpoint = Path(cfg.checkpoint_dir) / cfg.name / f"host{pi}.pkl"
    history, start = [], 0
    if cfg.resume and checkpoint.exists():
        saved = restore_checkpoint(checkpoint, cfg)
        params, state, start, history = saved["params"], saved["opt_state"], saved["step"], saved["history"]
    else:
        params = model.init_params(mc, jax.random.PRNGKey(cfg.seed))
        state = optimizer_init(params)
    if pc > 1:
        starts = multihost_utils.process_allgather(np.int32(start))
        if np.any(starts != start):
            raise ValueError("hosts have inconsistent checkpoint steps")
    params, state = jax.device_put(params, rep), jax.device_put(state, rep)
    train_step, evaluate, shard = build_steps(cfg, mc, mesh)
    from algebraic_vision.regularization import prepare_batch
    prepare = jax.jit(lambda x, y, key, mixing: prepare_batch(cfg, mc.num_classes, x, y, key, mixing),
                       in_shardings=(shard, shard, rep, rep), out_shardings=(shard, shard))

    if cfg.synthetic:
        source = SyntheticSource(max(cfg.synthetic_examples, cfg.batch_size), max(256, cfg.image_size), mc.num_classes)
        indices = np.arange(len(source))
        held = indices[:cfg.eval_batch]
        metadata = {"synthetic": True, "warning": "pipeline verification, no accuracy evidence"}
    else:
        root = Path(cfg.data_root)
        source = data.open_source(root, "train", workers=cfg.decoder_workers,
                                   max_batch=max(cfg.batch_size, cfg.eval_batch) // pc)
        indices, held = np.load(root / "train_indices.npy"), np.load(root / "minival_indices.npy")
        if len(np.intersect1d(indices, held)) or len(indices) != cfg.train_images:
            raise ValueError("training/minival leakage or incorrect training image budget")
        metadata = json.loads((root / "metadata.json").read_text()) if (root / "metadata.json").exists() else {}
        metadata["train_indices_sha256"] = hashlib.sha256(indices.tobytes()).hexdigest()
        metadata["minival_indices_sha256"] = hashlib.sha256(held.tobytes()).hexdigest()
    from algebraic_vision.prefetch import CheckedLoader
    loader = CheckedLoader(source, data.EpochSampler(indices, cfg.seed), cfg.batch_size, pi, pc,
                               start_step=start, prefetch=cfg.prefetch)
    def put(a):
        return jax.make_array_from_process_local_data(shard, a, (cfg.batch_size,) + a.shape[1:])
    timings, compile_s, hlo_path, memory = [], None, None, None
    limit = min(cfg.steps(), start + cfg.stop_after_steps) if cfg.stop_after_steps else cfg.steps()
    try:
        for step in range(start, limit):
            images, labels = loader.next()
            x, y = put(images), put(labels)
            key = jax.random.fold_in(jax.random.PRNGKey(cfg.seed + 1_000_003), step)
            # Beta samples are host-side and replay from (seed, step), including resume.
            rng = np.random.default_rng([cfg.seed, step, 991])
            choose_cut = bool(cfg.cutmix_alpha and (not cfg.mixup_alpha or rng.random() < 0.5))
            alpha = cfg.cutmix_alpha if choose_cut else cfg.mixup_alpha
            mixing = np.asarray([rng.beta(alpha, alpha) if alpha else 1.0, choose_cut], np.float32)
            x, targets = prepare(x, y, key, mixing)
            scalars = host_schedule(cfg, step)
            if compile_s is None:
                t = time.perf_counter()
                executable = train_step.lower(params, state, x, targets, y, scalars, key).compile()
                compile_s = time.perf_counter() - t
                from scripts.audit_production import scan_hlo, scan_jaxpr
                audit = scan_hlo(executable.as_text())
                nested = scan_jaxpr(jax.make_jaxpr(train_step)(params, state, x, targets, y, scalars, key))
                if cfg.arm != "standard" and (not audit["pure"] or not nested["pure"]):
                    raise RuntimeError(f"training graph is not algebraic: {audit['forbidden']}, {nested['forbidden']}")
                if pi == 0:
                    hlo_path = Path(cfg.output).with_suffix(".hlo_audit.json")
                    atomic_json(hlo_path, dict(audit, nested_programs=nested, source_sha256=source_hash(), arm=cfg.arm,
                                               scale=cfg.scale, backend=jax.default_backend()))
                    analysis = executable.memory_analysis()
                    memory = {name: int(getattr(analysis, name)) for name in (
                        "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}
                    print(f"{cfg.name}: compiled in {compile_s:.1f}s; {len(devices)} devices; {model.count_params(params):,} params", flush=True)
            t = time.perf_counter()
            params, state, metrics = train_step(params, state, x, targets, y, scalars, key)
            m = to_host(metrics)
            elapsed = time.perf_counter() - t
            timings.append(elapsed)
            if not all(np.isfinite(v).all() for v in m.values()):
                raise FloatingPointError(f"nonfinite metrics at step {step}")
            row = {"step": step + 1, **{k: float(v) for k, v in m.items()}, "lr": float(scalars[0]), "seconds": elapsed}
            history.append(row)
            if pi == 0 and ((step + 1) % cfg.log_every == 0 or step + 1 == limit):
                print(f"{cfg.name}: step {step+1}/{cfg.steps()} loss={row['loss']:.5f} images/s={cfg.batch_size/elapsed:.1f}", flush=True)
            if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
                row["minival"] = run_eval(cfg, evaluate, params, source, held, shard, mesh)
            if cfg.checkpoint_every and ((step + 1) % cfg.checkpoint_every == 0 or step + 1 == limit):
                save_checkpoint(checkpoint, cfg, params, state, step + 1, history)
                if pc > 1:
                    multihost_utils.sync_global_devices(f"checkpoint-{step+1}")
    finally:
        loader.close()
    results = {"minival": run_eval(cfg, evaluate, params, source, held, shard, mesh)}
    if cfg.evaluate_test:
        val = source if cfg.synthetic else data.open_source(Path(cfg.data_root), "val", workers=cfg.decoder_workers,
                                  max_batch=cfg.eval_batch // pc)
        results["val"] = run_eval(cfg, evaluate, params, val, np.arange(len(val))[:cfg.eval_batch] if cfg.synthetic else np.arange(len(val)), shard, mesh)
        if val is not source:
            val.close()
    diagnostics = []
    if cfg.diagnostics_batches:
        from analysis.diagnostics import distributed_diagnostics
        diagnostics = distributed_diagnostics(cfg, mc, params, source, held, mesh, shard)
    source.close()
    steady = timings[1:] or timings
    record = {"config": dataclasses.asdict(cfg), "model_config": dataclasses.asdict(mc),
              "params": model.count_params(params), "steps": limit, "complete": limit == cfg.steps(),
              "images_seen": limit * cfg.batch_size, "patch_tokens_seen": limit * cfg.batch_size * mc.num_patches,
              "nonfinite_steps": 0, "train_images_per_sec": cfg.batch_size / np.mean(steady) if steady else None,
              "compile_seconds": compile_s, "memory": memory, "devices": len(devices), "processes": pc,
              "device_kind": devices[0].device_kind, "jax_version": jax.__version__,
              "git_commit": os.environ.get("GIT_COMMIT", ""), "source_sha256": source_hash(),
              "dataset_metadata": metadata, "history": history, "results": results, "diagnostics": diagnostics,
              "checkpoint": str(checkpoint), "resumed_at_step": start}
    if pi == 0:
        atomic_json(cfg.output, record)
        if cfg.save_params:
            path = Path(cfg.save_params)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                pickle.dump(to_host(params), stream, protocol=pickle.HIGHEST_PROTOCOL)
    if pc > 1:
        multihost_utils.sync_global_devices("done")
    if limit == cfg.steps() and not cfg.keep_completed_checkpoint and checkpoint.exists():
        checkpoint.unlink()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_compilation_cache_dir", os.environ.get("JAX_COMPILATION_CACHE_DIR", "/dev/shm/algebraic-jax-cache"))
    return train(ProductionConfig(**json.loads(args.config.read_text())))


if __name__ == "__main__":
    main()
