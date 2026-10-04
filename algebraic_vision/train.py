"""Train and evaluate one ViT arm on ImageNet.

The same code runs on one chip, one host or the whole pod: parameters are
replicated, the batch is sharded over every device, and each process feeds
only its share of the global batch (see data.py). Both arms use AdamW with a
linear-warmup cosine schedule, global-norm clipping, and identical data,
augmentation and initialisation streams for a given seed.

Usage::

    python -m algebraic_vision.train --config path/to/run.json [key=value ...]
"""

import argparse
import dataclasses
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from algebraic_vision import augment, data, losses
from algebraic_vision.model import (algebraic_config, count_params, forward,
                                    init_params, standard_config)


@dataclass
class TrainConfig:
    name: str = "run"
    arm: str = "standard"              # "standard" or "algebraic"
    model: dict = field(default_factory=dict)
    # "" (the arm's own: ce or power), ce, power, fy_radical, fy_squareplus
    # or entmax15
    loss: str = ""
    label_smoothing: float = 0.1
    power_alpha_exponent: int = -6     # a = 1 - 2^-6 = 63/64
    power_order: int = 64              # E_64 link
    power_normalize: str = "none"
    power_eps: float = 1.0
    data_root: str = "/dev/shm/data/in64"
    image_size: int = 64
    crop_scale_min: float = 0.35
    batch_size: int = 512
    total_images: int = 0
    total_steps: int = 0
    lr: float = 1e-3
    warmup_steps: int = 500
    end_lr: float = 0.0
    weight_decay: float = 0.05
    b1: float = 0.9
    b2: float = 0.999
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 100
    eval_every: int = 0
    eval_batch: int = 1000
    output: str = "results/runs/run.json"
    save_params: str = ""
    checkpoint_dir: str = ""
    checkpoint_every: int = 0
    distributed: bool = False

    def resolved_loss(self):
        return self.loss or ("ce" if self.arm == "standard" else "power")

    def steps(self):
        if self.total_steps:
            return self.total_steps
        return math.ceil(self.total_images / self.batch_size)


def build_model_config(cfg):
    make = standard_config if cfg.arm == "standard" else algebraic_config
    overrides = dict(cfg.model)
    overrides.setdefault("image_size", cfg.image_size)
    return make(**overrides)


def loss_and_probs(cfg):
    """Return (training loss fn, predictive-distribution fn) for the arm."""
    kind = cfg.resolved_loss()
    if kind == "ce":
        def train_loss(logits, labels):
            return losses.cross_entropy(logits, labels, cfg.label_smoothing)

        def probs(logits):
            return jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    elif kind == "fy_squareplus":
        def train_loss(logits, labels):
            return losses.squareplus_fy(logits, labels, cfg.label_smoothing)
        probs = losses.squareplus_probabilities
    elif kind == "fy_radical":
        def train_loss(logits, labels):
            return losses.radical_fy(logits, labels, cfg.power_order,
                                     cfg.label_smoothing)

        def probs(logits):
            return losses.radical_fy_probabilities(logits, cfg.power_order)
    elif kind == "entmax15":
        def train_loss(logits, labels):
            return losses.entmax15_fy(logits, labels, cfg.label_smoothing)
        probs = losses.entmax15_probabilities
    elif kind == "power":
        def train_loss(logits, labels):
            return losses.power_score(
                logits, labels, cfg.power_alpha_exponent, cfg.power_order,
                cfg.label_smoothing, cfg.power_normalize, cfg.power_eps)

        def probs(logits):
            z = losses._normalize_logits(logits.astype(jnp.float32),
                                         cfg.power_normalize, cfg.power_eps)
            return losses.radical_probabilities(z, cfg.power_order)
    else:
        raise ValueError(f"unknown loss {kind!r}")
    return train_loss, probs


def decay_mask(params):
    def keep(path, x):
        name = "/".join(str(getattr(k, "key", k)) for k in path)
        return x.ndim >= 2 and "posemb" not in name
    return jax.tree_util.tree_map_with_path(keep, params)


def make_optimizer(cfg):
    steps = cfg.steps()
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, cfg.lr, min(cfg.warmup_steps, steps - 1), steps, cfg.end_lr)
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(schedule, cfg.b1, cfg.b2, weight_decay=cfg.weight_decay,
                    mask=decay_mask))
    return tx, schedule


def _global_norm(tree):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32)))
                        for x in jax.tree_util.tree_leaves(tree)))


def build_steps(cfg, model_cfg, tx, mesh):
    train_loss, probs = loss_and_probs(cfg)
    rep = NamedSharding(mesh, P())
    shard = NamedSharding(mesh, P("data"))
    base_key = jax.random.PRNGKey(cfg.seed + 1_000_003)

    def train_step(params, opt_state, images, labels, step):
        key = jax.random.fold_in(base_key, step)
        x = augment.train_augment(key, images, cfg.image_size,
                                  scale=(cfg.crop_scale_min, 1.0))

        def objective(p):
            logits = forward(model_cfg, p, x)
            return train_loss(logits, labels), logits

        (loss, logits), grads = jax.value_and_grad(objective,
                                                   has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        acc = jnp.mean(jnp.argmax(logits, -1) == labels)
        metrics = {"loss": loss, "grad_norm": _global_norm(grads),
                   "train_top1": acc}
        return params, opt_state, metrics

    def eval_step(params, images, labels, valid):
        x = augment.eval_preprocess(images, cfg.image_size)
        logits = forward(model_cfg, params, x)
        p = probs(logits)
        pred = jnp.argmax(logits, -1)
        conf = jnp.max(p, -1)
        p_true = jnp.take_along_axis(p, labels[:, None], -1)[:, 0]
        nll = -jnp.log(jnp.maximum(p_true, 1e-30))   # evaluation only
        correct = (pred == labels).astype(jnp.float32)
        v = valid.astype(jnp.float32)
        bins = jnp.clip((conf * 15).astype(jnp.int32), 0, 14)
        onehot = jax.nn.one_hot(bins, 15) * v[:, None]
        return {"count": jnp.sum(v), "correct": jnp.sum(correct * v),
                "nll": jnp.sum(nll * v),
                "bin_count": jnp.sum(onehot, 0),
                "bin_conf": jnp.sum(onehot * conf[:, None], 0),
                "bin_acc": jnp.sum(onehot * correct[:, None], 0)}

    train = jax.jit(train_step,
                    in_shardings=(rep, rep, shard, shard, rep),
                    out_shardings=(rep, rep, rep), donate_argnums=(0, 1))
    evaluate = jax.jit(eval_step, in_shardings=(rep, shard, shard, shard),
                       out_shardings=rep)
    return train, evaluate, shard


def to_host(tree):
    """Copy a (possibly multi-host, replicated) pytree to NumPy.

    Replicated arrays on a pod span devices of other processes, so the local
    replica is read instead of gathering the global array.
    """
    def one(x):
        if isinstance(x, jax.Array) and not x.is_fully_addressable:
            x = x.addressable_data(0)
        return np.asarray(x)
    return jax.tree_util.tree_map(one, tree)


def run_eval(cfg, evaluate, params, source, indices, shard, mesh):
    pi, pc = jax.process_index(), jax.process_count()
    totals = None
    for images, labels, valid in data.eval_batches(
            source, indices, cfg.eval_batch, pi, pc):
        gshape = (cfg.eval_batch,)
        x = jax.make_array_from_process_local_data(
            shard, images, gshape + images.shape[1:])
        y = jax.make_array_from_process_local_data(shard, labels, gshape)
        v = jax.make_array_from_process_local_data(shard, valid, gshape)
        out = evaluate(params, x, y, v)
        totals = out if totals is None else jax.tree_util.tree_map(
            jnp.add, totals, out)
    t = jax.tree_util.tree_map(lambda a: a.astype(np.float64),
                               to_host(totals))
    n = t["count"]
    ece = float(np.sum(np.abs(t["bin_acc"] - t["bin_conf"])) / n)
    return {"images": int(n), "top1": float(t["correct"] / n),
            "nll": float(t["nll"] / n), "ece": ece}


def save_tree(path, tree):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    host = to_host(tree)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(host, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def train(cfg):
    if cfg.distributed:
        jax.distributed.initialize()
    pi, pc = jax.process_index(), jax.process_count()
    devices = jax.devices()
    mesh = Mesh(np.asarray(devices), ("data",))
    model_cfg = build_model_config(cfg)
    steps = cfg.steps()
    if cfg.batch_size % len(devices):
        raise ValueError("batch size must divide across devices")

    root = Path(cfg.data_root)
    train_source = data.open_source(root, "train")
    train_indices = np.load(root / "train_indices.npy")
    minival_indices = np.load(root / "minival_indices.npy")
    sampler = data.EpochSampler(train_indices, cfg.seed)

    params = init_params(model_cfg, jax.random.PRNGKey(cfg.seed))
    tx, schedule = make_optimizer(cfg)
    opt_state = tx.init(params)
    start = 0
    history = []
    ckpt = Path(cfg.checkpoint_dir) / f"{cfg.name}.pkl" \
        if cfg.checkpoint_dir else None
    if ckpt is not None and ckpt.exists():
        with ckpt.open("rb") as f:
            saved = pickle.load(f)
        params, opt_state = saved["params"], saved["opt_state"]
        start, history = saved["step"], saved["history"]

    train_step, evaluate, shard = build_steps(cfg, model_cfg, tx, mesh)
    rep = NamedSharding(mesh, P())
    params = jax.device_put(params, rep)
    opt_state = jax.device_put(opt_state, rep)
    loader = data.TrainLoader(train_source, sampler, cfg.batch_size, pi, pc,
                              start_step=start)
    gshape = (cfg.batch_size,)

    def put(images, labels):
        x = jax.make_array_from_process_local_data(
            shard, images, gshape + images.shape[1:])
        y = jax.make_array_from_process_local_data(shard, labels, gshape)
        return x, y

    if pi == 0:
        print(f"[{cfg.name}] arm={cfg.arm} loss={cfg.resolved_loss()} "
              f"params={count_params(params):,} steps={steps} "
              f"devices={len(devices)} processes={pc}", flush=True)

    nonfinite = 0
    t_train = 0.0
    timed_steps = 0
    window = []
    t0 = time.perf_counter()
    for step in range(start, steps):
        images, labels = loader.next()
        x, y = put(images, labels)
        params, opt_state, metrics = train_step(params, opt_state, x, y,
                                                jnp.int32(step))
        window.append(metrics)
        last = step + 1 == steps
        if (step + 1) % cfg.log_every == 0 or last:
            m = to_host(window)
            jax.block_until_ready(params)
            t1 = time.perf_counter()
            if step + 1 - len(window) > start:      # exclude the compile step
                t_train += t1 - t0
                timed_steps += len(window)
            t0 = t1
            losses_w = np.array([float(v["loss"]) for v in m])
            nonfinite += int(np.sum(~np.isfinite(losses_w)))
            row = {"step": step + 1,
                   "loss": float(np.mean(losses_w)),
                   "grad_norm": float(np.mean([float(v["grad_norm"])
                                               for v in m])),
                   "train_top1": float(np.mean([float(v["train_top1"])
                                                for v in m])),
                   "lr": float(schedule(step))}
            history.append(row)
            window = []
            if pi == 0:
                rate = (timed_steps * cfg.batch_size / t_train
                        if t_train else float("nan"))
                print(f"[{cfg.name}] step {step + 1}/{steps} "
                      f"loss {row['loss']:.4f} gnorm {row['grad_norm']:.3f} "
                      f"top1 {row['train_top1']:.3f} lr {row['lr']:.2e} "
                      f"img/s {rate:,.0f}", flush=True)
            if nonfinite and not math.isfinite(row["loss"]):
                break
        if (cfg.eval_every and (step + 1) % cfg.eval_every == 0
                and not last):
            res = run_eval(cfg, evaluate, params, train_source,
                           minival_indices, shard, mesh)
            history[-1]["minival_top1"] = res["top1"]
            if pi == 0:
                print(f"[{cfg.name}] minival top1 {res['top1']:.4f}",
                      flush=True)
            t0 = time.perf_counter()
        if (ckpt is not None and cfg.checkpoint_every
                and (step + 1) % cfg.checkpoint_every == 0 and not last):
            # Every host keeps its own copy so any of them can resume.
            save_tree(ckpt, {"params": params, "opt_state": opt_state,
                             "step": step + 1, "history": history})
            if pc > 1:
                multihost_utils.sync_global_devices(f"ckpt-{step + 1}")
    loader.close()

    results = {}
    results["minival"] = run_eval(cfg, evaluate, params, train_source,
                                  minival_indices, shard, mesh)
    val_source = data.open_source(root, "val")
    results["val"] = run_eval(cfg, evaluate, params, val_source,
                              np.arange(len(val_source)), shard, mesh)
    val_source.close()
    train_source.close()

    record = {
        "config": dataclasses.asdict(cfg),
        "model_config": dataclasses.asdict(model_cfg),
        "params": count_params(params),
        "steps": steps,
        "images_seen": steps * cfg.batch_size,
        "patch_tokens_seen": steps * cfg.batch_size * model_cfg.num_patches,
        "nonfinite_steps": nonfinite,
        "train_images_per_sec": (timed_steps * cfg.batch_size / t_train
                                 if t_train else None),
        "devices": len(devices),
        "processes": pc,
        "device_kind": devices[0].device_kind,
        "jax_version": jax.__version__,
        "git_commit": os.environ.get("GIT_COMMIT", ""),
        "history": history,
        "results": results,
    }
    if pi == 0:
        out = Path(cfg.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1) + "\n")
        print(f"[{cfg.name}] minival top1 {results['minival']['top1']:.4f} "
              f"val top1 {results['val']['top1']:.4f} "
              f"nll {results['val']['nll']:.4f} -> {out}", flush=True)
        if cfg.save_params:
            save_tree(cfg.save_params, params)
    if ckpt is not None and ckpt.exists():
        ckpt.unlink()
    if pc > 1:
        multihost_utils.sync_global_devices("done")
    return record


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("overrides", nargs="*", help="key=value (JSON value)")
    args = parser.parse_args(argv)
    values = json.loads(args.config.read_text()) if args.config else {}
    for item in args.overrides:
        key, value = item.split("=", 1)
        if key.startswith("model."):
            values.setdefault("model", {})[key[6:]] = parse_value(value)
        else:
            values[key] = parse_value(value)
    cfg = TrainConfig(**values)
    train(cfg)


if __name__ == "__main__":
    sys.exit(main())
