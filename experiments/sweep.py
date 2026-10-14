#!/usr/bin/env python3
"""Learning-rate sweep for ImageNet-1k Vision Transformer training.

Sweeps a grid of peak learning rates for standard and algebraic models across
paired random seeds to select the optimal learning rate on minival.

Usage:
    python experiments/sweep.py > jobs.json
    python experiments/sweep.py candidates > results/sweep/candidates.json
    python scripts/pod_queue.py jobs.json
    python -m analysis.select_lr
"""

import argparse
import json
import math
import sys

TRAIN_IMAGES = 1_281_167
DEFAULT_LRS = (2.5e-4, 3.5e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3)
SEEDS = (42, 43, 44)

MODELS = {
    "vit-b16": {
        "patch_size": 16,
        "width": 768,
        "depth": 12,
        "heads": 12,
        "mlp_dim": 3072,
    },
    "vit-s16": {
        "patch_size": 16,
        "width": 384,
        "depth": 12,
        "heads": 6,
        "mlp_dim": 1536,
    },
}

ARMS = {
    "standard": {"arm": "standard"},
    "algebraic": {
        "arm": "algebraic",
        "loss": "power",
        "power_alpha_exponent": -6,
        "power_order": 64,
    },
}


def build_base_config(*, epochs=30, batch_size=1024, model_name="vit-b16",
                      data_root="/dev/shm/data/in256", image_size=224, warmup_epochs=3):
    total_images = epochs * TRAIN_IMAGES
    warmup_steps = math.ceil(warmup_epochs * TRAIN_IMAGES / batch_size)
    return {
        "data_root": data_root,
        "image_size": image_size,
        "model": dict(MODELS[model_name]),
        "batch_size": batch_size,
        "total_images": total_images,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.05,
        "b1": 0.9,
        "b2": 0.999,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "crop_scale_min": 0.35,
        "log_every": 100,
        "eval_batch": batch_size,
    }


def make_job(base, arm, lr, seed):
    cfg = json.loads(json.dumps(base))
    cfg.update(ARMS[arm])
    cfg.update(
        lr=lr,
        seed=seed,
        name=f"sweep_{arm}_lr{lr:g}_s{seed}",
        output=f"results/sweep/{arm}_lr{lr:g}_s{seed}.json",
    )
    return cfg


def get_jobs(base, lrs=DEFAULT_LRS, seeds=SEEDS, arms=tuple(ARMS.keys())):
    return [
        make_job(base, arm, lr, seed)
        for seed in seeds
        for lr in lrs
        for arm in arms
    ]


def candidates_meta(base, lrs=DEFAULT_LRS, seeds=SEEDS):
    steps = math.ceil(base["total_images"] / base["batch_size"])
    return {
        "learning_rates": list(lrs),
        "seeds": list(seeds),
        "steps": steps,
        "images": base["total_images"],
        "base": base,
        "arms": ARMS,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["jobs", "candidates"], nargs="?", default="jobs",
                        help="Generate jobs list or candidates metadata")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs for screening sweep (default: 30)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Global batch size (default: 1024)")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="vit-b16",
                        help="Model architecture (default: vit-b16)")
    parser.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS,
                        help="Learning rate candidates")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="Random seeds")
    parser.add_argument("--data-root", default="/dev/shm/data/in256",
                        help="Prepared ImageNet directory")
    args = parser.parse_args()

    base = build_base_config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model,
        data_root=args.data_root,
    )

    if args.mode == "candidates":
        out = candidates_meta(base, lrs=args.lrs, seeds=args.seeds)
    else:
        out = get_jobs(base, lrs=args.lrs, seeds=args.seeds)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
