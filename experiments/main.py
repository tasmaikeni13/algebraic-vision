#!/usr/bin/env python3
"""Main training runs on ImageNet-1k.

Standard full Vision Transformer training (300 epochs) comparing standard
and algebraic models under matched data, architecture, optimizer, and budget.

Usage:
    python experiments/main.py [STD_LR] [ALG_LR] [--epochs 300] [--controls] > jobs.json
    python scripts/pod_queue.py jobs.json
"""

import argparse
import json
import math
import sys

TRAIN_IMAGES = 1_281_167  # ImageNet-1k train split
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

CONTROLS = {
    "standard_rope2d": {"arm": "standard", "model": {"rotary": "rope2d"}},
}


def create_jobs(std_lr, alg_lr, *, epochs=300, batch_size=1024, model_name="vit-b16",
                seeds=(42, 43, 44), controls=False, data_root="/dev/shm/data/in256",
                image_size=224, warmup_epochs=5):
    total_images = epochs * TRAIN_IMAGES
    steps = math.ceil(total_images / batch_size)
    warmup_steps = math.ceil(warmup_epochs * TRAIN_IMAGES / batch_size)

    base = {
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
        "eval_every": 2500,
        "eval_batch": batch_size,
        "checkpoint_dir": "/dev/shm/ckpt",
        "checkpoint_every": 5000,
    }

    arms = CONTROLS if controls else ARMS
    jobs = []
    for seed in seeds:
        for arm_name, change in arms.items():
            cfg = json.loads(json.dumps(base))
            cfg.update(change)
            lr = alg_lr if arm_name == "algebraic" else std_lr
            cfg.update(
                lr=lr,
                seed=seed,
                name=f"main_{arm_name}_s{seed}",
                output=f"results/main/{arm_name}_s{seed}.json",
            )
            if seed == 42 and not controls:
                cfg["save_params"] = f"/dev/shm/params/main_{arm_name}_s42.pkl"
            jobs.append(cfg)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("std_lr", type=float, nargs="?", default=5e-4,
                        help="Peak learning rate for standard arm (default: 5e-4)")
    parser.add_argument("alg_lr", type=float, nargs="?", default=2e-3,
                        help="Peak learning rate for algebraic arm (default: 2e-3)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of ImageNet-1k epochs (default: 300)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Global batch size (default: 1024)")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="vit-b16",
                        help="Model architecture (default: vit-b16)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Random seeds (default: 42 43 44)")
    parser.add_argument("--data-root", default="/dev/shm/data/in256",
                        help="Prepared ImageNet directory")
    parser.add_argument("--controls", action="store_true",
                        help="Generate RoPE-2D control runs instead of standard/algebraic pair")
    args = parser.parse_args()

    jobs = create_jobs(
        args.std_lr,
        args.alg_lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model,
        seeds=tuple(args.seeds),
        controls=args.controls,
        data_root=args.data_root,
    )
    print(json.dumps(jobs, indent=2))


if __name__ == "__main__":
    main()
