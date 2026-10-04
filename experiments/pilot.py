#!/usr/bin/env python3
"""Pilot: ViT-S/4 on 64x64 ImageNet for 10 epochs, both arms, seeds 42-43.

The learning rates are each arm's best from the small-scale trials:

    python experiments/pilot.py 5e-4 1.5e-3 > jobs.json
    python scripts/pod_queue.py jobs.json
"""

import json
import math
import sys

TRAIN_IMAGES = 1_271_167        # ImageNet-1k train split minus minival
EPOCHS = 10
BATCH = 512
STEPS = math.ceil(EPOCHS * TRAIN_IMAGES / BATCH)
WARMUP = round(0.40 * STEPS)    # long warmup suited both arms at small scale
BASE = {
    "data_root": "/dev/shm/data/in64", "image_size": 64,
    "model": {"patch_size": 4, "width": 384, "depth": 12, "heads": 6,
              "mlp_dim": 1536},
    "batch_size": BATCH, "total_images": EPOCHS * TRAIN_IMAGES,
    "warmup_steps": WARMUP, "weight_decay": 0.05, "b2": 0.999,
    "grad_clip": 1.0, "label_smoothing": 0.1, "crop_scale_min": 0.35,
    "log_every": 500, "eval_every": 5000, "eval_batch": 1000,
}
ARMS = {
    "standard": {"arm": "standard"},
    "algebraic": {"arm": "algebraic", "loss": "power",
                  "power_alpha_exponent": -6, "power_order": 64},
}


def jobs(std_lr, alg_lr, seeds=(42, 43)):
    out = []
    for arm, change in ARMS.items():
        for seed in seeds:
            cfg = json.loads(json.dumps(BASE))
            cfg.update(change)
            cfg["lr"] = std_lr if arm == "standard" else alg_lr
            cfg["seed"] = seed
            cfg["name"] = f"pilot_{arm}_s{seed}"
            cfg["output"] = f"results/pilot/{arm}_s{seed}.json"
            out.append(cfg)
    return out


if __name__ == "__main__":
    print(json.dumps(jobs(float(sys.argv[1]), float(sys.argv[2])), indent=1))
