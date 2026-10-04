#!/usr/bin/env python3
"""Main runs: ViT-B/16 at 224 px, 2.5B patch tokens, seeds 42-44.

The learning rates are each arm's selection from the sweep
(results/sweep/selection.json); everything else is fixed here. The control
is a standard ViT with sine/cosine 2-D rotary positions trained with the
standard arm's hyperparameters, which separates the effect of relative
positions from that of the algebraic components.

    python experiments/main.py STD_LR ALG_LR [--controls] > jobs.json
    python scripts/pod_queue.py jobs.json
"""

import json
import math
import sys

TOKENS = 2_500_000_000
PATCHES = 196
BATCH = 1024
IMAGES = math.ceil(TOKENS / PATCHES)
STEPS = math.ceil(IMAGES / BATCH)
SEEDS = (42, 43, 44)
BASE = {
    "data_root": "/dev/shm/data/in256", "image_size": 224, "model": {},
    "batch_size": BATCH, "total_images": IMAGES,
    "warmup_steps": round(0.40 * STEPS), "weight_decay": 0.05, "b2": 0.999,
    "grad_clip": 1.0, "label_smoothing": 0.1, "crop_scale_min": 0.35,
    "log_every": 100, "eval_every": 2500, "eval_batch": 1024,
    "checkpoint_dir": "/dev/shm/ckpt", "checkpoint_every": 4000,
}
ARMS = {
    "standard": {"arm": "standard"},
    "algebraic": {"arm": "algebraic", "loss": "power",
                  "power_alpha_exponent": -6, "power_order": 64},
}
CONTROLS = {
    "standard_rope2d": {"arm": "standard", "model": {"rotary": "rope2d"}},
}


def jobs(std_lr, alg_lr, controls=False):
    out = []
    arms = CONTROLS if controls else ARMS
    for seed in SEEDS:
        for arm, change in arms.items():
            cfg = json.loads(json.dumps(BASE))
            cfg.update(change)
            lr = alg_lr if arm == "algebraic" else std_lr
            cfg.update(lr=lr, seed=seed, name=f"main_{arm}_s{seed}",
                       output=f"results/main/{arm}_s{seed}.json")
            if seed == 42 and not controls:
                # Kept for the attention statistics of the trained models.
                cfg["save_params"] = f"/dev/shm/params/main_{arm}_s42.pkl"
            out.append(cfg)
    return out


if __name__ == "__main__":
    print(json.dumps(jobs(float(sys.argv[1]), float(sys.argv[2]),
                          controls="--controls" in sys.argv), indent=1))
