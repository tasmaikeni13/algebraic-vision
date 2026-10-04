#!/usr/bin/env python3
"""Learning-rate sweep: ViT-B/16 at 224 px, 600M patch tokens per run.

Both arms run the same three peak learning rates on seeds 42-44, and each
arm's learning rate for the main runs is chosen on minival by
analysis/select_lr.py. The standard arm's best value sat at the lower edge of
this grid, so its grid was extended to 2.5e-4, 3.5e-4 and 7e-4; the
algebraic grid was not extended.

    python experiments/sweep.py > jobs.json             # both arms
    python experiments/sweep.py extension > jobs.json   # standard arm only
    python experiments/sweep.py candidates              # the sweep's spec
    python scripts/pod_queue.py jobs.json
"""

import json
import math
import sys

TOKENS = 600_000_000
PATCHES = 196
BATCH = 1024
IMAGES = math.ceil(TOKENS / PATCHES)
STEPS = math.ceil(IMAGES / BATCH)
CANDIDATES = (5e-4, 1e-3, 2e-3)
STANDARD_EXTENSION = (2.5e-4, 3.5e-4, 7e-4)
SEEDS = (42, 43, 44)
BASE = {
    "data_root": "/dev/shm/data/in256", "image_size": 224, "model": {},
    "batch_size": BATCH, "total_images": IMAGES,
    "warmup_steps": round(0.40 * STEPS), "weight_decay": 0.05, "b2": 0.999,
    "grad_clip": 1.0, "label_smoothing": 0.1, "crop_scale_min": 0.35,
    "log_every": 100, "eval_batch": 1024,
}
ARMS = {
    "standard": {"arm": "standard"},
    "algebraic": {"arm": "algebraic", "loss": "power",
                  "power_alpha_exponent": -6, "power_order": 64},
}


def job(arm, lr, seed):
    cfg = json.loads(json.dumps(BASE))
    cfg.update(ARMS[arm])
    cfg.update(lr=lr, seed=seed, name=f"sweep_{arm}_lr{lr:g}_s{seed}",
               output=f"results/sweep/{arm}_lr{lr:g}_s{seed}.json")
    return cfg


def jobs():
    return [job(arm, lr, seed) for seed in SEEDS for lr in CANDIDATES
            for arm in ARMS]


def extension_jobs():
    return [job("standard", lr, seed) for lr in STANDARD_EXTENSION
            for seed in SEEDS]


def candidates():
    return {"learning_rates": CANDIDATES, "seeds": SEEDS, "steps": STEPS,
            "images": STEPS * BATCH, "patch_tokens": STEPS * BATCH * PATCHES,
            "base": BASE, "arms": ARMS,
            "baseline_extension": {
                "standard_learning_rates": STANDARD_EXTENSION}}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "jobs"
    out = {"jobs": jobs, "extension": extension_jobs,
           "candidates": candidates}[mode]()
    print(json.dumps(out, indent=1))
