"""Generate bounded, paired B/16 and L/16 sweep/main/smoke matrices.

Generation never launches a job. Main jobs require minival selections produced
from the complete sweep matrix; smoke jobs are synthetic and at most five steps.
"""

import argparse
import dataclasses
import json
import math
from pathlib import Path

from algebraic_vision.production import ProductionConfig, atomic_json

ARMS = ("standard", "algebraic", "algebraic_v2")
SEEDS = (42, 43, 44)
MODELS = {
    "vit-b16": dict(patch_size=16, width=768, depth=12, heads=12, mlp_dim=3072),
    "vit-l16": dict(patch_size=16, width=1024, depth=24, heads=16, mlp_dim=4096, remat=True),
}
# Equal candidate counts and token budgets; each arm has its own LR grid.
LRS = {"standard": (0.0003, 0.0005, 0.0008),
       "algebraic": (0.0008, 0.0015, 0.002),
       "algebraic_v2": (0.0003, 0.0008, 0.0015)}
WEIGHT_DECAYS = (0.03, 0.1)


def base(scale, *, stage, data_root, smoke_steps=3):
    large = scale == "vit-l16"
    batch = 512 if large else 1024
    epochs = 30 if stage == "sweep" else 300
    train_images = 1_271_167
    cfg = ProductionConfig(scale=scale, stage=stage, data_root=data_root, image_size=224,
               model=dict(MODELS[scale]), batch_size=batch, eval_batch=batch,
               total_images=epochs * train_images, train_images=train_images,
               warmup_steps=math.ceil((3 if stage == "sweep" else 5) * train_images / batch),
               log_every=100, eval_every=2500, distributed=True, require_pod=True,
               checkpoint_dir="results/checkpoints", checkpoint_every=1000,
               keep_completed_checkpoint=stage != "sweep",
               evaluate_test=stage != "sweep", diagnostics_batches=1 if stage != "sweep" else 0,
               stochastic_depth=0.2 if large else 0.0,
               randaugment_layers=2 if large else 0, mixup_alpha=0.8 if large else 0.0,
               cutmix_alpha=1.0 if large else 0.0)
    if stage == "smoke":
        cfg.total_steps, cfg.total_images, cfg.warmup_steps = smoke_steps, 0, 0
        cfg.synthetic = True
        cfg.synthetic_examples = batch
        cfg.checkpoint_dir = "/dev/shm/algebraic-vision-smoke"
        cfg.checkpoint_every, cfg.eval_every, cfg.log_every = 1, 1, 1
    return cfg


def make_job(scale, arm, seed, stage, data_root, lr, wd, candidate=None, smoke_steps=3):
    cfg = base(scale, stage=stage, data_root=data_root, smoke_steps=smoke_steps)
    cfg.arm, cfg.seed, cfg.lr, cfg.weight_decay = arm, seed, lr, wd
    if arm == "standard":
        cfg.model["attention_backend"] = "pallas_flash"
    suffix = f"_c{candidate}" if candidate is not None else ""
    cfg.name = f"{stage}_{scale}_{arm}_s{seed}{suffix}"
    cfg.output = f"results/{stage}/{scale}/{arm}_s{seed}{suffix}.json"
    cfg.checkpoint_dir = "/dev/shm/algebraic-vision-smoke" if stage == "smoke" else f"results/checkpoints/{stage}"
    cfg.validate()
    return dataclasses.asdict(cfg)


def jobs(stage, scales=tuple(MODELS), data_root="/dev/shm/data/in256", selections=None, smoke_steps=3, template=False):
    out = []
    for scale in scales:
        for seed in SEEDS:
            for arm in ARMS:
                if stage == "sweep":
                    candidates = [(lr, wd) for lr in LRS[arm] for wd in WEIGHT_DECAYS]
                elif stage == "main":
                    if template:
                        choice = {"lr": LRS[arm][0], "weight_decay": WEIGHT_DECAYS[0]}
                    elif not selections or selections.get("split") != "minival" or selections.get("synthetic"):
                        raise ValueError("main matrix requires real, complete minival sweep selections")
                    else:
                        choice = selections["choices"][scale][arm]
                    candidates = [(choice["lr"], choice["weight_decay"])]
                else:
                    candidates = [(LRS[arm][0], WEIGHT_DECAYS[0])]
                for i, (lr, wd) in enumerate(candidates):
                    job = make_job(scale, arm, seed, stage, data_root, lr, wd,
                               i if stage == "sweep" else None, smoke_steps)
                    if template:
                        job["launch_ready"] = False
                    out.append(job)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sweep", "main", "smoke"))
    parser.add_argument("--scales", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--data-root", default="/dev/shm/data/in256")
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", action="store_true", help="prepare a main matrix with launch blocked pending selection")
    args = parser.parse_args()
    selections = json.loads(args.selections.read_text()) if args.selections else None
    if args.template and args.stage != "main":
        parser.error("--template applies only to main matrices")
    if args.stage == "main" and not args.template:
        from analysis.production_select import verify_selection
        verify_selection(selections)
    matrix = jobs(args.stage, args.scales, args.data_root, selections, args.smoke_steps, args.template)
    from analysis.production_fairness import check_matrix
    check_matrix(matrix)
    atomic_json(args.out, matrix)
    print(f"Prepared {len(matrix)} jobs → {args.out}; no runs launched")


if __name__ == "__main__":
    main()
