#!/usr/bin/env python3
"""Small-scale trials: ViT-S/8 on 64x64 ImageNet for 4 epochs.

Each round is a set of configurations run on paired seeds as single-chip
jobs:

    python experiments/small_scale.py round1 > jobs.json
    python scripts/grid.py jobs.json

Round 1 swaps each algebraic component of the first design (v1) into the
standard ViT on its own, and all of them together. Round 2 sweeps the
learning rate for both arms. Round 3 screens redesigns of the components
that lost in round 1. Round 4 tests components on the radical-attention base,
Fenchel-Young losses and the second design (v2), adds seeds for the standard
arm's best setting and a RoPE-2D control. Round 5 searches learning rate and
warmup for the final design (v3) as broadly as the standard arm had been
searched; round 6 extends the standard arm's search around its optimum and
confirms v3's best setting on two more seeds.
"""

import json
import sys

TRAIN_IMAGES = 1_271_167        # ImageNet-1k train split minus minival
BASE = {
    "data_root": "/dev/shm/data/in64",
    "image_size": 64,
    "model": {"patch_size": 8, "width": 384, "depth": 12, "heads": 6,
              "mlp_dim": 1536},
    "batch_size": 512,
    "total_images": 4 * TRAIN_IMAGES,
    "lr": 1e-3,
    "warmup_steps": 1000,
    "weight_decay": 0.05,
    "b2": 0.999,
    "grad_clip": 1.0,
    "label_smoothing": 0.1,
    "crop_scale_min": 0.35,
    "log_every": 500,
    "eval_batch": 1000,
}

# Model overrides of the algebraic designs. v1: radical attention with a
# learned sink, the ALU gate and fixed-frequency 2-D Cayley rotations. v2 and
# v3 (the final design) share the model: no sink, the radical logistic gate
# and learned mixed Cayley rotations; v3 adds the final power score.
V1 = {"attention": "radical", "sink": "learned", "activation": "alu",
      "rotary": "cayley2d"}
V2 = {"attention": "radical", "sink": "none", "activation": "rgelu",
      "rotary": "cayley2d_mixed"}
RAD = {"attention": "radical"}

# Power scores (see algebraic_vision.losses.power_score): exponent -j gives
# a = 1 - 2^-j and power_order n the link E_n. (-3, 8) is a = 7/8 with an
# E_8 link; (-6, 64) is the final a = 63/64 with an E_64 link.
POWER_7_8 = {"loss": "power", "power_alpha_exponent": -3, "power_order": 8}
POWER_63_64 = {"loss": "power", "power_alpha_exponent": -6,
               "power_order": 64}

ALG_V1 = {"arm": "algebraic", "model": V1, **POWER_7_8}
ALG_V3 = {"arm": "algebraic", "model": V2, **POWER_63_64}
STD_BEST = {"arm": "standard", "lr": 5e-4, "warmup_steps": 2500}

ROUNDS = {}
ROUNDS["round1"] = {
    "std": {"arm": "standard"},
    "alg": ALG_V1,
    "std_rad": {"arm": "standard", "model": RAD},
    "std_rad_sink": {"arm": "standard",
                     "model": {"attention": "radical", "sink": "learned"}},
    "std_alu": {"arm": "standard", "model": {"activation": "alu"}},
    "std_cay": {"arm": "standard", "model": {"rotary": "cayley2d"}},
    "std_pow": {"arm": "standard", **POWER_7_8},
    "alg_nocay": dict(ALG_V1, model=dict(V1, rotary="none")),
}
ROUNDS["round2"] = {}
for _lr, _tag in ((2.5e-4, "lr2.5e-4"), (5e-4, "lr5e-4"), (2e-3, "lr2e-3")):
    ROUNDS["round2"][f"std_{_tag}"] = {"arm": "standard", "lr": _lr}
    ROUNDS["round2"][f"alg_{_tag}"] = dict(ALG_V1, lr=_lr)
    ROUNDS["round2"][f"std_rad_{_tag}"] = {"arm": "standard", "lr": _lr,
                                           "model": RAD}
ROUNDS["round2"]["std_lr5e-4_wu2500"] = {"arm": "standard", "lr": 5e-4,
                                         "warmup_steps": 2500}
ROUNDS["round2"]["std_lr1e-3_wu2500"] = {"arm": "standard", "lr": 1e-3,
                                         "warmup_steps": 2500}
ROUNDS["round3"] = {
    "std_rgelu": {"arm": "standard", "model": {"activation": "rgelu"}},
    "std_rsilu": {"arm": "standard",
                  "model": {"activation": "rgelu", "act_beta": 1.0}},
    "std_rgelu16": {"arm": "standard",
                    "model": {"activation": "rgelu", "act_order": 16}},
    "std_alu_c2": {"arm": "standard",
                   "model": {"activation": "alu", "alu_c": 2.0}},
    "std_pow15m16": {"arm": "standard", "loss": "power",
                     "power_alpha_exponent": -4, "power_order": 16},
    "std_pow7m16c": {"arm": "standard", "loss": "power",
                     "power_alpha_exponent": -3, "power_order": 16,
                     "power_normalize": "center"},
    "std_pow15m16c": {"arm": "standard", "loss": "power",
                      "power_alpha_exponent": -4, "power_order": 16,
                      "power_normalize": "center"},
    "std_powp3": {"arm": "standard", "loss": "power",
                  "power_alpha_exponent": 3, "power_order": 8},
    "std_pow31m32": {"arm": "standard", "loss": "power",
                     "power_alpha_exponent": -5, "power_order": 32},
    "std_rad_sinkm8": {"arm": "standard",
                       "model": {"attention": "radical", "sink": "learned",
                                 "sink_init": -8.0}},
    "std_rad_n16": {"arm": "standard",
                    "model": {"attention": "radical", "order": 16}},
    "std_rad_n4": {"arm": "standard",
                   "model": {"attention": "radical", "order": 4}},
    "std_rad_t2": {"arm": "standard",
                   "model": {"attention": "radical", "temperature": 2.0}},
    "std_rad_cay": {"arm": "standard",
                    "model": {"attention": "radical", "rotary": "cayley2d"}},
}
ROUNDS["round4"] = {
    "rad_pow15m16": {"arm": "standard", "model": RAD, "loss": "power",
                     "power_alpha_exponent": -4, "power_order": 16},
    "rad_pow63m64": {"arm": "standard", "model": RAD, **POWER_63_64},
    "rad_fyr8": {"arm": "standard", "model": RAD, "loss": "fy_radical",
                 "power_order": 8},
    "rad_fyr16": {"arm": "standard", "model": RAD, "loss": "fy_radical",
                  "power_order": 16},
    "rad_fysq": {"arm": "standard", "model": RAD, "loss": "fy_squareplus"},
    "rad_ent15": {"arm": "standard", "model": RAD, "loss": "entmax15"},
    "rad_rgelu": {"arm": "standard",
                  "model": {"attention": "radical", "activation": "rgelu"}},
    "rad_caymix": {"arm": "standard",
                   "model": {"attention": "radical",
                             "rotary": "cayley2d_mixed"}},
    "algv2_fyr8": {"arm": "algebraic", "model": V2, "loss": "fy_radical",
                   "power_order": 8},
    "algv2_pow15": {"arm": "algebraic", "model": V2, "loss": "power",
                    "power_alpha_exponent": -4, "power_order": 16},
    "std_rope": dict(STD_BEST, model={"rotary": "rope2d"}),
}
ROUNDS["round4_seeds"] = {"std_best": STD_BEST}
ROUNDS["round5"] = {}
for _lr, _tag in ((7e-4, "7e-4"), (1e-3, "1e-3"), (1.5e-3, "1.5e-3"),
                  (2e-3, "2e-3")):
    for _wu in (1000, 2500):
        ROUNDS["round5"][f"v3_lr{_tag}_wu{_wu}"] = dict(ALG_V3, lr=_lr,
                                                        warmup_steps=_wu)
ROUNDS["round6"] = {
    "std_lr3.5e-4_wu2500": {"arm": "standard", "lr": 3.5e-4,
                            "warmup_steps": 2500},
    "std_lr7e-4_wu2500": {"arm": "standard", "lr": 7e-4,
                          "warmup_steps": 2500},
    "std_lr5e-4_wu4000": {"arm": "standard", "lr": 5e-4,
                          "warmup_steps": 4000},
    "v3_lr1.5e-3_wu4000": dict(ALG_V3, lr=1.5e-3, warmup_steps=4000),
}
ROUNDS["round6_confirm"] = {
    "v3_lr1.5e-3_wu2500": dict(ALG_V3, lr=1.5e-3, warmup_steps=2500),
}

SEEDS = {"round1": (0, 1, 2, 3), "round2": (0, 1), "round3": (0, 1),
         "round4": (0, 1), "round4_seeds": (0, 1, 2, 3), "round5": (0, 1),
         "round6": (0, 1), "round6_confirm": (2, 3)}
# The confirmation runs complete round 5's best setting to four seeds.
OUTPUT_DIR = {"round6_confirm": "round5"}
# Trained parameters kept for the attention statistics (seeds 0 and 2).
SAVE = {"round2": ("std_lr5e-4", "alg_lr5e-4", "std_rad_lr5e-4"),
        "round6_confirm": ("v3_lr1.5e-3_wu2500",)}


def jobs(round_name):
    out = []
    for name, change in ROUNDS[round_name].items():
        for seed in SEEDS[round_name]:
            cfg = json.loads(json.dumps(BASE))
            for key, value in change.items():
                if key == "model":
                    cfg["model"].update(value)
                else:
                    cfg[key] = value
            directory = OUTPUT_DIR.get(round_name, round_name)
            cfg["seed"] = seed
            cfg["name"] = f"small_{round_name}_{name}_s{seed}"
            cfg["output"] = (f"results/small_scale/{directory}/"
                             f"{name}_s{seed}.json")
            if seed in (0, 2) and name in SAVE.get(round_name, ()):
                cfg["save_params"] = f"/dev/shm/params/{cfg['name']}.pkl"
            out.append(cfg)
    return out


if __name__ == "__main__":
    print(json.dumps(jobs(sys.argv[1]), indent=1))
