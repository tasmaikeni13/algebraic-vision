#!/usr/bin/env python3
"""Check that paired standard/algebraic runs differ only where they should.

Runs are paired by seed and, in a learning-rate grid (a directory with
several runs per arm and seed), also by learning rate. Within a pair the
training configurations must agree on every field except the arm-defining
ones (arm, loss and its parameters, run name, output paths) and, outside a
grid, each arm's own selected learning rate. The model configurations may
differ only in the algebraic components, and both runs must have seen the
same number of steps, images and patch tokens on the same number of devices.

Usage: python analysis/fairness_check.py results/sweep [results/main ...]
"""

import json
import sys
from collections import Counter
from pathlib import Path

ARM_FIELDS = {"arm", "loss", "power_alpha_exponent", "power_order",
              "power_normalize", "power_eps", "name", "output", "save_params"}
MODEL_FIELDS = {"attention", "order", "temperature", "sink", "sink_init",
                "activation", "alu_c", "act_beta", "act_order", "rotary"}


def load_runs(directory):
    runs = []
    for path in sorted(Path(directory).glob("*.json")):
        rec = json.loads(path.read_text())
        if not isinstance(rec, dict) or "config" not in rec:
            continue                       # job lists, summaries
        if rec["config"].get("model", {}).get("rotary") == "rope2d":
            continue                       # controls are not part of a pair
        runs.append(rec)
    return runs


def pairs(runs):
    per_seed = Counter((r["config"]["seed"], r["config"]["arm"])
                       for r in runs)
    grid = any(count > 1 for count in per_seed.values())
    grouped = {}
    for rec in runs:
        cfg = rec["config"]
        key = (cfg["seed"], cfg["lr"] if grid else None)
        grouped.setdefault(key, {})[cfg["arm"]] = rec
    return grid, {k: v for k, v in grouped.items() if len(v) == 2}


def compare(s, a, allowed):
    problems = []
    for field in sorted(set(s["config"]) | set(a["config"])):
        if field in allowed or field == "model":
            continue
        if s["config"].get(field) != a["config"].get(field):
            problems.append(f"config.{field} {s['config'].get(field)!r} != "
                            f"{a['config'].get(field)!r}")
    sm, am = s["model_config"], a["model_config"]
    for field in sorted(set(sm) | set(am)):
        if field not in MODEL_FIELDS and sm.get(field) != am.get(field):
            problems.append(f"model.{field} {sm.get(field)!r} != "
                            f"{am.get(field)!r}")
    for field in ("steps", "images_seen", "patch_tokens_seen", "devices"):
        if s[field] != a[field]:
            problems.append(f"{field} {s[field]} != {a[field]}")
    return problems


def main():
    problems, checked = [], 0
    for directory in sys.argv[1:]:
        grid, paired = pairs(load_runs(directory))
        allowed = ARM_FIELDS if grid else ARM_FIELDS | {"lr"}
        for key, pair in sorted(paired.items(), key=str):
            for p in compare(pair["standard"], pair["algebraic"], allowed):
                problems.append(f"{directory} {key}: {p}")
            checked += 1
    print(f"checked {checked} standard/algebraic pairs")
    for p in problems:
        print("MISMATCH", p)
    print("PASS" if not problems else "FAIL")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
