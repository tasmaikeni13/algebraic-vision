"""Fail closed on incomplete or mismatched three-arm paired matrices."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.production import ARMS, SEEDS

ARM_FIELDS = {"arm", "loss", "power_alpha_exponent", "power_order", "lr", "weight_decay",
              "name", "output", "save_params"}


def check_matrix(jobs):
    if not jobs:
        raise ValueError("empty matrix")
    grouped = defaultdict(list)
    names, outputs = set(), set()
    for cfg in jobs:
        if cfg["name"] in names or cfg["output"] in outputs:
            raise ValueError("duplicate job name or output")
        names.add(cfg["name"])
        outputs.add(cfg["output"])
        if cfg["arm"] not in ARMS or cfg["seed"] not in SEEDS:
            raise ValueError("matrix requires the three declared arms and seeds")
        key = (cfg["scale"], cfg["stage"], cfg["seed"])
        grouped[key].append(cfg)
    for key, group in grouped.items():
        counts = {arm: sum(c["arm"] == arm for c in group) for arm in ARMS}
        if len(set(counts.values())) != 1 or not all(counts.values()):
            raise ValueError(f"incomplete/unbalanced arms: {key}: {counts}")
        def shared(cfg):
            values={k:v for k,v in cfg.items() if k not in ARM_FIELDS}
            values["model"]={k:v for k,v in cfg["model"].items() if k!="attention_backend"}
            backend=cfg["model"].get("attention_backend","auto")
            if backend != ("pallas_flash" if cfg["arm"]=="standard" else "auto"):
                raise ValueError("unmatched attention backend for declared arm")
            return values
        baseline = shared(group[0])
        for cfg in group:
            actual = shared(cfg)
            if baseline != actual:
                differing = sorted(k for k in set(baseline) | set(actual) if baseline.get(k) != actual.get(k))
                raise ValueError(f"unmatched training fields: {key}: {differing}")
    for scale, stage in {(c["scale"], c["stage"]) for c in jobs}:
        seeds = {c["seed"] for c in jobs if c["scale"] == scale and c["stage"] == stage}
        if seeds != set(SEEDS):
            raise ValueError(f"missing paired seeds: {scale}/{stage}")
    return {"pass": True, "jobs": len(jobs), "paired_groups": len(grouped)}


def check_records(records, matrix):
    check_matrix(matrix)
    by_name = {r["config"]["name"]: r for r in records}
    if len(by_name) != len(records) or set(by_name) != {c["name"] for c in matrix}:
        raise ValueError("results do not cover the entire declared matrix exactly once")
    for cfg in matrix:
        r = by_name[cfg["name"]]
        if r["config"] != cfg or not r["complete"] or r["nonfinite_steps"]:
            raise ValueError(f"invalid or incomplete result: {cfg['name']}")
        if r["devices"] != 16 or r["processes"] != 4:
            raise ValueError("results were not run on the full pod")
    hashes = {r["source_sha256"] for r in records}
    metadata = {json.dumps(r["dataset_metadata"], sort_keys=True) for r in records}
    if len(hashes) != 1 or len(metadata) != 1:
        raise ValueError("code or dataset differs across paired runs")
    return {"pass": True, "records": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    result = check_matrix(matrix)
    if args.results:
        records = [json.loads((args.results / Path(c["output"]).relative_to("results")).read_text()) for c in matrix]
        result = check_records(records, matrix)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
