"""Choose one LR/WD pair per scale/arm using paired minival means only."""

import argparse
import json
import hashlib
from collections import defaultdict
from pathlib import Path

from algebraic_vision.production import atomic_json
from analysis.production_fairness import check_records
from experiments.production import ARMS, SEEDS


def select(matrix, records, allow_synthetic=False):
    check_records(records, matrix)
    if any(c["stage"] not in ("sweep", "smoke") or
           (c["evaluate_test"] and not (allow_synthetic and c["stage"]=="smoke" and c["synthetic"])) for c in matrix):
        raise ValueError("selection requires held-out-only sweep records")
    synthetic = any(r["config"]["synthetic"] for r in records)
    if synthetic and not allow_synthetic:
        raise ValueError("synthetic smoke results cannot select production hyperparameters")
    grouped = defaultdict(list)
    for r in records:
        c = r["config"]
        grouped[(c["scale"], c["arm"], c["lr"], c["weight_decay"])].append(r)
    scores = defaultdict(list)
    for (scale, arm, lr, wd), group in grouped.items():
        if {r["config"]["seed"] for r in group} != set(SEEDS):
            raise ValueError("missing paired seeds for a candidate")
        score = sum(r["results"]["minival"]["top1"] for r in group) / len(SEEDS)
        scores[(scale, arm)].append({"lr": lr, "weight_decay": wd, "minival_top1_mean": score,
                                    "seeds": list(SEEDS), "run_names": [r["config"]["name"] for r in group]})
    choices = defaultdict(dict)
    for (scale, arm), candidates in scores.items():
        # Deterministic ties: higher minival Top-1, then lower LR and decay.
        choices[scale][arm] = sorted(candidates, key=lambda v: (-v["minival_top1_mean"], v["lr"], v["weight_decay"]))[0]
    return {"split": "minival", "synthetic": synthetic, "choices": dict(choices),
             "matrix": matrix,
             "record_sha256": {r["config"]["name"]: hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in records},
             "source_sha256": records[0]["source_sha256"], "selection_metric": "mean Top-1 over seeds 42/43/44",
             "candidates": {f"{scale}/{arm}": values for (scale, arm), values in scores.items()}}


def verify_selection(selection):
    if not selection or selection.get("synthetic") or "matrix" not in selection:
        raise ValueError("real complete minival selection evidence required")
    records = [json.loads(Path(c["output"]).read_text()) for c in selection["matrix"]]
    if select(selection["matrix"], records) != selection:
        raise ValueError("selected hyperparameters or sweep evidence were changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/selections.json"))
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    records = [json.loads(Path(c["output"]).read_text()) for c in matrix]
    atomic_json(args.output, select(matrix, records))
    print(f"Selected on minival → {args.output}")


if __name__ == "__main__":
    main()
