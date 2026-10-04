#!/usr/bin/env python3
"""Choose each arm's learning rate for the main runs from the sweep.

Rule (fixed before the sweep ran): per arm, the learning rate with the
highest mean minival top-1 over seeds 42-44, among candidates with all three
seeds complete and no non-finite training step; ties go to the lower
learning rate. Validation accuracy is reported but not used. Writes
results/sweep/selection.json.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "results/sweep"


def main():
    sweep = json.loads((SWEEP_DIR / "candidates.json").read_text())
    extension = sweep.get("baseline_extension", {})
    table, selection = {}, {}
    for arm in ("standard", "algebraic"):
        lrs = list(sweep["learning_rates"])
        if arm == "standard":
            lrs = sorted(lrs + extension.get("standard_learning_rates", []))
        rows = []
        for lr in lrs:
            recs = []
            for seed in sweep["seeds"]:
                path = SWEEP_DIR / f"{arm}_lr{lr:g}_s{seed}.json"
                if path.exists():
                    recs.append(json.loads(path.read_text()))
            complete = len(recs) == len(sweep["seeds"])
            finite = all(r["nonfinite_steps"] == 0 for r in recs)
            row = {"lr": lr, "seeds": len(recs),
                   "eligible": complete and finite}
            if recs:
                for split in ("minival", "val"):
                    vals = [r["results"][split]["top1"] for r in recs]
                    row[split] = float(np.mean(vals))
                    row[f"{split}_per_seed"] = vals
                row["val_nll"] = float(np.mean([r["results"]["val"]["nll"]
                                                for r in recs]))
                row["images_per_sec"] = float(np.mean(
                    [r["train_images_per_sec"] for r in recs]))
            rows.append(row)
        table[arm] = rows
        eligible = [r for r in rows if r["eligible"]]
        if eligible:
            best = max(eligible, key=lambda r: (r["minival"], -r["lr"]))
            selection[arm] = {"lr": best["lr"], "minival": best["minival"],
                              "val": best["val"]}
    complete = all(r["seeds"] == len(sweep["seeds"])
                   for rows in table.values() for r in rows)
    out = {"complete": complete, "table": table, "selection": selection}
    if complete and len(selection) == 2:
        out["advisory_algebraic_ahead_on_minival"] = (
            selection["algebraic"]["minival"]
            > selection["standard"]["minival"])
    (SWEEP_DIR / "selection.json").write_text(json.dumps(out, indent=1))
    for arm, rows in table.items():
        for r in rows:
            if r["seeds"]:
                print(f"{arm:9s} lr {r['lr']:<7g} seeds {r['seeds']} "
                      f"minival {100 * r['minival']:6.2f} "
                      f"val {100 * r['val']:6.2f} "
                      f"nll {r['val_nll']:.3f} eligible {r['eligible']}")
    print("selection:", json.dumps(selection), "complete:", complete)


if __name__ == "__main__":
    main()
