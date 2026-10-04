#!/usr/bin/env python3
"""Paired-seed comparison of the runs in one directory.

Usage: python analysis/compare_runs.py results/small_scale/round1 \
           [--baseline std]

For every configuration found as <name>_s<seed>.json, prints mean validation
and minival top-1, NLL and ECE, and the paired difference to the baseline
over shared seeds with a one-sided t-test (H1: config > baseline).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load(directory):
    runs = defaultdict(dict)
    for path in sorted(Path(directory).glob("*_s*.json")):
        m = re.match(r"(.+)_s(\d+)\.json$", path.name)
        rec = json.loads(path.read_text())
        runs[m.group(1)][int(m.group(2))] = rec
    return runs


def summary(runs, baseline, metric="val"):
    rows = []
    base = runs.get(baseline, {})
    for name, by_seed in sorted(runs.items()):
        recs = list(by_seed.values())
        acc = {s: r["results"][metric]["top1"] for s, r in by_seed.items()}
        mini = [r["results"]["minival"]["top1"] for r in recs]
        nll = [r["results"][metric]["nll"] for r in recs]
        ece = [r["results"][metric]["ece"] for r in recs]
        top1 = list(acc.values())
        row = {"name": name, "seeds": len(acc),
               "top1": float(np.mean(top1)),
               "top1_sd": float(np.std(top1, ddof=1)) if len(acc) > 1 else 0.0,
               "minival": float(np.mean(mini)), "nll": float(np.mean(nll)),
               "ece": float(np.mean(ece)),
               "nonfinite": sum(r["nonfinite_steps"] for r in recs)}
        shared = sorted(set(acc) & set(base))
        if name != baseline and len(shared) >= 2:
            d = np.array([acc[s] - base[s]["results"][metric]["top1"]
                          for s in shared])
            t = stats.ttest_1samp(d, 0.0, alternative="greater")
            two_sided = stats.ttest_1samp(d, 0.0).pvalue
            row.update({"diff": float(d.mean()),
                        "diff_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                        "wins": int((d > 0).sum()), "pairs": len(d),
                        "p_greater": float(t.pvalue),
                        "p_two_sided": float(two_sided)})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", default="std")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = summary(load(args.directory), args.baseline)
    print(f"{'config':24s} {'n':>2s} {'val top1':>9s} {'sd':>6s} "
          f"{'minival':>8s} {'nll':>6s} {'ece':>6s} {'diff':>7s} {'se':>6s} "
          f"{'wins':>5s} {'p(>)':>8s}")
    for r in sorted(rows, key=lambda r: -r["top1"]):
        extra = (f"{100 * r['diff']:+7.2f} {100 * r['diff_se']:6.2f} "
                 f"{r['wins']}/{r['pairs']:<3d} {r['p_greater']:8.2g}"
                 if "diff" in r else "")
        print(f"{r['name']:24s} {r['seeds']:2d} {100 * r['top1']:9.2f} "
              f"{100 * r['top1_sd']:6.2f} {100 * r['minival']:8.2f} "
              f"{r['nll']:6.3f} {r['ece']:6.3f} {extra}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
