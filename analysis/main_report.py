#!/usr/bin/env python3
"""Summarise the main runs and test the criterion fixed before they ran.

Criterion: algebraic top-1 above standard top-1 on every seed, a positive
mean paired difference, and a one-sided paired t-test p < 0.05, all on the
validation set. If every seed favours the algebraic model but p >= 0.05,
the plan was to add seeds 45 and 46. Writes results/main/summary.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def load(arm):
    out = {}
    for path in sorted((ROOT / "results/main").glob(f"{arm}_s*.json")):
        rec = json.loads(path.read_text())
        out[rec["config"]["seed"]] = rec
    return out


def compare_control(alg, control):
    """Algebraic model against the RoPE-2D control (not part of the
    criterion; the control uses the standard model's learning rate)."""
    seeds = sorted(set(alg) & set(control))
    if len(seeds) < 2:
        return None
    top1 = [control[s]["results"]["val"]["top1"] for s in seeds]
    d = np.array([alg[s]["results"]["val"]["top1"] - c
                  for s, c in zip(seeds, top1)])
    se = d.std(ddof=1) / np.sqrt(len(d))
    tcrit = stats.t.ppf(0.975, len(d) - 1)
    return {"seeds": seeds, "val": top1, "val_mean": float(np.mean(top1)),
            "algebraic_minus_control": d.tolist(),
            "diff_mean": float(d.mean()),
            "diff_ci95": [float(d.mean() - tcrit * se),
                          float(d.mean() + tcrit * se)],
            "p_one_sided": float(stats.ttest_1samp(
                d, 0.0, alternative="greater").pvalue)}


def main():
    std, alg = load("standard"), load("algebraic")
    seeds = sorted(set(std) & set(alg))
    summary = {"seeds": seeds, "per_seed": {}, "metrics": {}}
    for s in seeds:
        summary["per_seed"][s] = {
            arm: {"val": rec[s]["results"]["val"]["top1"],
                  "minival": rec[s]["results"]["minival"]["top1"],
                  "nll": rec[s]["results"]["val"]["nll"],
                  "ece": rec[s]["results"]["val"]["ece"],
                  "images_per_sec": rec[s]["train_images_per_sec"],
                  "nonfinite": rec[s]["nonfinite_steps"],
                  "patch_tokens": rec[s]["patch_tokens_seen"],
                  "lr": rec[s]["config"]["lr"]}
            for arm, rec in (("standard", std), ("algebraic", alg))}
    for metric in ("val", "minival", "nll", "ece", "images_per_sec"):
        a = np.array([summary["per_seed"][s]["algebraic"][metric]
                      for s in seeds])
        b = np.array([summary["per_seed"][s]["standard"][metric]
                      for s in seeds])
        d = a - b
        entry = {"algebraic_mean": float(a.mean()),
                 "standard_mean": float(b.mean()),
                 "diff_mean": float(d.mean()),
                 "diff_per_seed": d.tolist()}
        if len(seeds) > 1:
            entry["diff_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
            entry["algebraic_se"] = float(a.std(ddof=1) / np.sqrt(len(a)))
            entry["standard_se"] = float(b.std(ddof=1) / np.sqrt(len(b)))
            tcrit = stats.t.ppf(0.975, len(d) - 1)
            entry["diff_ci95"] = [float(d.mean() - tcrit * entry["diff_se"]),
                                  float(d.mean() + tcrit * entry["diff_se"])]
            if metric in ("val", "minival"):
                entry["p_one_sided"] = float(stats.ttest_1samp(
                    d, 0.0, alternative="greater").pvalue)
        summary["metrics"][metric] = entry
    val = summary["metrics"].get("val", {})
    wins = all(x > 0 for x in val.get("diff_per_seed", [0]))
    criterion = {
        "all_seeds_algebraic_better": bool(wins and seeds),
        "mean_diff_positive": bool(val.get("diff_mean", 0) > 0),
        "p_lt_0.05": bool(val.get("p_one_sided", 1.0) < 0.05),
    }
    criterion["pass"] = all(criterion.values()) and len(seeds) >= 3
    criterion["escalate_seeds_45_46"] = bool(
        criterion["all_seeds_algebraic_better"]
        and criterion["mean_diff_positive"]
        and not criterion["p_lt_0.05"])
    summary["criterion"] = criterion
    control = compare_control(alg, load("standard_rope2d"))
    if control:
        summary["control_rope2d"] = control
    (ROOT / "results/main/summary.json").write_text(
        json.dumps(summary, indent=1) + "\n")
    for s in seeds:
        std = summary["per_seed"][s]["standard"]["val"]
        alg = summary["per_seed"][s]["algebraic"]["val"]
        print(f"seed {s}: standard {100 * std:.2f}  algebraic {100 * alg:.2f}"
              f"  diff {100 * (alg - std):+.2f}")
    if val:
        ci = ""
        if "p_one_sided" in val:
            lo, hi = val["diff_ci95"]
            ci = (f" (95% CI {100 * lo:+.2f}...{100 * hi:+.2f}, "
                  f"p={val['p_one_sided']:.2g})")
        print(f"mean val: standard {100 * val['standard_mean']:.2f} "
              f"algebraic {100 * val['algebraic_mean']:.2f} "
              f"diff {100 * val['diff_mean']:+.2f}" + ci)
    print("criterion:", summary["criterion"])
    if control:
        lo, hi = control["diff_ci95"]
        print(f"RoPE-2D control: {100 * control['val_mean']:.2f}; algebraic "
              f"ahead by {100 * control['diff_mean']:+.2f} (95% CI "
              f"{100 * lo:+.2f}...{100 * hi:+.2f}, "
              f"p={control['p_one_sided']:.2g})")


if __name__ == "__main__":
    main()
