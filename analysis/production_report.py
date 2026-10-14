"""Three-arm paired statistics and exportable post-run figures."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from algebraic_vision.production import atomic_json
from analysis.production_fairness import check_records
from experiments.production import ARMS, SEEDS


def paired(a, b, higher_is_better=True):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    radius = float(stats.t.ppf(0.975, len(d) - 1)) * se
    if se == 0:
        improved = mean > 0 if higher_is_better else mean < 0
        p = 0.0 if improved else (1.0 if mean else 0.5)
    else:
        p = float(stats.ttest_1samp(d, 0.0, alternative="greater" if higher_is_better else "less").pvalue)
    return {"differences": d.tolist(), "mean_difference": mean, "ci95": [mean-radius, mean+radius],
             "p_one_sided": p, "alternative": "greater" if higher_is_better else "less", "n": len(d)}


def report(matrix, records, output):
    check_records(records, matrix)
    if any("val" not in r["results"] for r in records):
        raise ValueError("main reporting requires validation evaluation for every run")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scales = sorted({r["config"]["scale"] for r in records})
    summary = {"synthetic": any(r["config"]["synthetic"] for r in records),
               "seeds": list(SEEDS), "scales": {}, "source_sha256": records[0]["source_sha256"],
               "statistical_note": "Three paired seeds; confidence intervals and tests are descriptive, unadjusted for multiple comparisons."}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight", "pdf.fonttype": 42})
    for scale in scales:
        by_arm = {arm: sorted([r for r in records if r["config"]["scale"] == scale and r["config"]["arm"] == arm], key=lambda r:r["config"]["seed"]) for arm in ARMS}
        entry = {"arms": {}, "comparisons": {}}
        for arm, group in by_arm.items():
            entry["arms"][arm] = {metric: [r["results"]["val"][metric] for r in group] for metric in ("top1", "nll", "ece")}
            entry["arms"][arm]["images_per_second"] = [r["train_images_per_sec"] for r in group]
        for b, a in itertools.combinations(ARMS, 2):
            entry["comparisons"][f"{a}_minus_{b}"] = {metric: paired(entry["arms"][a][metric], entry["arms"][b][metric], metric == "top1") for metric in ("top1", "nll", "ece")}
        summary["scales"][scale] = entry
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, metric in zip(axes, ("top1", "nll", "ece")):
            for i, arm in enumerate(ARMS):
                values = np.asarray(entry["arms"][arm][metric])
                ax.plot(np.full(3, i), values, "o", alpha=0.7)
                mean, radius = float(values.mean()), stats.t.ppf(0.975, 2) * values.std(ddof=1) / np.sqrt(3)
                ax.errorbar(i, mean, yerr=radius, fmt="s", color="black", capsize=4)
            ax.set_xticks(range(3), ["Standard", "v1", "v2"])
            ax.set_ylabel(metric.upper())
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{scale}: paired seeds" + (" (synthetic smoke)" if summary["synthetic"] else ""))
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"{scale}_metrics.{suffix}", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for arm, group in by_arm.items():
            blocks = np.asarray([[d["attention_entropy"] for d in r["diagnostics"]] for r in group])
            scores = np.asarray([[d["mean_example_median_abs_score"] for d in r["diagnostics"]] for r in group])
            if blocks.shape[1]:
                axes[0].plot(np.arange(blocks.shape[1])+1, blocks.mean(0), label=arm)
                axes[1].plot(np.arange(scores.shape[1])+1, scores.mean(0), label=arm)
        for ax, ylabel in zip(axes, ("Attention entropy (nats)", "Mean example median |score|")):
            ax.set_xlabel("Block")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        axes[0].legend()
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"{scale}_attention.{suffix}", dpi=180)
        plt.close(fig)
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/main/report"))
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    records = [json.loads(Path(c["output"]).read_text()) for c in matrix]
    report(matrix, records, args.output)
    print(f"Report and PDF/PNG figures → {args.output}")


if __name__ == "__main__":
    main()
