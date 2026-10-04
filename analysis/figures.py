#!/usr/bin/env python3
"""Figures for the paper, drawn from the JSON records in results/.

Usage: python analysis/figures.py [OUTDIR]   (default paper/figures)
"""

import glob
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper/figures"

# Validated categorical slots (light surface): standard, algebraic, control.
STD, ALG, CTRL = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def load(path):
    return json.loads(Path(path).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def label_ends(ax, ends, min_gap_pt=8.0):
    """Label line ends with their values, nudged apart so they never touch.

    ends: (x, y, text) per line; the text is placed right of (x, y).
    """
    ends = sorted(ends, key=lambda e: e[1])
    y0, y1 = ax.get_ylim()
    pt_per_unit = ax.get_window_extent().height * 72 / ax.figure.dpi / (
        y1 - y0)
    placed = []
    for x, y, text in ends:
        pos = y * pt_per_unit
        if placed and pos - placed[-1] < min_gap_pt:
            pos = placed[-1] + min_gap_pt
        placed.append(pos)
        ax.annotate(text, (x, y), xytext=(4, pos - y * pt_per_unit),
                    textcoords="offset points", color=INK2, fontsize=7.5,
                    va="center")


def fig_kernels():
    """E_n(s) against e^s (log scale) and the gates against GELU."""
    s = np.linspace(-12, 12, 481)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ax = axes[0]
    ax.semilogy(s, np.exp(s), color=STD, label="$e^s$ (softmax)")
    for n, alpha in ((8, 1.0), (16, 0.55)):
        e = (s / n + np.sqrt(1 + (s / n) ** 2)) ** n
        ax.semilogy(s, e, color=ALG, alpha=alpha,
                    label=f"$E_{{{n}}}(s)$", linestyle="-" if n == 8 else "--")
    ax.set_xlabel("score $s$")
    ax.set_title("Attention kernel")
    ax.legend(loc="upper left")
    ax = axes[1]
    x = np.linspace(-4, 4, 401)
    from scipy.special import erf
    gelu = x * 0.5 * (1 + erf(x / np.sqrt(2)))
    e = (-1.702 * x / 8 + np.sqrt(1 + (1.702 * x / 8) ** 2)) ** 8
    rgelu = x / (1 + e)
    ax.plot(x, gelu, color=STD, label="GELU")
    ax.plot(x, rgelu, color=ALG, linestyle="--", label="radical gate")
    ax.set_xlabel("pre-activation $x$")
    ax.set_title("MLP gate")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "kernels")


def fig_stress():
    """Entropy and saturated rows of attention as scores are scaled."""
    d = load(ROOT / "results/analysis/stress.json")["gaussian"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    series = (("softmax", "softmax", STD, "-"),
              ("radical_n8", "$E_8$", ALG, "-"),
              ("radical_n16", "$E_{16}$", ALG, "--"))
    for key, name, color, ls in series:
        rows = d[key]
        a = [r["alpha"] for r in rows]
        axes[0].plot(a, [r["entropy"] for r in rows], marker="o",
                     color=color, linestyle=ls, label=name)
        axes[1].plot(a, [100 * r["saturated_rows"] for r in rows],
                     marker="o", color=color, linestyle=ls, label=name)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"score scale $\alpha$")
    axes[0].set_ylabel("entropy / log N")
    axes[0].set_title("Attention entropy")
    axes[1].set_ylabel("rows with $\\|J\\|_F < 10^{-4}$ (%)")
    axes[1].set_title("Saturated rows")
    axes[1].annotate("$E_8$, $E_{16}$: below 2%", (2 ** 4, 4), color=INK2,
                     fontsize=7.5)
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "stress")


# Small-scale settings with 25% warmup (2500 of 9930 steps), one entry per
# learning rate with all of its seeds.
SMALL_LR = {
    "standard": {3.5e-4: "round6/std_lr3.5e-4_wu2500",
                 5e-4: "round4_seeds/std_best",
                 7e-4: "round6/std_lr7e-4_wu2500",
                 1e-3: "round2/std_lr1e-3_wu2500"},
    "algebraic": {7e-4: "round5/v3_lr7e-4_wu2500",
                  1e-3: "round5/v3_lr1e-3_wu2500",
                  1.5e-3: "round5/v3_lr1.5e-3_wu2500",
                  2e-3: "round5/v3_lr2e-3_wu2500"},
}


def seed_runs(prefix):
    """Records RESULTS/<prefix>_s<seed>.json, matching the name exactly."""
    path = ROOT / "results" / prefix
    pattern = re.compile(rf"{re.escape(path.name)}_s\d+\.json")
    return [load(p) for p in sorted(path.parent.glob("*.json"))
            if pattern.fullmatch(p.name)]


def _lr_table(paths, pattern):
    rows = {}
    for path in paths:
        m = re.match(pattern, Path(path).name)
        if not m:
            continue
        rec = load(path)
        rows.setdefault(float(m.group(1)), []).append(
            rec["results"]["val"]["top1"])
    lrs = sorted(rows)
    return lrs, [100 * np.mean(rows[k]) for k in lrs]


def _lr_axis(ax, lrs):
    """Log learning-rate axis labelled in units of 1e-4 at the grid points."""
    ax.set_xscale("log")
    ax.set_xticks(lrs)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x * 1e4:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(r"peak learning rate ($\times10^{-4}$)")


def fig_lr():
    """Validation top-1 against peak learning rate, both scales."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ax = axes[0]
    grid = set()
    for arm, color in (("standard", STD), ("algebraic", ALG)):
        lrs, vals = [], []
        for lr, prefix in sorted(SMALL_LR[arm].items()):
            rs = seed_runs(f"small_scale/{prefix}")
            if rs:
                lrs.append(lr)
                vals.append(100 * np.mean([r["results"]["val"]["top1"]
                                           for r in rs]))
        ax.plot(lrs, vals, marker="o", color=color, label=arm)
        grid.update(lrs)
    _lr_axis(ax, sorted(grid))
    ax.set_ylabel("validation top-1 (%)")
    ax.set_title("ViT-S/8, 64 px, 4 epochs")
    ax.legend(loc="lower right")
    ax = axes[1]
    grid = set()
    for arm, color in (("standard", STD), ("algebraic", ALG)):
        paths = glob.glob(str(ROOT / f"results/sweep/{arm}_lr*_s*.json"))
        lrs, vals = _lr_table(paths, rf"{arm}_lr([0-9.e-]+)_s\d+\.json")
        ax.plot(lrs, vals, marker="o", color=color, label=arm)
        grid.update(lrs)
    _lr_axis(ax, sorted(grid))
    ax.set_title("ViT-B/16, 224 px, 600M tokens")
    fig.tight_layout()
    save(fig, "learning_rate")


def _curves(paths):
    out = []
    for path in sorted(paths):
        rec = load(path)
        pts = [(h["step"], 100 * h["minival_top1"]) for h in rec["history"]
               if "minival_top1" in h]
        pts.append((rec["steps"], 100 * rec["results"]["minival"]["top1"]))
        out.append(np.array(pts))
    return out


def fig_curves(name, title, pattern_std, pattern_alg, pattern_ctrl=None):
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    sets = [("standard", STD, pattern_std), ("algebraic", ALG, pattern_alg)]
    if pattern_ctrl:
        sets.append(("standard + RoPE-2D", CTRL, pattern_ctrl))
    ends = []
    for arm, color, pattern in sets:
        curves = _curves(glob.glob(str(ROOT / pattern)))
        if not curves:
            continue
        for c in curves:
            ax.plot(c[:, 0], c[:, 1], color=color, alpha=0.35, linewidth=0.9)
        mean = np.mean([c[:, 1] for c in curves], axis=0)
        ax.plot(curves[0][:, 0], mean, color=color, marker="o", label=arm)
        ends.append((curves[0][-1, 0], mean[-1], f"{mean[-1]:.1f}"))
    fig.tight_layout()
    label_ends(ax, ends)
    ax.set_xlabel("training step")
    ax.set_ylabel("minival top-1 (%)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, name)


def fig_entropy(path, name, title):
    """Attention entropy and score size per block of trained models."""
    if not Path(path).exists():
        return
    d = load(path)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for key, label, color in (("std", "standard", STD),
                              ("alg", "algebraic", ALG)):
        if key not in d:
            continue
        blocks = range(1, len(d[key]) + 1)
        axes[0].plot(blocks, [r["entropy"] for r in d[key]], marker="o",
                     color=color, label=label)
        axes[1].plot(blocks, [r["score_q50"] for r in d[key]], marker="o",
                     color=color, label=label)
    for ax in axes:
        ax.set_xlabel("block")
        ax.set_xticks(range(1, 13))
    axes[0].set_ylabel("attention entropy / log N")
    axes[0].set_title(f"Attention entropy ({title})")
    axes[0].legend(loc="lower left")
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    axes[1].yaxis.set_minor_formatter(mticker.NullFormatter())
    axes[1].set_ylabel("median |score|")
    axes[1].set_title("Score size")
    fig.tight_layout()
    save(fig, name)


def main():
    fig_kernels()
    fig_stress()
    fig_lr()
    fig_curves("pilot_curves", "ViT-S/4, 64 px, 10 epochs",
               "results/pilot/standard_s*.json",
               "results/pilot/algebraic_s*.json")
    if glob.glob(str(ROOT / "results/main/*.json")):
        fig_curves("main_curves", "ViT-B/16, 224 px, 2.5B tokens",
                   "results/main/standard_s*.json",
                   "results/main/algebraic_s*.json",
                   "results/main/standard_rope2d_s*.json")
    fig_entropy(ROOT / "results/analysis/attention_stats_main.json",
                "attention_entropy", "trained ViT-B/16, seed 42")


if __name__ == "__main__":
    main()
