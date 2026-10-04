#!/usr/bin/env python3
"""Write paper/numbers.tex: every measured number quoted in the paper.

Each value is read from the records in results/; a missing record is written
as ?? so that it shows up in the PDF.
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper/numbers.tex"


def load(path):
    return json.loads(Path(path).read_text())


def runs(directory, name):
    """Records <name>_s<seed>.json in a results directory (exact name)."""
    pattern = re.compile(rf"{re.escape(name)}_s\d+\.json")
    return [load(p) for p in sorted((RESULTS / directory).glob("*.json"))
            if pattern.fullmatch(p.name)]


def by_seed(rs, split="val"):
    return {r["config"]["seed"]: r["results"][split]["top1"] for r in rs}


def mean_of(rs, metric="top1", split="val"):
    return float(np.mean([r["results"][split][metric] for r in rs]))


def paired(a_runs, b_runs, split="val"):
    a, b = by_seed(a_runs, split), by_seed(b_runs, split)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    d = np.array([a[s] - b[s] for s in seeds])
    se = d.std(ddof=1) / np.sqrt(len(d))
    tcrit = stats.t.ppf(0.975, len(d) - 1)
    return {"diff": d.mean(), "se": se, "wins": int((d > 0).sum()),
            "n": len(d),
            "p": stats.ttest_1samp(d, 0.0, alternative="greater").pvalue,
            "ci": (d.mean() - tcrit * se, d.mean() + tcrit * se)}


def pct(x, digits=2):
    return f"{100 * x:.{digits}f}"


def signed(x, digits=2):
    text = f"{100 * x:+.{digits}f}"
    return text.replace("-", "$-$")


def sci(x, digits=1):
    mant, exp = f"{x:.{digits}e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def power10(x):
    mant, exp = f"{x:.1e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    if mant == "1":
        return f"10^{{{int(exp)}}}"
    return f"{mant}\\times10^{{{int(exp)}}}"


def p_value(p):
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return f"{mant}\\times10^{{{int(exp)}}}"
    return f"{p:.3f}"


def thousands(x):
    return f"{x:,.0f}".replace(",", "{,}")


def small_scale(n):
    std = runs("small_scale/round1", "std")
    if std:
        n["compStd"] = pct(mean_of(std))
    components = (("Rad", "std_rad"), ("Cay", "std_cay"),
                  ("Alu", "std_alu"), ("Pow", "std_pow"))
    for key, name in components:
        rs = runs("small_scale/round1", name)
        pr = paired(rs, std)
        if pr:
            n[f"comp{key}"] = pct(mean_of(rs))
            n[f"comp{key}Diff"] = signed(pr["diff"])
    rad_cay = runs("small_scale/round4", "rad_caymix")
    if rad_cay:
        n["compRadCay"] = pct(mean_of(rad_cay))

    # Each arm at its best setting, four paired seeds.
    alg = runs("small_scale/round5", "v3_lr1.5e-3_wu2500")
    std_best = runs("small_scale/round4_seeds", "std_best")
    pr = paired(alg, std_best)
    if pr:
        n["microDiff"] = f"{100 * pr['diff']:+.1f}"
        n["microP"] = p_value(pr["p"])
        n["microAlg"] = pct(mean_of(alg))
        n["microStd"] = pct(mean_of(std_best))
    rope = runs("small_scale/round4", "std_rope")
    if rope:
        n["microRope"] = pct(mean_of(rope))
    for key, directory, name in (
            ("microStdLong", "round6", "std_lr5e-4_wu4000"),
            ("microAlgLong", "round6", "v3_lr1.5e-3_wu4000"),
            ("microStdCollapse", "round2", "std_lr2e-3"),
            ("microRadCollapse", "round2", "std_rad_lr2e-3")):
        rs = runs(f"small_scale/{directory}", name)
        if rs:
            n[key] = pct(mean_of(rs), 1 if "Collapse" in key else 2)


def pilot(n):
    alg, std = runs("pilot", "algebraic"), runs("pilot", "standard")
    pr = paired(alg, std)
    if pr:
        n["pilotAlg"] = pct(mean_of(alg))
        n["pilotStd"] = pct(mean_of(std))
        n["pilotDiff"] = f"{100 * pr['diff']:+.1f}"
        n["pilotAlgNll"] = f"{mean_of(alg, 'nll'):.2f}"
        n["pilotStdNll"] = f"{mean_of(std, 'nll'):.2f}"


def sweep(n):
    path = RESULTS / "sweep/selection.json"
    if not path.exists():
        return
    selection = load(path)["selection"]
    for arm, key in (("standard", "Std"), ("algebraic", "Alg")):
        if arm in selection:
            n[f"sweep{key}Lr"] = power10(selection[arm]["lr"])
            n[f"sweep{key}Val"] = pct(selection[arm]["val"])


def main_runs(n):
    alg, std = runs("main", "algebraic"), runs("main", "standard")
    rope = runs("main", "standard_rope2d")
    if alg:
        n["mainTokens"] = thousands(alg[0]["patch_tokens_seen"])
    for key, rs in (("Alg", alg), ("Std", std), ("Rope", rope)):
        if not rs:
            continue
        top1 = [r["results"]["val"]["top1"] for r in rs]
        se = np.std(top1, ddof=1) / np.sqrt(len(top1)) if len(rs) > 1 else 0
        n[f"main{key}Val"] = pct(np.mean(top1))
        n[f"main{key}Se"] = pct(se)
        n[f"main{key}Nll"] = f"{mean_of(rs, 'nll'):.3f}"
        n[f"main{key}Ece"] = f"{mean_of(rs, 'ece'):.3f}"
        n[f"main{key}Ips"] = thousands(
            np.mean([r["train_images_per_sec"] for r in rs]))
        n[f"main{key}Lr"] = f"${power10(rs[0]['config']['lr'])}$"
    pr = paired(alg, std)
    if pr:
        n["mainDiff"] = f"{100 * pr['diff']:+.2f}"
        n["mainP"] = p_value(pr["p"])
        n["mainCI"] = (f"[{100 * pr['ci'][0]:+.2f}, "
                       f"{100 * pr['ci'][1]:+.2f}]")
        n["mainWins"] = f"{pr['wins']}/{pr['n']}"
        a, b = by_seed(alg), by_seed(std)
        n["mainDiffSeeds"] = ", ".join(
            f"{100 * (a[s] - b[s]):+.2f}" for s in sorted(set(a) & set(b)))
        ratio = (np.mean([r["train_images_per_sec"] for r in alg])
                 / np.mean([r["train_images_per_sec"] for r in std]))
        n["mainSpeedRatio"] = f"{ratio:.2f}$\\times$"
    pr = paired(rope, std)
    if pr:
        n["mainRopeDiff"] = f"{100 * pr['diff']:+.2f}"
    pr = paired(alg, rope)
    if pr:
        n["mainAlgRopeDiff"] = f"{100 * pr['diff']:.2f}"
        n["mainAlgRopeWins"] = f"{pr['wins']}/{pr['n']}"
        n["mainAlgRopeP"] = p_value(pr["p"])
        n["mainAlgRopeCI"] = (f"[{100 * pr['ci'][0]:+.2f}, "
                              f"{100 * pr['ci'][1]:+.2f}]")


def analysis(n):
    stress = load(RESULTS / "analysis/stress.json")["gaussian"]
    n["stressSoftEnt"] = f"{stress['softmax'][-1]['entropy']:.3f}"
    n["stressRadEnt"] = f"{stress['radical_n8'][-1]['entropy']:.2f}"
    n["stressSoftSat"] = pct(stress["softmax"][-1]["saturated_rows"], 0)
    n["stressRadSat"] = pct(stress["radical_n8"][-1]["saturated_rows"], 0)

    prec = load(RESULTS / "analysis/precision.json")
    rows = {"Soft": ("attention", "softmax", "own_tv", "round_tv"),
            "Rad": ("attention", "radical_n8", "own_tv", "round_tv"),
            "Ce": ("loss", "cross_entropy", "own_grad", "round_grad"),
            "Pow": ("loss", "power_63_64", "own_grad", "round_grad")}
    for key, (group, name, own, rnd) in rows.items():
        row = next(r for r in prec[group][name] if r["scale"] == 30.0)
        n[f"prec{key}Own"] = sci(row[own])
        n[f"prec{key}Round"] = sci(row[rnd])

    checks = load(RESULTS / "analysis/symbolic_checks.json")
    n["symbolicChecks"] = str(checks["count"])
    purity = load(RESULTS / "analysis/purity_tpu_b16.json")
    n["purityRsqrt"] = str(purity["algebraic"]["rsqrt"])
    n["purityStdExp"] = str(
        purity["standard"]["transcendental_ops"].get("exponential", 0))

    smoke_alg = RESULTS / "smoke/smoke_b16_v3.json"
    smoke_std = RESULTS / "smoke/smoke_b16_standard.json"
    if smoke_alg.exists() and smoke_std.exists():
        n["cayleyParams"] = thousands(
            load(smoke_alg)["params"] - load(smoke_std)["params"])

    attn = RESULTS / "analysis/attention_stats_main.json"
    if attn.exists():
        stats_ = load(attn)
        std, alg = stats_["std"], stats_["alg"]
        n["attnStdScore"] = f"{max(r['score_q50'] for r in std):.0f}"
        n["attnAlgScore"] = f"{max(r['score_q50'] for r in alg):.0f}"
        n["attnBroader"] = str(sum(a["entropy"] > b["entropy"]
                                   for a, b in zip(alg, std)))
        n["attnBlocks"] = str(len(alg))
        n["attnStdOnehot"] = pct(max(r["onehot_rows"] for r in std), 1)
        n["attnAlgOnehotMid"] = pct(max(r["onehot_rows"]
                                        for r in alg[1:-2]), 1)

    bench = RESULTS / "kernels/attention_bench.json"
    if bench.exists():
        shape = load(bench)["shapes"]["main ViT-B/16@224"]
        for key, impl in (("SoftPallas", "softmax/pallas"),
                          ("RadPallas", "radical/pallas"),
                          ("SoftXla", "softmax/xla"),
                          ("RadXla", "radical/xla")):
            n[f"kern{key}"] = f"{shape[impl]['fwd_bwd_ms']:.3f}"


NAMES = [
    "compStd", "compRad", "compRadDiff", "compCay", "compCayDiff", "compAlu",
    "compAluDiff", "compPow", "compPowDiff", "compRadCay",
    "microDiff", "microP", "microAlg", "microStd", "microRope",
    "microStdLong", "microAlgLong", "microStdCollapse", "microRadCollapse",
    "pilotAlg", "pilotStd", "pilotDiff", "pilotAlgNll", "pilotStdNll",
    "sweepStdLr", "sweepStdVal", "sweepAlgLr", "sweepAlgVal",
    "mainTokens", "mainAlgVal", "mainStdVal", "mainRopeVal", "mainAlgSe",
    "mainStdSe", "mainRopeSe", "mainAlgNll", "mainStdNll", "mainRopeNll",
    "mainAlgEce", "mainStdEce", "mainRopeEce", "mainAlgIps", "mainStdIps",
    "mainRopeIps", "mainAlgLr", "mainStdLr", "mainRopeLr", "mainDiff",
    "mainP", "mainCI", "mainWins", "mainDiffSeeds", "mainSpeedRatio",
    "mainRopeDiff", "mainAlgRopeDiff", "mainAlgRopeWins", "mainAlgRopeP",
    "mainAlgRopeCI",
    "stressSoftEnt", "stressRadEnt", "stressSoftSat", "stressRadSat",
    "precSoftOwn", "precSoftRound", "precRadOwn", "precRadRound",
    "precCeOwn", "precCeRound", "precPowOwn", "precPowRound",
    "symbolicChecks", "purityRsqrt", "purityStdExp", "cayleyParams",
    "attnStdScore", "attnAlgScore", "attnBroader", "attnBlocks",
    "attnStdOnehot", "attnAlgOnehotMid",
    "kernSoftPallas", "kernRadPallas", "kernSoftXla", "kernRadXla",
]


def main():
    n = {}
    for part in (small_scale, pilot, sweep, main_runs, analysis):
        part(n)
    lines = ["% Generated by analysis/paper_numbers.py from results/; "
             "do not edit."]
    lines += [f"\\newcommand{{\\{key}}}{{{n.get(key, '??')}}}"
              for key in NAMES]
    OUT.write_text("\n".join(lines) + "\n")
    missing = [key for key in NAMES if key not in n]
    print(f"wrote {OUT}; missing: {missing}")


if __name__ == "__main__":
    main()
