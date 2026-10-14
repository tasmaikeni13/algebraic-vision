"""Audit concrete deliverables and refuse completion with missing evidence."""

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from algebraic_vision.production import ProductionConfig, atomic_json, source_hash
from analysis.production_fairness import check_matrix, check_records
from experiments.production import jobs, ARMS, MODELS

ROOT=Path(__file__).resolve().parents[2]


def main():
    evidence=[]
    def require(name,check,proof):
        evidence.append({"requirement":name,"satisfied":bool(check),"evidence":proof})
    smoke=json.loads((ROOT/"results/configs/smoke.json").read_text())
    sweep=json.loads((ROOT/"results/configs/sweep.json").read_text())
    template=json.loads((ROOT/"results/configs/main.template.json").read_text())
    for name,matrix,expected in [("smoke",smoke,18),("sweep",sweep,108),("main_template",template,18)]:
        check_matrix(matrix)
        require(f"{name}: full paired B/L matrix",len(matrix)==expected, f"results/configs/{'main.template' if name=='main_template' else name}.json; fairness validator")
    require("operator-only full runs",all(not c["launch_ready"] for c in template),"main template cannot pass ProductionConfig.validate; scripts/pod.py requires explicit execution flag")
    require("108-run equal tuning budget and minival-only selection",all(not c["evaluate_test"] and c["stage"]=="sweep" for c in sweep),"sweep config matrix and analysis/production_select.py")
    require("full-budget run matrices were not executed",not list((ROOT/"results/sweep").glob("*/*_s*.json")) and not list((ROOT/"results/main").glob("*/*_s*.json")),"no sweep/main result records; only synthetic smoke executed")
    records=[]
    for c in smoke:
        path=ROOT/c["output"]
        require(f"bounded distributed run: {c['name']}",path.exists(),c["output"])
        if path.exists():records.append(json.loads(path.read_text()))
    current=source_hash()
    if len(records)==18:
        check_records(records,smoke)
        require("all training evidence uses the current implementation",all(r["source_sha256"]==current for r in records),"record source hashes match production.source_hash()")
        require("all arms/scales run on full v4-32 topology",all(r["devices"]==16 and r["processes"]==4 and "v4" in r["device_kind"].lower() for r in records),"actual runtime device/host metadata")
        require("full B/L batch sizes and 1–5 finite steps",all(r["steps"]==3 and r["config"]["batch_size"]==(1024 if r["config"]["scale"]=="vit-b16" else 512) and not r["nonfinite_steps"] for r in records),"actual trained steps, batches, histories")
        require("compilation, memory footprint and throughput",all(r["compile_seconds"]>0 and r["memory"] and r["train_images_per_sec"]>0 and math.isfinite(r["train_images_per_sec"]) for r in records),"runtime compile timing, memory_analysis(), synchronized images/s")
        require("multi-host checkpoint resume",records[0]["resumed_at_step"]==1 and records[0]["history"][0]["step"]==1 and records[0]["history"][-1]["step"]==3,"first pod run stopped at step 1 then resumed to 3")
        require("validation Top-1, NLL, ECE and per-block diagnostics",all(set(("top1","nll","ece"))<=set(r["results"]["val"]) and len(r["diagnostics"])==r["model_config"]["depth"] for r in records),"distributed masked evaluation plus per-block entropy/score diagnostics")
        audits=[json.loads((ROOT/Path(c["output"]).with_suffix(".hlo_audit.json")).read_text()) for c in smoke]
        algebraic=[a for a in audits if a["arm"]!="standard"]
        require("TPU forward/loss/VJP/optimizer purity at both scales",all(a["pure"] and a["nested_programs"]["pure"] and a["source_sha256"]==current and a["backend"]=="tpu" for a in algebraic),"whole-step optimized HLO plus recursive Pallas program audits for all 12 algebraic runs")
        reductions={}
        for scale in MODELS:
            v1=next(a for a in algebraic if a["scale"]==scale and a["arm"]=="algebraic")
            v2=next(a for a in algebraic if a["scale"]==scale and a["arm"]=="algebraic_v2")
            reductions[scale]={"v1":v1["rsqrt"],"v2":v2["rsqrt"],"reduction":v1["rsqrt"]-v2["rsqrt"]}
        require("exact compiled root-count reduction",all(r["reduction"]>0 for r in reductions.values()),reductions)
    kernels=json.loads((ROOT/"results/verification/kernels_tpu.json").read_text())
    expected={(s,a,d) for s in MODELS for a in (*ARMS,"flash_standard","rational_ablation") for d in ("float32","bfloat16")}
    actual={(r["scale"],r["arm"],r["dtype"]) for r in kernels["records"]}
    require("full-pod B/L FP32/BF16 Pallas forward and analytic VJP validation",actual==expected and kernels["devices"]==16 and kernels["processes"]==4 and all(r["passed"] for r in kernels["records"]),"results/verification/kernels_tpu.json; all Q/K/V and mass/margin errors")
    require("current kernel benchmark provenance",kernels["source_sha256"]==current,"kernel producer source snapshot")
    ratios=[r["latency_ratio_to_flash_standard"] for r in kernels["records"] if r["arm"]=="algebraic_v2" and r["dtype"]=="bfloat16" and "latency_ratio_to_flash_standard" in r]
    require("selected v2 within 15% of standard FlashAttention",len(ratios)==2 and all(v<=1.15 for v in ratios),ratios)
    require("root-free proposal available only as rejected ablation",all(r["nested_programs"]["rsqrt"]==0 and r["nested_programs"]["pure"] for r in kernels["records"] if r["arm"]=="rational_ablation"),"rational_kernel.py preprocessing separated; Phase 1 retained-E8 selection explicitly documented")
    report_path=ROOT/"results/smoke/report/summary.json"
    selection_path=ROOT/"results/smoke/selections.json"
    require("selection serialization",selection_path.exists() and json.loads(selection_path.read_text()).get("synthetic"),"synthetic smoke selections; production main refuses them")
    require("paired differences, CIs, one-sided tests and figures",report_path.exists() and all((ROOT/f"results/smoke/report/{s}_{kind}.{suffix}").exists() for s in MODELS for kind in ("metrics","attention") for suffix in ("pdf","png")),"production_report summary and standalone PDF/PNG figures")
    log=(ROOT/"results/verification/tests.log").read_text()
    match=re.search(r"(\d+) passed",log)
    require("CPU regression, baseline equivalence, kernel and resume tests",bool(match and int(match.group(1))>=112 and "failed" not in log.lower()),"results/verification/tests.log; includes exact nonzero-head frozen baseline forward/VJP comparison")
    formal=subprocess.run([str(Path.home()/".elan/bin/lake"),"build"],cwd=ROOT/"formal",capture_output=True,text=True)
    (ROOT/"results/verification/lean.log").write_text(formal.stdout+formal.stderr)
    require("Lean build with existing proofs preserved",formal.returncode==0,"results/verification/lean.log")
    leanfiles=list((ROOT/"formal").rglob("*.lean"))
    leanfiles=[p for p in leanfiles if ".lake" not in p.parts]
    require("zero sorry admissions",not any("sorry" in p.read_text() for p in leanfiles),"all repository Lean source files")
    axioms=subprocess.run([str(Path.home()/".elan/bin/lake"),"env","lean",str(ROOT/"results/phase1/axioms.lean")],cwd=ROOT/"formal",capture_output=True,text=True)
    (ROOT/"results/verification/axioms.log").write_text(axioms.stdout+axioms.stderr)
    rows=re.findall(r"depends on axioms:\s*\[([^\]]*)\]",axioms.stdout,re.S)
    disclosed={a.strip() for row in rows for a in row.split(",") if a.strip()}
    require("formal theorem axiom audit",axioms.returncode==0 and len(rows)>100 and disclosed<={"propext","Classical.choice","Quot.sound"},"results/verification/axioms.log; Phase 1 inventory re-audited against current Lean build")
    require("copy-paste operator commands",all(s in (ROOT/"PHASES.md").read_text() for s in ("--execute-long-runs","analysis.production_select","analysis.production_report","--scales vit-b16","--scales vit-l16")),"PHASES.md Phase 4 and Phase 5 operator commands")
    result={"complete":all(e["satisfied"] for e in evidence),"source_sha256":current,"requirements":evidence,
            "phase1_refinement":"Selected E8 attention retained; zero inner roots applies to rejected rational ablation only. No claim that selected v2 cures E8 score attenuation.",
            "scope":"Phases 2–3 implementation/execution and phases 4–5 preparation/smoke; long training deliberately unexecuted"}
    atomic_json(ROOT/"results/verification/completion.json",result)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["complete"] else 1)


if __name__=="__main__":main()
