"""Verify Phase 1 artifacts, frozen files, tests, Lean build and all axioms."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASELINE="e265e045ca9dcef0a236efd1a155a129e064fc9f"


def input_snapshot():
    paths=list((ROOT/"analysis/phase1").glob("*.py"))+list((ROOT/"tests").glob("*.py"))
    paths+=list((ROOT/"formal/AlgebraicVision").glob("*.lean"))
    paths += [ROOT/name for name in ["formal/AlgebraicVision.lean","formal/lakefile.toml",
              "formal/lake-manifest.json","formal/lean-toolchain","requirements-phase1.txt","pytest.ini"]]
    paths += [ROOT/name for name in ["theory.md","README.md","analysis/phase1/README.md","formal/README.md","results/README.md"]]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def command(args,cwd,env,log):
    start=time.monotonic()
    result=subprocess.run(args,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log.append(f"$ {' '.join(str(a) for a in args)}\n{result.stdout}\nexit={result.returncode}; seconds={time.monotonic()-start:.3f}\n")
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"results/phase1")
    args=parser.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,JAX_PLATFORMS="cpu",JAX_ENABLE_X64="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    snapshot=input_snapshot()
    log=[]
    tests=command([sys.executable,"-m","pytest","-q"],ROOT,env,log)
    print(tests.stdout,flush=True)
    lake=shutil.which("lake") or str(Path.home()/".elan/bin/lake")
    formal=command([lake,"build"],ROOT/"formal",env,log)
    print(formal.stdout,flush=True)
    names=[]
    for path in sorted((ROOT/"formal/AlgebraicVision").glob("*.lean")):
        source=path.read_text()
        namespace=re.search(r"^namespace ([\w.]+)",source,re.M).group(1)
        names += [namespace+"."+name for name in re.findall(r"^(?:theorem|lemma) ([\w']+)",source,re.M)]
    audit=out/"axioms.lean"
    audit.write_text("import AlgebraicVision\n\n"+"\n".join("#print axioms "+name for name in names)+"\n")
    axioms=command([lake,"env","lean",str(audit)],ROOT/"formal",env,log)
    disclosed=re.findall(r"depends on axioms:\s*\[([^\]]*)\]",axioms.stdout,re.S)
    all_axioms={a.strip() for row in disclosed for a in row.split(",") if a.strip()}
    declarations=len(disclosed)+axioms.stdout.count("does not depend on any axioms")
    standard=bool(axioms.returncode==0 and declarations==len(names) and all_axioms<={"propext","Classical.choice","Quot.sound"})
    print(f"Axiom audit: {declarations}/{len(names)} declarations; {sorted(all_axioms)}",flush=True)
    files=subprocess.check_output(["git","ls-tree","-r","--name-only",BASELINE],cwd=ROOT,text=True).splitlines()
    allowed={"README.md","formal/README.md","formal/AlgebraicVision.lean","results/README.md"}
    frozen={}
    mismatches=[]
    for name in files:
        if name in allowed:
            continue
        old=subprocess.check_output(["git","show",f"{BASELINE}:{name}"],cwd=ROOT)
        digest=hashlib.sha256(old).hexdigest()
        frozen[name]=digest
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            mismatches.append(name)
    validation={"tests_passed":tests.returncode==0,"test_summary":tests.stdout.strip().splitlines()[-1],
                "lean_build_passed":formal.returncode==0,"axioms_standard_only":standard,
                "audited_declarations":names,"axioms":sorted(all_axioms),
                "baseline_commit":BASELINE,"frozen_baselines_unchanged":not mismatches,
                "frozen_file_sha256":frozen,"frozen_file_mismatches":mismatches,
                "verification_source_sha256":snapshot,"verification_inputs_unchanged":snapshot==input_snapshot(),
                "scope":"CPU tests, pinned full Lean build and every local theorem/lemma axiom audit; no Phase 2-5 work"}
    (out/"verification.log").write_text("\n".join(log))
    (out/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    from .report import build
    gate=build(out)
    passed=all([tests.returncode==0,formal.returncode==0,standard,not mismatches,validation["verification_inputs_unchanged"],
                not gate.get("evidence_integrity_issues"),not gate.get("coverage_issues"),not gate.get("missing_evidence"),not gate.get("missing_revision_evidence")])
    print(f"Verification {'PASS' if passed else 'FAIL'}; research gate {gate['decision']}",flush=True)
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main())
