"""Run Phase 1 only. No production model, kernel, dataset or launcher edits.

    python -m analysis.phase1.run
    python -m analysis.phase1.run --sections symbolic,precision
    python -m analysis.phase1.run --report-only --require-pass
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("JAX_ENABLE_X64","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")

import jax  # noqa: E402

from . import convergence, propagation, simulations, symbolic  # noqa: E402

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUT=ROOT/"results/phase1"
SECTIONS=["symbolic","stress","crowding","outliers","precision","positions","propagation","convergence","purity"]


def source_hashes():
    paths=sorted(p for p in (ROOT/"analysis/phase1").glob("*.py")
                 if p.name not in {"run.py","report.py","verify.py"})
    paths+=sorted((ROOT/"algebraic_vision").glob("*.py"))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections",default=",".join(SECTIONS))
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--seeds",type=int,nargs="+",default=list(range(42,50)))
    parser.add_argument("--prototype-seeds",type=int,nargs="+",default=[42,43,44])
    parser.add_argument("--steps",type=int,default=200)
    parser.add_argument("--require-pass",action="store_true")
    parser.add_argument("--report-only",action="store_true",help="Validate cached evidence and rebuild the report without simulations")
    args=parser.parse_args()
    if args.steps<1 or not args.seeds or not args.prototype_seeds:
        parser.error("positive steps and nonempty seeds required")
    sections=[] if args.report_only else args.sections.split(",")
    if any(name not in SECTIONS for name in sections):
        parser.error(f"sections must be drawn from {SECTIONS}")
    from . import purity
    functions={"symbolic":symbolic.run,"stress":lambda:simulations.stress(args.seeds),
               "crowding":lambda:simulations.crowding(args.seeds),"outliers":lambda:simulations.outliers(args.seeds),
               "precision":lambda:simulations.precision(args.seeds),"positions":simulations.positions,
               "propagation":lambda:propagation.run(args.prototype_seeds),
               "convergence":lambda:convergence.run(args.prototype_seeds,args.steps),"purity":purity.run}
    for section in sections:
        start=time.monotonic()
        hashes=source_hashes()
        print(f"START {section}",flush=True)
        data=functions[section]()
        data["provenance"]={"source_sha256":hashes,"elapsed_seconds":time.monotonic()-start,
                            "seeds":args.seeds if section in ["stress","crowding","outliers","precision"] else args.prototype_seeds if section in ["propagation","convergence"] else [],
                            "python":platform.python_version(),"backend":jax.default_backend(),
                            "jax":jax.__version__,"x64":jax.config.x64_enabled,
                            "platform":platform.platform(),
                            "dependencies":{name:importlib.metadata.version(name) for name in ["jax","jaxlib","numpy","sympy","mpmath","scipy","matplotlib"]}}
        if hashes!=source_hashes():
            raise RuntimeError("sources changed during section; rerun to preserve provenance")
        write(args.out/f"{section}.json",data)
        print(f"DONE {section}: {time.monotonic()-start:.1f}s",flush=True)
        jax.clear_caches()
    if not args.report_only and set(sections)==set(SECTIONS):
        # Run the final refinement separately so each section keeps its own
        # immutable source snapshot and the original evidence stays intact.
        import sys
        cmd=[sys.executable,"-m","analysis.phase1.revision_study","--out",str(args.out),
             "--steps",str(args.steps),"--seeds",*[str(s) for s in args.seeds],
             "--prototype-seeds",*[str(s) for s in args.prototype_seeds]]
        subprocess.run(cmd,check=True)
        from .costs import run as count_costs
        write(args.out/"costs.json",count_costs())
    if all((args.out/f"{name}.json").exists() for name in SECTIONS):
        from .report import build
        gate=build(args.out)
        print(f"Phase 1 decision: {gate['decision']}",flush=True)
        if args.require_pass and gate["decision"]!="GO":
            return 2
    elif args.require_pass:
        print("Required evidence is missing; refusing a passing decision",flush=True)
        return 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
