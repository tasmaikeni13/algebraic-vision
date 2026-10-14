"""Bounded pod integration: partial checkpoint, resume, matrix and reports."""

import argparse
import json
from pathlib import Path

from algebraic_vision.production import atomic_json
from experiments.production import jobs
from scripts.pod import ROOT,run_command,run_jobs,sync


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",nargs=3,default=["w1","w2","w3"])
    parser.add_argument("--timeout",type=int,default=3600)
    parser.add_argument("--skip-resume-probe",action="store_true")
    args=parser.parse_args()
    matrix=jobs("smoke")
    atomic_json(ROOT/"results/configs/smoke.json",matrix)
    # First run stops cleanly after one of the configured three steps. The
    # queue then resumes it using the exact same full-budget configuration.
    if not args.skip_resume_probe and not (ROOT/matrix[0]["output"]).exists():
        partial=dict(matrix[0],stop_after_steps=1)
        atomic_json(ROOT/"results/configs/smoke_partial.json",partial)
        sync(args.workers)
        run_command([".venv/bin/python","-m","algebraic_vision.production","--config",
                     "results/configs/smoke_partial.json"],"smoke_resume_probe",args.workers,
                    (partial["output"],),args.timeout)
    run_jobs(matrix,args.workers,args.timeout)
    records=[json.loads((ROOT/c["output"]).read_text()) for c in matrix]
    from analysis.production_fairness import check_records
    from analysis.production_select import select
    from analysis.production_report import report
    check_records(records,matrix)
    selections=select(matrix,records,allow_synthetic=True)
    atomic_json(ROOT/"results/smoke/selections.json",selections)
    report(matrix,records,ROOT/"results/smoke/report")
    if records[0]["resumed_at_step"]!=1 and not args.skip_resume_probe:
        raise AssertionError("pod checkpoint resume was not exercised")
    print("Pod smoke, checkpoint resume, selection and reporting passed",flush=True)


if __name__=="__main__":
    main()
