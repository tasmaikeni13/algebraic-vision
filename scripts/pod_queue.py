#!/usr/bin/env python3
"""Run a list of training jobs one after another on the whole TPU pod.

Each job is a TrainConfig dictionary (with a unique name and an output
path); it is written to results/configs/ and launched on every host with
scripts/pod_run.sh, which copies the output back from whichever host wrote
it. Jobs whose output already exists are skipped, so the queue can be
restarted after an interruption.

Usage: python scripts/pod_queue.py JOBS.json
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text())
    (ROOT / "results/configs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/logs").mkdir(parents=True, exist_ok=True)
    for i, job in enumerate(jobs):
        if (ROOT / job["output"]).exists():
            print(f"[{i + 1}/{len(jobs)}] {job['name']} done already",
                  flush=True)
            continue
        job = dict(job, distributed=True)
        cfg = ROOT / "results/configs" / f"{job['name']}.json"
        cfg.write_text(json.dumps(job, indent=1))
        log = ROOT / "results/logs" / f"{job['name']}.log"
        start = time.time()
        code = subprocess.run(
            [str(ROOT / "scripts/pod_run.sh"), str(log),
             ".venv/bin/python -m algebraic_vision.train --config "
             f"results/configs/{job['name']}.json"],
            cwd=ROOT, env=dict(os.environ, COLLECT=job["output"])).returncode
        ok = (ROOT / job["output"]).exists()
        print(f"[{i + 1}/{len(jobs)}] {job['name']} exit={code} "
              f"result={'yes' if ok else 'MISSING'} "
              f"{(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
