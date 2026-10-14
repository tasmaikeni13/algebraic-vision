#!/usr/bin/env python3
"""Run many single-chip training jobs across the TPU pod.

Each of the pod's hosts runs up to four independent jobs, one per chip. The
repository is synced to every host first; each job's JSON result is copied
back when it finishes. Jobs whose result already exists are skipped, so an
interrupted grid can simply be restarted.

Usage::

    python scripts/grid.py jobs.json [--hosts localhost w1 w2 w3]

``jobs.json`` is a list of TrainConfig dictionaries; each needs a unique
``name`` and an ``output`` path relative to the repository root.
"""

import argparse
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "~/algebraic-vision"
CHIP_ENV = ("TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 "
            "TPU_VISIBLE_CHIPS={chip} TPU_PROCESS_PORT={port} "
            "TPU_PROCESS_ADDRESSES=localhost:{port} CLOUD_TPU_TASK_ID=0")
HOST_ENV = ("TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 "
            "TPU_VISIBLE_CHIPS=0,1,2,3")


def sync(hosts):
    for host in hosts:
        if host == "localhost":
            continue
        subprocess.run(
            ["rsync", "-a", "--exclude", ".venv", "--exclude", ".git",
             "--exclude", "formal/.lake", "--exclude", "__pycache__",
             "--exclude", "results/", f"{ROOT}/", f"{host}:{REMOTE_ROOT}/"],
            check=True)
        subprocess.run(["rsync", "-a", f"{ROOT}/results/configs/",
                        f"{host}:{REMOTE_ROOT}/results/configs/"],
                       check=True)


def commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except subprocess.CalledProcessError:
        return ""


def worker(host, chip, jobs, log_dir, rev, lock, done):
    if chip is None:
        env = HOST_ENV
    else:
        env = CHIP_ENV.format(chip=chip, port=8476 + chip)
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        cfg_path = f"results/configs/{job['name']}.json"
        cmd = (f"cd {REMOTE_ROOT} && {env} GIT_COMMIT={rev} "
               f"PYTHONPATH=. .venv/bin/python -m algebraic_vision.train "
               f"--config {cfg_path}")
        argv = ["bash", "-c", cmd] if host == "localhost" else \
            ["ssh", host, cmd]
        start = time.time()
        with open(log_dir / f"{job['name']}.log", "w") as log:
            code = subprocess.run(argv, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        if host != "localhost" and code == 0:
            out = ROOT / job["output"]
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a",
                            f"{host}:{REMOTE_ROOT}/{job['output']}", str(out)])
        with lock:
            done.append(job["name"])
            print(f"[{len(done)}] {job['name']} on {host}:{chip} exit={code} "
                  f"{(time.time() - start) / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--hosts", nargs="+",
                        default=["localhost", "w1", "w2", "w3"])
    parser.add_argument("--chips", type=int, default=4)
    parser.add_argument("--reserve", nargs="*", default=[],
                        help="HOST:CHIP slots to leave free")
    parser.add_argument("--whole-host", action="store_true",
                        help="one job per host on all four chips")
    args = parser.parse_args()

    jobs = json.loads(args.jobs.read_text())
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise ValueError("job names must be unique")
    cfg_dir = ROOT / "results/configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "results/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    for job in jobs:
        (cfg_dir / f"{job['name']}.json").write_text(json.dumps(job, indent=1))
        if not (ROOT / job["output"]).exists():
            pending.put(job)
    print(f"{pending.qsize()} of {len(jobs)} jobs to run on "
          f"{len(args.hosts)} hosts"
          + ("" if args.whole_host else f" x {args.chips} chips"), flush=True)
    sync(args.hosts)
    rev = commit()
    lock, done = threading.Lock(), []
    if args.whole_host:
        slots = [(h, None) for h in args.hosts]
    else:
        slots = [(h, c) for c in range(args.chips) for h in args.hosts
                 if f"{h}:{c}" not in args.reserve]
    threads = [threading.Thread(target=worker,
                                args=(h, c, pending, log_dir, rev, lock, done))
               for h, c in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
