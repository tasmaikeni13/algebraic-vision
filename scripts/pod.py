"""Run validated production matrices on all four TPU hosts.

Long sweeps/main runs require --execute-long-runs, supplied by the operator.
Synthetic smoke verification is separately bounded by the config validator.
"""

import argparse
import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from algebraic_vision.production import ProductionConfig, atomic_json, config_hash
from analysis.production_fairness import check_matrix

ROOT = Path(__file__).resolve().parents[1]


def device_users(host):
    command=["fuser","/tmp/libtpu_lockfile",*[f"/dev/accel{i}" for i in range(4)]]
    argv=command if host=="localhost" else ["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10",host,shlex.join(command)]
    result=subprocess.run(argv,capture_output=True,text=True)
    if result.returncode not in (0,1):
        raise RuntimeError(f"cannot check TPU occupancy on {host}: {result.stderr.strip()}")
    return sorted(set(result.stdout.split()))


def wait_for_idle(workers):
    hosts=["localhost",*workers]
    previous=None
    while True:
        with ThreadPoolExecutor(max_workers=4) as pool:
            occupants=dict(zip(hosts,pool.map(device_users,hosts)))
        busy={host:pids for host,pids in occupants.items() if pids}
        if not busy:
            return
        if busy!=previous:
            print(f"Waiting for occupied TPU devices: {busy}",flush=True)
            previous=busy
        time.sleep(15)


def sync(workers):
    for worker in workers:
        subprocess.run(["ssh", "-o", "BatchMode=yes", worker,
                         "mkdir -p ~/algebraic-vision/results/configs"], check=True)
        subprocess.run(["rsync", "-a", "--exclude=.venv", "--exclude=.git", "--exclude=formal/.lake",
                         "--exclude=__pycache__", "--exclude=results/", str(ROOT) + "/",
                         f"{worker}:algebraic-vision/"], check=True)
        subprocess.run(["rsync", "-a", str(ROOT / "results/configs") + "/",
                         f"{worker}:algebraic-vision/results/configs/"], check=True)


def run_command(command, name, workers, collect=(), timeout=3600):
    wait_for_idle(workers)
    logs = ROOT / "results/logs"
    logs.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    shell_command = f"env GIT_COMMIT={shlex.quote(revision)} PYTHONPATH=. " + shlex.join(command)
    processes, streams = [], []
    try:
        for host in ["localhost", *workers]:
            stream = (logs / f"{name}.{host}.log").open("w")
            streams.append(stream)
            argv = ["bash", "-c", shell_command] if host == "localhost" else [
                "ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", host,
                "cd ~/algebraic-vision && " + shell_command]
            processes.append(subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
        start = time.monotonic()
        while any(p.poll() is None for p in processes):
            failed = [p.returncode for p in processes if p.poll() is not None and p.returncode]
            if failed:
                raise RuntimeError(f"pod worker failed; see results/logs/{name}.*.log")
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"pod command exceeded {timeout}s; see {logs}")
            time.sleep(1)
        if any(p.returncode for p in processes):
            raise RuntimeError(f"pod command failed: {name}")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for stream in streams:
            stream.close()
    for path in collect:
        target = ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        for worker in workers:
            subprocess.run(["rsync", "-a", "--update", f"{worker}:algebraic-vision/{path}", str(target)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_jobs(matrix, workers, timeout=3600, execute_long=False):
    check_matrix(matrix)
    for values in matrix:
        cfg = ProductionConfig(**values)
        cfg.validate()
        if cfg.stage != "smoke" and not execute_long:
            raise ValueError("operator must supply --execute-long-runs to launch sweeps/main runs")
    (ROOT / "results/configs").mkdir(parents=True, exist_ok=True)
    for values in matrix:
        atomic_json(ROOT / "results/configs" / f"{values['name']}.json", values)
    sync(workers)
    for values in matrix:
        cfg = ProductionConfig(**values)
        out = ROOT / cfg.output
        if out.exists():
            record = json.loads(out.read_text())
            if record["complete"] and record["config"] == values:
                print(f"Already complete: {cfg.name}", flush=True)
                continue
            if config_hash(ProductionConfig(**record["config"])) != config_hash(cfg):
                raise ValueError(f"output belongs to a different config: {out}")
        print(f"Launching {cfg.name} across 4 hosts", flush=True)
        run_command([".venv/bin/python", "-m", "algebraic_vision.production", "--config",
                     f"results/configs/{cfg.name}.json"], cfg.name, workers,
                    (cfg.output, str(Path(cfg.output).with_suffix(".hlo_audit.json"))), timeout)
        if not out.exists():
            raise RuntimeError(f"pod did not produce {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--workers", nargs=3, default=["w1", "w2", "w3"])
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--execute-long-runs", action="store_true")
    args = parser.parse_args()
    run_jobs(json.loads(args.matrix.read_text()), args.workers, args.timeout, args.execute_long_runs)


if __name__ == "__main__":
    main()
