#!/usr/bin/env bash
# Run one command on every host of the TPU pod at the same time.
#
#   [COLLECT="path ..."] scripts/pod_run.sh LOGFILE COMMAND...
#
# The repository (without results/) is synced to the workers first. Host 0
# (this machine) writes to LOGFILE; worker w writes to LOGFILE.w. The script
# waits for all hosts and exits non-zero if any of them failed. JAX process 0,
# which writes the results, can be on any host, so the files named in COLLECT
# (paths relative to the repository) are copied back from the workers.
set -u
LOG=$1
shift
CMD="$*"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKERS=${WORKERS:-"w1 w2 w3"}

for w in $WORKERS; do
  rsync -a --exclude .venv --exclude .git --exclude formal/.lake \
    --exclude __pycache__ --exclude results/ "$ROOT/" "$w:algebraic-vision/"
  rsync -a "$ROOT/results/configs/" "$w:algebraic-vision/results/configs/" \
    2>/dev/null
done

REV=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "")
pids=()
for w in $WORKERS; do
  ssh "$w" "cd ~/algebraic-vision && GIT_COMMIT=$REV PYTHONPATH=. $CMD" \
    > "$LOG.$w" 2>&1 &
  pids+=($!)
done
(cd "$ROOT" && GIT_COMMIT=$REV PYTHONPATH=. bash -c "$CMD") > "$LOG" 2>&1
status=$?
for p in "${pids[@]}"; do
  wait "$p" || status=1
done
for path in ${COLLECT:-}; do
  mkdir -p "$(dirname "$ROOT/$path")"
  for w in $WORKERS; do
    rsync -a --update "$w:algebraic-vision/$path" "$ROOT/$path" 2>/dev/null
  done
done
exit $status
