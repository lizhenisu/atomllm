#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dlz/projects/atomllm-private}"
SESSION="public-pretraining-watchdog-v2"
LOG="artifacts/logs/public-pretraining-watchdog-v2.log"

cd "$PROJECT_ROOT"
mkdir -p artifacts/logs artifacts/locks

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "$SESSION is already running"
  exit 0
fi

printf -v command '%q' \
  "cd '$PROJECT_ROOT' && exec bash scripts/watch_public_pretraining_pipeline.sh 2>&1 | tee -a '$LOG'"
tmux new-session -d -s "$SESSION" "bash -lc $command"
echo "started $SESSION"
