#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dlz/projects/atomllm-private}"
POLL_SECONDS="${POLL_SECONDS:-300}"

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p artifacts/locks artifacts/logs
exec 9>artifacts/locks/public-pretraining-pipeline-v2.lock
if ! flock -n 9; then
  echo "public pretraining pipeline watchdog is already running" >&2
  exit 1
fi

while true; do
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] reconcile pipeline\n' -1
  bash scripts/launch_public_pretraining_pipeline.sh
  sleep "$POLL_SECONDS"
done
