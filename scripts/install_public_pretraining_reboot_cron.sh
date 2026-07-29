#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dlz/projects/atomllm-private}"
MARKER="# atomllm-public-pretraining-v2"
ENTRY="@reboot /bin/bash -lc 'cd $PROJECT_ROOT && scripts/start_public_pretraining_pipeline.sh >> artifacts/logs/public-pretraining-reboot-v2.log 2>&1' $MARKER"

temporary="$(mktemp)"
trap 'rm -f "$temporary"' EXIT
crontab -l 2>/dev/null | grep -Fv "$MARKER" >"$temporary" || true
printf '%s\n' "$ENTRY" >>"$temporary"
crontab "$temporary"
echo "installed reboot recovery: $ENTRY"
