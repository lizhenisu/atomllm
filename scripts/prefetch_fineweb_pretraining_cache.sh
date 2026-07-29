#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dlz/projects/atomllm-private}"
REVISION="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
CONFIG_NAME="sample-350BT"
FILE_COUNT="${FILE_COUNT:-472}"
# Keep one Hub worker while source encoders stream concurrently. Two workers
# stalled behind the shared proxy in a live comparison, while four workers
# repeatedly forced committed-work rollback after SSL EOFs.
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-1}"
MAX_ATTEMPTS_PER_FILE="${MAX_ATTEMPTS_PER_FILE:-1000}"
CACHE_ROOT="artifacts/source-cache/huggingface/HuggingFaceFW/fineweb-edu/$REVISION"
COMPLETED="$CACHE_ROOT/PREFETCH-ALL472-COMPLETED"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export HF_HUB_DOWNLOAD_TIMEOUT=600
# hf-mirror does not reliably serve the Hub's Xet transfer protocol through
# the shared proxy: hf_xet can wait forever without an active network socket.
# Plain HTTP keeps local-dir partial files resumable across per-file retries.
export HF_HUB_DISABLE_XET=1

mapfile -t files < <(
  CONFIG_NAME="$CONFIG_NAME" REVISION="$REVISION" FILE_COUNT="$FILE_COUNT" \
    python - <<'PY'
import os

from datasets import load_dataset_builder

builder = load_dataset_builder(
    "HuggingFaceFW/fineweb-edu",
    os.environ["CONFIG_NAME"],
    revision=os.environ["REVISION"],
    token=True,
)
files = list(builder.config.data_files["train"])
count = int(os.environ["FILE_COUNT"])
if not 1 <= count <= len(files):
    raise SystemExit(f"FILE_COUNT must be in [1, {len(files)}]")
prefix = (
    "hf://datasets/HuggingFaceFW/fineweb-edu@"
    f"{os.environ['REVISION']}/"
)
for path in files[:count]:
    value = str(path)
    if not value.startswith(prefix):
        raise SystemExit(f"unexpected Hub path: {value}")
    print(value.removeprefix(prefix))
PY
)

if [[ "${#files[@]}" -ne "$FILE_COUNT" ]]; then
  echo "resolved file count does not match FILE_COUNT" >&2
  exit 1
fi

if [[ ! "$DOWNLOAD_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DOWNLOAD_WORKERS must be a positive integer" >&2
  exit 1
fi

download_file() {
  local index="$1"
  local relative="$2"
  local local_file="$CACHE_ROOT/$relative"
  local metadata="$CACHE_ROOT/.cache/huggingface/download/$relative.metadata"
  local attempt=0
  local delay

  if [[ -f "$local_file" && -f "$metadata" ]] \
    && [[ "$(sed -n '1p' "$metadata")" == "$REVISION" ]]; then
    echo "cached index=$index/$FILE_COUNT file=$relative"
    return
  fi

  while true; do
    if hf download HuggingFaceFW/fineweb-edu "$relative" \
      --repo-type dataset \
      --revision "$REVISION" \
      --local-dir "$CACHE_ROOT" \
      --max-workers "$DOWNLOAD_WORKERS"; then
      echo "downloaded index=$index/$FILE_COUNT file=$relative"
      return
    fi
    attempt=$((attempt + 1))
    if ((attempt >= MAX_ATTEMPTS_PER_FILE)); then
      echo "download failed index=$index/$FILE_COUNT file=$relative attempts=$attempt" >&2
      return 1
    fi
    delay=$((attempt * 5))
    if ((delay > 60)); then
      delay=60
    fi
    echo "download retry index=$index/$FILE_COUNT file=$relative attempt=$attempt delay=$delay" >&2
    sleep "$delay"
  done
}

for index in "${!files[@]}"; do
  download_file "$index" "${files[$index]}"
done

for index in "${!files[@]}"; do
  relative="${files[$index]}"
  local_file="$CACHE_ROOT/$relative"
  metadata="$CACHE_ROOT/.cache/huggingface/download/$relative.metadata"
  metadata_revision="$(sed -n '1p' "$metadata")"
  expected_sha256="$(sed -n '2p' "$metadata")"
  actual_sha256="$(sha256sum "$local_file" | cut -d ' ' -f 1)"
  if [[ "$metadata_revision" != "$REVISION" ]]; then
    echo "cache revision mismatch: $relative" >&2
    exit 1
  fi
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "cache SHA-256 mismatch: $relative" >&2
    exit 1
  fi
  echo "verified index=$index/$FILE_COUNT file=$relative sha256=$actual_sha256"
done

echo "FineWeb pretraining cache complete files=$FILE_COUNT config=$CONFIG_NAME"
printf 'revision=%s\nconfig=%s\nfiles=%d\n' \
  "$REVISION" "$CONFIG_NAME" "$FILE_COUNT" >"$COMPLETED"
