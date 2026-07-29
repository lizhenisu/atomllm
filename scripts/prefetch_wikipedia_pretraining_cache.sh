#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dlz/projects/atomllm-private}"
REVISION="b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-1}"
CACHE_ROOT="artifacts/source-cache/huggingface/wikimedia/wikipedia/$REVISION"
COMPLETED="$CACHE_ROOT/PREFETCH-EN-ZH-COMPLETED"
EN_COMPLETED="$CACHE_ROOT/PREFETCH-EN-COMPLETED"
ZH_COMPLETED="$CACHE_ROOT/PREFETCH-ZH-COMPLETED"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export HF_HUB_DOWNLOAD_TIMEOUT=600

if [[ ! "$DOWNLOAD_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DOWNLOAD_WORKERS must be a positive integer" >&2
  exit 1
fi

resolve_files() {
  local config="$1"
  REVISION="$REVISION" CONFIG="$config" python - <<'PY'
import os

from datasets import load_dataset_builder

revision = os.environ["REVISION"]
prefix = f"hf://datasets/wikimedia/wikipedia@{revision}/"
builder = load_dataset_builder(
    "wikimedia/wikipedia",
    os.environ["CONFIG"],
    revision=revision,
    token=True,
)
for path in builder.config.data_files["train"]:
    value = str(path)
    if not value.startswith(prefix):
        raise SystemExit(f"unexpected Hub path: {value}")
    print(value.removeprefix(prefix))
PY
}

mapfile -t zh_files < <(resolve_files "20231101.zh")
mapfile -t en_files < <(resolve_files "20231101.en")

if [[ "${#zh_files[@]}" -ne 6 || "${#en_files[@]}" -ne 41 ]]; then
  echo "expected 6 zh and 41 en frozen Wikipedia files" >&2
  exit 1
fi

download_and_verify() {
  local label="$1"
  local marker="$2"
  local array_name="$3"
  local -n selected_files="$array_name"
  local attempt=0
  until hf download wikimedia/wikipedia "${selected_files[@]}" \
    --repo-type dataset \
    --revision "$REVISION" \
    --local-dir "$CACHE_ROOT" \
    --max-workers "$DOWNLOAD_WORKERS"; do
    attempt=$((attempt + 1))
    delay=$((attempt * 5))
    if ((delay > 60)); then
      delay=60
    fi
    echo "$label download retry attempt=$attempt delay=$delay" >&2
    sleep "$delay"
  done

  for relative in "${selected_files[@]}"; do
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
    echo "verified label=$label file=$relative sha256=$actual_sha256"
  done
  printf 'revision=%s\nlabel=%s\nfiles=%d\n' \
    "$REVISION" "$label" "${#selected_files[@]}" >"$marker"
  echo "Wikipedia pretraining cache complete label=$label files=${#selected_files[@]}"
}

download_and_verify zh "$ZH_COMPLETED" zh_files
download_and_verify en "$EN_COMPLETED" en_files

printf 'revision=%s\nfiles=%d\n' \
  "$REVISION" "$((${#zh_files[@]} + ${#en_files[@]}))" >"$COMPLETED"
echo "Wikipedia pretraining cache complete files=47"
