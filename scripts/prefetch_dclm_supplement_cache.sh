#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dlz/projects/atomllm-private}"
REPO_ID="mlfoundations/dclm-baseline-1.0-parquet"
REVISION="817d6752765f6a41261085171dd546b104f60626"
PATTERN="filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/global-shard_01_of_10/local-shard_0_of_10/*.parquet"
CACHE_ROOT="artifacts/source-cache/huggingface/$REPO_ID/$REVISION"
FILE_LIST="$CACHE_ROOT/prefetch-dclm-supplement.files"
COMPLETED="$CACHE_ROOT/PREFETCH-DCLM-SUPPLEMENT-COMPLETED"
EXPECTED_FILES=279

cd "$PROJECT_ROOT"
source .venv/bin/activate
export HF_ENDPOINT=https://huggingface.co
mkdir -p "$CACHE_ROOT"

actual_files=0
if [[ -f "$FILE_LIST" ]]; then
  actual_files="$(wc -l <"$FILE_LIST")"
fi

if [[ "$actual_files" -ne "$EXPECTED_FILES" ]]; then
  HF_REPO_ID="$REPO_ID" HF_REVISION="$REVISION" HF_PATTERN="$PATTERN" \
    python - <<'PY' >"$FILE_LIST.tmp"
import fnmatch
import os
from huggingface_hub import HfApi

files = sorted(
    path
    for path in HfApi().list_repo_files(
        os.environ["HF_REPO_ID"],
        repo_type="dataset",
        revision=os.environ["HF_REVISION"],
        token=os.environ.get("HF_TOKEN") or None,
    )
    if fnmatch.fnmatch(path, os.environ["HF_PATTERN"])
)
print("\n".join(files))
PY
  mv "$FILE_LIST.tmp" "$FILE_LIST"
  actual_files="$(wc -l <"$FILE_LIST")"
fi

if [[ "$actual_files" -ne "$EXPECTED_FILES" ]]; then
  echo "DCLM file count changed: expected=$EXPECTED_FILES actual=$actual_files" >&2
  exit 1
fi

REPO_ID="$REPO_ID" REVISION="$REVISION" CACHE_ROOT="$CACHE_ROOT" \
  FILE_LIST="$FILE_LIST" COMPLETED_MARKER="$COMPLETED" \
  scripts/prefetch_hf_file_list.sh
