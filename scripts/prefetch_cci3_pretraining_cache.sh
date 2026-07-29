#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dlz/projects/atomllm-private}"
REVISION="d6f3aa30cebfef497e822ff968ed68a18bf90b8f"
# The committed iterator is already inside file 436. Earlier files will never
# be replayed unless the committed shard state is deliberately removed.
START_INDEX="${START_INDEX:-436}"
CACHE_ROOT="artifacts/source-cache/huggingface/BAAI/CCI3-HQ/$REVISION"
COMPLETED="$CACHE_ROOT/PREFETCH-FROM-${START_INDEX}-COMPLETED"
FILE_LIST="$CACHE_ROOT/prefetch-from-${START_INDEX}.files"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export HF_ENDPOINT=https://huggingface.co
mkdir -p "$CACHE_ROOT"

HF_REVISION="$REVISION" START_INDEX="$START_INDEX" python - <<'PY' >"$FILE_LIST.tmp"
import os
from huggingface_hub import HfApi

files = sorted(
    path for path in HfApi().list_repo_files(
        "BAAI/CCI3-HQ", repo_type="dataset", revision=os.environ["HF_REVISION"], token=True
    ) if path.startswith("data/") and path.endswith(".jsonl")
)
start = int(os.environ["START_INDEX"])
if not 0 <= start < len(files):
    raise SystemExit(f"START_INDEX must be in [0, {len(files) - 1}]")
print("\n".join(files[start:]))
PY
mv "$FILE_LIST.tmp" "$FILE_LIST"

REPO_ID=BAAI/CCI3-HQ REVISION="$REVISION" CACHE_ROOT="$CACHE_ROOT" \
  FILE_LIST="$FILE_LIST" COMPLETED_MARKER="$COMPLETED" \
  scripts/prefetch_hf_file_list.sh
