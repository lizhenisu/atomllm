#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dlz/projects/atomllm-private}"
REVISION="1721eecd696e4110d33a255440f3c7ce981140ee"
CACHE_ROOT="artifacts/source-cache/huggingface/BAAI/IndustryCorpus2/$REVISION"
COMPLETED="$CACHE_ROOT/PREFETCH-ZH-HIGH-COMPLETED"
FILE_LIST="$CACHE_ROOT/prefetch-zh-high.files"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export HF_ENDPOINT=https://huggingface.co
mkdir -p "$CACHE_ROOT"

HF_REVISION="$REVISION" python - <<'PY' >"$FILE_LIST.tmp"
import fnmatch
import os
from huggingface_hub import HfApi

files = sorted(
    path for path in HfApi().list_repo_files(
        "BAAI/IndustryCorpus2", repo_type="dataset", revision=os.environ["HF_REVISION"], token=True
    ) if fnmatch.fnmatch(path, "*/chinese/high/rank_*.parquet")
)
if not files:
    raise SystemExit("no IndustryCorpus2 high-quality Chinese files resolved")
print("\n".join(files))
PY
mv "$FILE_LIST.tmp" "$FILE_LIST"

REPO_ID=BAAI/IndustryCorpus2 REVISION="$REVISION" CACHE_ROOT="$CACHE_ROOT" \
  FILE_LIST="$FILE_LIST" COMPLETED_MARKER="$COMPLETED" \
  scripts/prefetch_hf_file_list.sh
