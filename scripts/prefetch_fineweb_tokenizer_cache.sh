#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dlz/projects/atomllm-private}"
REVISION="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
CACHE_BASE="artifacts/source-cache/huggingface"
CACHE_ROOT="$CACHE_BASE/HuggingFaceFW/fineweb-edu/$REVISION"
CORPUS_SESSION="tokenizer-corpus-en-zh-v1"

FILES=(
  sample/100BT/000_00005.parquet
  sample/100BT/000_00006.parquet
  sample/100BT/000_00007.parquet
  sample/100BT/000_00008.parquet
  sample/100BT/000_00009.parquet
  sample/100BT/001_00000.parquet
  sample/100BT/001_00001.parquet
  sample/100BT/001_00002.parquet
  sample/100BT/001_00003.parquet
  sample/100BT/001_00004.parquet
)

cd "$PROJECT_ROOT"
source .venv/bin/activate

corpus_pid="$(pgrep -f '^python -m atomllm.data.public_tokenizer_corpus' | sed -n '1p')"
if [[ -z "$corpus_pid" ]]; then
  echo "tokenizer corpus process is not running" >&2
  exit 1
fi

resume_corpus() {
  kill -CONT "$corpus_pid" 2>/dev/null || true
}
trap resume_corpus EXIT
kill -STOP "$corpus_pid"

export HF_HUB_DOWNLOAD_TIMEOUT=600
hf download HuggingFaceFW/fineweb-edu "${FILES[@]}" \
  --repo-type dataset \
  --revision "$REVISION" \
  --local-dir "$CACHE_ROOT" \
  --max-workers 1

for relative in "${FILES[@]}"; do
  local_file="$CACHE_ROOT/$relative"
  metadata="$CACHE_ROOT/.cache/huggingface/download/$relative.metadata"
  if [[ ! -f "$local_file" || ! -f "$metadata" ]]; then
    echo "incomplete cache entry: $relative" >&2
    exit 1
  fi
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
  echo "verified $relative $actual_sha256"
done

tmux respawn-pane -k -t "$CORPUS_SESSION:0.0" \
  "bash -lc 'set -o pipefail; cd $PROJECT_ROOT && source .venv/bin/activate && export HF_ENDPOINT=https://huggingface.co && export HF_HUB_DOWNLOAD_TIMEOUT=120 && export ATOMLLM_HF_SOURCE_CACHE=$CACHE_BASE && python -m atomllm.data.public_tokenizer_corpus --classification-workers 32 --maximum-source-restarts 1000 2>&1 | tee -a artifacts/logs/tokenizer-corpus-en-zh-v1.log'"
trap - EXIT

echo "prefetch complete; tokenizer corpus pane restarted with verified cache"
