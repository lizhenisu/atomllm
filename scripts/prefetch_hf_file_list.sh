#!/usr/bin/env bash
set -Eeuo pipefail

: "${REPO_ID:?REPO_ID is required}"
: "${REVISION:?REVISION is required}"
: "${CACHE_ROOT:?CACHE_ROOT is required}"
: "${FILE_LIST:?FILE_LIST is required}"
: "${COMPLETED_MARKER:?COMPLETED_MARKER is required}"

MAX_ATTEMPTS_PER_FILE="${MAX_ATTEMPTS_PER_FILE:-1000}"
HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export HF_HUB_DISABLE_XET=1

mapfile -t files <"$FILE_LIST"
if ((${#files[@]} == 0)); then
  echo "file list is empty: $FILE_LIST" >&2
  exit 1
fi

download_file() {
  local index="$1"
  local relative="$2"
  local attempt=0
  local delay
  while true; do
    if hf download "$REPO_ID" "$relative" \
      --repo-type dataset --revision "$REVISION" \
      --local-dir "$CACHE_ROOT" --max-workers "$HF_DOWNLOAD_WORKERS"; then
      local_file="$CACHE_ROOT/$relative"
      metadata="$CACHE_ROOT/.cache/huggingface/download/$(dirname "$relative")/$(basename "$relative").metadata"
      if [[ -f "$local_file" && -f "$metadata" ]]; then
        metadata_revision="$(sed -n '1p' "$metadata")"
        expected_sha256="$(sed -n '2p' "$metadata")"
        actual_sha256="$(sha256sum "$local_file" | cut -d ' ' -f 1)"
        if [[ "$metadata_revision" == "$REVISION" && "$actual_sha256" == "$expected_sha256" ]]; then
          echo "verified index=$index/${#files[@]} file=$relative sha256=$actual_sha256"
          return
        fi
      fi
      echo "download verification failed index=$index/${#files[@]} file=$relative" >&2
    fi
    attempt=$((attempt + 1))
    if ((attempt >= MAX_ATTEMPTS_PER_FILE)); then
      echo "download failed index=$index/${#files[@]} file=$relative attempts=$attempt" >&2
      return 1
    fi
    delay=$((attempt * 5))
    ((delay > 60)) && delay=60
    echo "download retry index=$index/${#files[@]} file=$relative attempt=$attempt delay=$delay" >&2
    sleep "$delay"
  done
}

for index in "${!files[@]}"; do
  download_file "$index" "${files[$index]}"
done

mkdir -p "$(dirname "$COMPLETED_MARKER")"
printf 'revision=%s\nfiles=%d\n' "$REVISION" "${#files[@]}" >"$COMPLETED_MARKER"
