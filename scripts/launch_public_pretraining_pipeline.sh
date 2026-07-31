#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/dlz/projects/atomllm-private}"
SOURCE_CACHE="artifacts/source-cache/huggingface"

stage_is_complete() {
  local name="$1"
  local marker=""
  case "$name" in
    tokenizer-corpus-en-zh-v1)
      marker=artifacts/tokenizer-data/atom-public-tokenizer-corpus-en-zh-v1/COMPLETED ;;
    tokenizer-corpus-audit-en-zh-v1)
      marker=artifacts/data-audits/atom-public-tokenizer-corpus-en-zh-v1/COMPLETED ;;
    tokenizer-snapshot-en-zh-084pct-h100-v2)
      marker=artifacts/tokenizer-snapshots/atom-public-en-zh-084pct-h100-v2/COMPLETED ;;
    tokenizer-train-en-zh-084pct-h100-v2-32k-v1)
      marker=artifacts/tokenizer-training-reports/atom-public-en-zh-084pct-h100-v2-32k-v1.json
      [[ -f artifacts/tokenizers/atom-tokenizer-en-zh-084pct-h100-v2-32k-v1/COMPLETED ]] || return 1
      ;;
    tokenizer-eval-en-zh-084pct-h100-v2-32k-v1)
      marker=artifacts/tokenizer-evaluations/atom-public-en-zh-084pct-h100-v2-32k-v1/COMPLETED ;;
    tokenizer-train-en-zh-084pct-h100-v2-48k-v1)
      marker=artifacts/tokenizer-training-reports/atom-public-en-zh-084pct-h100-v2-48k-v1.json
      [[ -f artifacts/tokenizers/atom-tokenizer-en-zh-084pct-h100-v2-48k-v1/COMPLETED ]] || return 1
      ;;
    tokenizer-eval-en-zh-084pct-h100-v2-48k-v1)
      marker=artifacts/tokenizer-evaluations/atom-public-en-zh-084pct-h100-v2-48k-v1/COMPLETED ;;
    tokenizer-select-en-zh-084pct-h100-v2)
      marker=artifacts/tokenizer-selections/atom-public-en-zh-084pct-h100-v2/COMPLETED ;;
    tokenizer-gpu-benchmark-32k-v1)
      marker=artifacts/tokenizer-gpu-benchmarks/atom-public-en-zh-084pct-h100-v2-32k-v1/COMPLETED ;;
    tokenizer-gpu-benchmark-48k-v1)
      marker=artifacts/tokenizer-gpu-benchmarks/atom-public-en-zh-084pct-h100-v2-48k-v1/COMPLETED ;;
    tokenizer-select-gpu-en-zh-084pct-h100-v2)
      marker=artifacts/tokenizer-selections/atom-public-en-zh-084pct-h100-v2-gpu-v1/COMPLETED ;;
    prefetch-wikipedia-en-zh-v1)
      marker=artifacts/source-cache/huggingface/wikimedia/wikipedia/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/PREFETCH-EN-ZH-COMPLETED ;;
    prefetch-cci3-from436-v1)
      marker=artifacts/source-cache/huggingface/BAAI/CCI3-HQ/d6f3aa30cebfef497e822ff968ed68a18bf90b8f/PREFETCH-FROM-436-COMPLETED ;;
    prefetch-industry-zh-high-v1)
      marker=artifacts/source-cache/huggingface/BAAI/IndustryCorpus2/1721eecd696e4110d33a255440f3c7ce981140ee/PREFETCH-ZH-HIGH-COMPLETED ;;
    prefetch-fineweb-350bt-all472-v3)
      marker=artifacts/source-cache/huggingface/HuggingFaceFW/fineweb-edu/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/PREFETCH-ALL472-COMPLETED ;;
    public-plan-extension-dclm-v3)
      marker=artifacts/training-data/public-token-groups-100b-v2/_plan-migrations/0131b1ef5ffb-to-0fe514ee8ac9/receipt.json ;;
    prefetch-dclm-supplement-v1)
      marker=artifacts/source-cache/huggingface/mlfoundations/dclm-baseline-1.0-parquet/817d6752765f6a41261085171dd546b104f60626/PREFETCH-DCLM-SUPPLEMENT-COMPLETED ;;
    public-token-shards-en-100b-v2)
      marker=artifacts/training-data/public-token-groups-100b-v2/en/COMPLETED ;;
    public-token-shards-code-100b-v2)
      marker=artifacts/training-data/public-token-groups-100b-v2/code/COMPLETED ;;
    public-token-shards-zh-100b-v2)
      marker=artifacts/training-data/public-token-groups-100b-v2/zh-Hans/COMPLETED ;;
    public-token-shards-assemble-100b-v2)
      marker=artifacts/training-data/public-token-shards-100b-v2/COMPLETED ;;
    public-pretraining-release-100b-v2)
      marker=artifacts/training-releases/atom-base-300m-public-100b-v2/COMPLETED ;;
    public-pretraining-smoke-30-v2)
      marker=artifacts/training-smoke-gates/atom-base-300m-public-100b-v2/COMPLETED ;;
    public-pretraining-100b-v2)
      marker=artifacts/training-runs/atom-base-300m-public-100b-v2/COMPLETED ;;
    public-base-evaluation-full-v4)
      marker=artifacts/evaluations/atom-base-300m-public-100b-full-v4/COMPLETED ;;
    *) return 1 ;;
  esac
  [[ -f "$marker" ]] || return 1
  case "$name" in
    tokenizer-train-en-zh-084pct-h100-v2-32k-v1|tokenizer-train-en-zh-084pct-h100-v2-48k-v1)
      "$PROJECT_ROOT/.venv/bin/python" -c 'import json,sys; report=json.load(open(sys.argv[1])); raise SystemExit(0 if report.get("return_code") == 0 and report.get("memory_limit_exceeded") is False else 1)' "$marker"
      ;;
  esac
}

ensure_session() {
  local name="$1"
  local command="$2"
  local encoded
  if stage_is_complete "$name"; then
    return
  fi
  if tmux has-session -t "$name" 2>/dev/null; then
    return
  fi
  encoded="$(printf '%s' "$command" | base64 -w 0)"
  tmux new-session -d -s "$name" \
    "printf '%s' '$encoded' | base64 --decode | /bin/bash" 8>&-
  echo "started $name"
}

cd "$PROJECT_ROOT"
mkdir -p artifacts/locks artifacts/logs
exec 8>artifacts/locks/public-pretraining-reconcile-v2.lock
flock 8

if [[ ! -f artifacts/tokenizer-data/atom-public-tokenizer-corpus-en-zh-v1/COMPLETED ]]; then
  ensure_session tokenizer-corpus-en-zh-v1 "
    set -o pipefail
    cd '$PROJECT_ROOT'
    source .venv/bin/activate
    export HF_ENDPOINT=https://huggingface.co
    export HF_HUB_DOWNLOAD_TIMEOUT=600
    export HF_HUB_ETAG_TIMEOUT=120
    export ATOMLLM_HF_SOURCE_CACHE='$SOURCE_CACHE'
    python -m atomllm.data.public_tokenizer_corpus \
      --classification-workers 32 --maximum-source-restarts 1000 \
      2>&1 | tee -a artifacts/logs/tokenizer-corpus-en-zh-v1.log
  "
fi

ensure_session tokenizer-corpus-audit-en-zh-v1 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/tokenizer-data/atom-public-tokenizer-corpus-en-zh-v1/COMPLETED ]]; do
    tmux has-session -t tokenizer-corpus-en-zh-v1 2>/dev/null || exit 1
    sleep 30
  done
  python -m atomllm.data.public_corpus_audit \
    --corpus-dir artifacts/tokenizer-data/atom-public-tokenizer-corpus-en-zh-v1 \
    --output-dir artifacts/data-audits/atom-public-tokenizer-corpus-en-zh-v1 \
    --workers 64 2>&1 | tee artifacts/logs/tokenizer-corpus-audit-en-zh-v1.log
"

ensure_session tokenizer-snapshot-en-zh-084pct-h100-v2 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/data-audits/atom-public-tokenizer-corpus-en-zh-v1/COMPLETED ]]; do
    tmux has-session -t tokenizer-corpus-audit-en-zh-v1 2>/dev/null || exit 1
    sleep 30
  done
  python -m atomllm.tokenizer.public_snapshot \
    --source-dir artifacts/tokenizer-data/atom-public-tokenizer-corpus-en-zh-v1 \
    --audit-dir artifacts/data-audits/atom-public-tokenizer-corpus-en-zh-v1 \
    --output-dir artifacts/tokenizer-snapshots/atom-public-en-zh-084pct-h100-v2 \
    --tokenizer-output-dir artifacts/tokenizers --sample-ratio 0.84 \
    --vocab-sizes 32000 48000 --heldout-documents-per-source 100 \
    --artifact-label 084pct-h100-v2 \
    2>&1 | tee artifacts/logs/tokenizer-snapshot-en-zh-084pct-h100-v2.log
"

ensure_session tokenizer-train-en-zh-084pct-h100-v2-32k-v1 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/tokenizer-snapshots/atom-public-en-zh-084pct-h100-v2/COMPLETED ]]; do
    tmux has-session -t tokenizer-snapshot-en-zh-084pct-h100-v2 2>/dev/null || exit 1
    sleep 30
  done
  python -m atomllm.tokenizer.training_supervisor \
    --config artifacts/tokenizer-snapshots/atom-public-en-zh-084pct-h100-v2/tokenizer-training-32k.yaml \
    --workers 256 --maximum-rss-gib 480 --poll-seconds 1 \
    --report artifacts/tokenizer-training-reports/atom-public-en-zh-084pct-h100-v2-32k-v1.json \
    2>&1 | tee artifacts/logs/tokenizer-train-en-zh-084pct-h100-v2-32k-v1.log
"

ensure_session tokenizer-eval-en-zh-084pct-h100-v2-32k-v1 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/tokenizers/atom-tokenizer-en-zh-084pct-h100-v2-32k-v1/COMPLETED ]]; do
    tmux has-session -t tokenizer-train-en-zh-084pct-h100-v2-32k-v1 2>/dev/null || exit 1
    sleep 30
  done
  python -m atomllm.tokenizer.public_heldout_evaluation \
    --tokenizer-dir artifacts/tokenizers/atom-tokenizer-en-zh-084pct-h100-v2-32k-v1 \
    --snapshot-dir artifacts/tokenizer-snapshots/atom-public-en-zh-084pct-h100-v2 \
    --output-dir artifacts/tokenizer-evaluations/atom-public-en-zh-084pct-h100-v2-32k-v1 \
    2>&1 | tee artifacts/logs/tokenizer-eval-en-zh-084pct-h100-v2-32k-v1.log
"

ensure_session tokenizer-train-en-zh-084pct-h100-v2-48k-v1 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/tokenizer-evaluations/atom-public-en-zh-084pct-h100-v2-32k-v1/COMPLETED ]]; do
    tmux has-session -t tokenizer-eval-en-zh-084pct-h100-v2-32k-v1 2>/dev/null || exit 1
    sleep 30
  done
  while [[ ! -f artifacts/tokenizer-training-reports/atom-public-en-zh-084pct-h100-v2-32k-v1.json ]]; do
    tmux has-session -t tokenizer-train-en-zh-084pct-h100-v2-32k-v1 2>/dev/null || exit 1
    sleep 10
  done
  python -m atomllm.tokenizer.training_report_gate \
    --report artifacts/tokenizer-training-reports/atom-public-en-zh-084pct-h100-v2-32k-v1.json \
    --minimum-rss-gib 440 --maximum-rss-gib 480 --expected-workers 256
  python -m atomllm.tokenizer.training_supervisor \
    --config artifacts/tokenizer-snapshots/atom-public-en-zh-084pct-h100-v2/tokenizer-training-48k.yaml \
    --workers 256 --maximum-rss-gib 480 --poll-seconds 1 \
    --report artifacts/tokenizer-training-reports/atom-public-en-zh-084pct-h100-v2-48k-v1.json \
    2>&1 | tee artifacts/logs/tokenizer-train-en-zh-084pct-h100-v2-48k-v1.log
"

ensure_session tokenizer-eval-en-zh-084pct-h100-v2-48k-v1 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/tokenizers/atom-tokenizer-en-zh-084pct-h100-v2-48k-v1/COMPLETED ]]; do
    tmux has-session -t tokenizer-train-en-zh-084pct-h100-v2-48k-v1 2>/dev/null || exit 1
    sleep 30
  done
  python -m atomllm.tokenizer.public_heldout_evaluation \
    --tokenizer-dir artifacts/tokenizers/atom-tokenizer-en-zh-084pct-h100-v2-48k-v1 \
    --snapshot-dir artifacts/tokenizer-snapshots/atom-public-en-zh-084pct-h100-v2 \
    --output-dir artifacts/tokenizer-evaluations/atom-public-en-zh-084pct-h100-v2-48k-v1 \
    2>&1 | tee artifacts/logs/tokenizer-eval-en-zh-084pct-h100-v2-48k-v1.log
"

ensure_session tokenizer-select-en-zh-084pct-h100-v2 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/tokenizer-evaluations/atom-public-en-zh-084pct-h100-v2-48k-v1/COMPLETED ]]; do
    tmux has-session -t tokenizer-eval-en-zh-084pct-h100-v2-48k-v1 2>/dev/null || exit 1
    sleep 30
  done
  # Tokenizer COMPLETED is written by the child before it exits, while the
  # supervisor writes the aggregate RSS report only after child termination.
  # Held-out evaluation can therefore finish before this required report.
  while [[ ! -f artifacts/tokenizer-training-reports/atom-public-en-zh-084pct-h100-v2-48k-v1.json ]]; do
    tmux has-session -t tokenizer-train-en-zh-084pct-h100-v2-48k-v1 2>/dev/null || exit 1
    sleep 10
  done
  python -m atomllm.tokenizer.training_report_gate \
    --report artifacts/tokenizer-training-reports/atom-public-en-zh-084pct-h100-v2-48k-v1.json \
    --minimum-rss-gib 440 --maximum-rss-gib 480 --expected-workers 256
  python -m atomllm.tokenizer.public_selection \
    --evaluation-32k-dir artifacts/tokenizer-evaluations/atom-public-en-zh-084pct-h100-v2-32k-v1 \
    --evaluation-48k-dir artifacts/tokenizer-evaluations/atom-public-en-zh-084pct-h100-v2-48k-v1 \
    --memory-32k-report artifacts/tokenizer-training-reports/atom-public-en-zh-084pct-h100-v2-32k-v1.json \
    --memory-48k-report artifacts/tokenizer-training-reports/atom-public-en-zh-084pct-h100-v2-48k-v1.json \
    --output-dir artifacts/tokenizer-selections/atom-public-en-zh-084pct-h100-v2 \
    --minimum-rss-gib 440 --maximum-rss-gib 480 \
    2>&1 | tee artifacts/logs/tokenizer-select-en-zh-084pct-h100-v2.log
"

ensure_session tokenizer-gpu-benchmark-32k-v1 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/tokenizer-selections/atom-public-en-zh-084pct-h100-v2/COMPLETED ]]; do
    tmux has-session -t tokenizer-select-en-zh-084pct-h100-v2 2>/dev/null || exit 1
    sleep 30
  done
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 \
    -m atomllm.tokenizer.gpu_benchmark \
    --tokenizer-dir artifacts/tokenizers/atom-tokenizer-en-zh-084pct-h100-v2-32k-v1 \
    --evaluation-dir artifacts/tokenizer-evaluations/atom-public-en-zh-084pct-h100-v2-32k-v1 \
    --snapshot-dir artifacts/tokenizer-snapshots/atom-public-en-zh-084pct-h100-v2 \
    --base-model-config configs/model/atom-base-300m.yaml \
    --output-dir artifacts/tokenizer-gpu-benchmarks/atom-public-en-zh-084pct-h100-v2-32k-v1 \
    2>&1 | tee artifacts/logs/tokenizer-gpu-benchmark-32k-v1.log
"

ensure_session tokenizer-gpu-benchmark-48k-v1 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/tokenizer-gpu-benchmarks/atom-public-en-zh-084pct-h100-v2-32k-v1/COMPLETED ]]; do
    tmux has-session -t tokenizer-gpu-benchmark-32k-v1 2>/dev/null || exit 1
    sleep 30
  done
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 \
    -m atomllm.tokenizer.gpu_benchmark \
    --tokenizer-dir artifacts/tokenizers/atom-tokenizer-en-zh-084pct-h100-v2-48k-v1 \
    --evaluation-dir artifacts/tokenizer-evaluations/atom-public-en-zh-084pct-h100-v2-48k-v1 \
    --snapshot-dir artifacts/tokenizer-snapshots/atom-public-en-zh-084pct-h100-v2 \
    --base-model-config configs/model/atom-base-300m.yaml \
    --output-dir artifacts/tokenizer-gpu-benchmarks/atom-public-en-zh-084pct-h100-v2-48k-v1 \
    2>&1 | tee artifacts/logs/tokenizer-gpu-benchmark-48k-v1.log
"

ensure_session tokenizer-select-gpu-en-zh-084pct-h100-v2 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/tokenizer-gpu-benchmarks/atom-public-en-zh-084pct-h100-v2-48k-v1/COMPLETED ]]; do
    tmux has-session -t tokenizer-gpu-benchmark-48k-v1 2>/dev/null || exit 1
    sleep 30
  done
  python -m atomllm.tokenizer.public_gpu_selection \
    --heldout-selection-dir artifacts/tokenizer-selections/atom-public-en-zh-084pct-h100-v2 \
    --evaluation-32k-dir artifacts/tokenizer-evaluations/atom-public-en-zh-084pct-h100-v2-32k-v1 \
    --evaluation-48k-dir artifacts/tokenizer-evaluations/atom-public-en-zh-084pct-h100-v2-48k-v1 \
    --gpu-32k-dir artifacts/tokenizer-gpu-benchmarks/atom-public-en-zh-084pct-h100-v2-32k-v1 \
    --gpu-48k-dir artifacts/tokenizer-gpu-benchmarks/atom-public-en-zh-084pct-h100-v2-48k-v1 \
    --tokenizer-32k-dir artifacts/tokenizers/atom-tokenizer-en-zh-084pct-h100-v2-32k-v1 \
    --tokenizer-48k-dir artifacts/tokenizers/atom-tokenizer-en-zh-084pct-h100-v2-48k-v1 \
    --output-dir artifacts/tokenizer-selections/atom-public-en-zh-084pct-h100-v2-gpu-v1 \
    2>&1 | tee artifacts/logs/tokenizer-select-gpu-en-zh-084pct-h100-v2.log
"

ensure_session prefetch-wikipedia-en-zh-v1 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  DOWNLOAD_WORKERS=4 scripts/prefetch_wikipedia_pretraining_cache.sh \
    2>&1 | tee -a artifacts/logs/prefetch-wikipedia-en-zh-v1.log
"

ensure_session prefetch-cci3-from436-v1 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  while [[ ! -f artifacts/source-cache/huggingface/wikimedia/wikipedia/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/PREFETCH-EN-ZH-COMPLETED ]]; do
    tmux has-session -t prefetch-wikipedia-en-zh-v1 2>/dev/null || exit 1
    sleep 30
  done
  scripts/prefetch_cci3_pretraining_cache.sh \
    2>&1 | tee -a artifacts/logs/prefetch-cci3-from436-v1.log
"

ensure_session prefetch-industry-zh-high-v1 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  while [[ ! -f artifacts/source-cache/huggingface/BAAI/CCI3-HQ/d6f3aa30cebfef497e822ff968ed68a18bf90b8f/PREFETCH-FROM-436-COMPLETED ]]; do
    tmux has-session -t prefetch-cci3-from436-v1 2>/dev/null || exit 1
    sleep 30
  done
  scripts/prefetch_industry_corpus2_pretraining_cache.sh \
    2>&1 | tee -a artifacts/logs/prefetch-industry-zh-high-v1.log
"

ensure_session prefetch-fineweb-350bt-all472-v3 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  while [[ ! -f artifacts/source-cache/huggingface/BAAI/IndustryCorpus2/1721eecd696e4110d33a255440f3c7ce981140ee/PREFETCH-ZH-HIGH-COMPLETED ]]; do
    tmux has-session -t prefetch-industry-zh-high-v1 2>/dev/null || exit 1
    sleep 30
  done
  DOWNLOAD_WORKERS=1 scripts/prefetch_fineweb_pretraining_cache.sh \
    2>&1 | tee -a artifacts/logs/prefetch-fineweb-350bt-all472-v3.log
"

ensure_session public-plan-extension-dclm-v3 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  python -m atomllm.training.public_plan_extension \
    --old-plan configs/data/public-pretraining-plan-100b-v2.yaml \
    --new-plan configs/data/public-pretraining-plan-100b-v3.yaml \
    --group-root artifacts/training-data/public-token-groups-100b-v2 \
    2>&1 | tee artifacts/logs/public-plan-extension-dclm-v3.log
"

ensure_session prefetch-dclm-supplement-v1 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  while [[ ! -f artifacts/training-data/public-token-groups-100b-v2/_plan-migrations/0131b1ef5ffb-to-0fe514ee8ac9/receipt.json ]]; do
    tmux has-session -t public-plan-extension-dclm-v3 2>/dev/null || exit 1
    sleep 10
  done
  exec bash -o pipefail -c \
    'scripts/prefetch_dclm_supplement_cache.sh 2>&1 | tee -a artifacts/logs/prefetch-dclm-supplement-v1.log'
"

for group in en code zh-Hans; do
  case "$group" in
    en)
      session=public-token-shards-en-100b-v2; workers=64; log=en
      cache_marker=artifacts/source-cache/huggingface/mlfoundations/dclm-baseline-1.0-parquet/817d6752765f6a41261085171dd546b104f60626/PREFETCH-DCLM-SUPPLEMENT-COMPLETED
      cache_session=prefetch-dclm-supplement-v1
      ;;
    code)
      session=public-token-shards-code-100b-v2; workers=32; log=code
      cache_marker=; cache_session=
      ;;
    zh-Hans)
      session=public-token-shards-zh-100b-v2; workers=32; log=zh
      cache_marker=artifacts/source-cache/huggingface/BAAI/IndustryCorpus2/1721eecd696e4110d33a255440f3c7ce981140ee/PREFETCH-ZH-HIGH-COMPLETED
      cache_session=prefetch-industry-zh-high-v1
      ;;
  esac
  ensure_session "$session" "
    set -o pipefail
    cd '$PROJECT_ROOT'
    source .venv/bin/activate
    export HF_ENDPOINT=https://huggingface.co
    export HF_HUB_DOWNLOAD_TIMEOUT=600
    export HF_HUB_ETAG_TIMEOUT=120
    export ATOMLLM_HF_SOURCE_CACHE='$SOURCE_CACHE'
    while [[ ! -f artifacts/tokenizer-selections/atom-public-en-zh-084pct-h100-v2-gpu-v1/COMPLETED ]]; do
      tmux has-session -t tokenizer-select-gpu-en-zh-084pct-h100-v2 2>/dev/null || exit 1
      sleep 30
    done
    if [[ -n '$cache_marker' ]]; then
      while [[ ! -f '$cache_marker' ]]; do
        tmux has-session -t '$cache_session' 2>/dev/null || exit 1
        sleep 30
      done
    fi
    python -m atomllm.training.public_token_shards \
      --plan configs/data/public-pretraining-plan-100b-v3.yaml \
      --tokenizer-selection-dir artifacts/tokenizer-selections/atom-public-en-zh-084pct-h100-v2-gpu-v1 \
      --output-root artifacts/training-data/public-token-groups-100b-v2 \
      --group '$group' --workers '$workers' --encode-batch-size 1024 \
      --maximum-source-restarts 1000 \
      2>&1 | tee -a artifacts/logs/public-token-shards-$log-100b-v2.log
  "
done

ensure_session public-token-shards-assemble-100b-v2 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/training-data/public-token-groups-100b-v2/en/COMPLETED ]]; do
    tmux has-session -t public-token-shards-en-100b-v2 2>/dev/null || exit 1
    sleep 60
  done
  while [[ ! -f artifacts/training-data/public-token-groups-100b-v2/code/COMPLETED ]]; do
    tmux has-session -t public-token-shards-code-100b-v2 2>/dev/null || exit 1
    sleep 60
  done
  while [[ ! -f artifacts/training-data/public-token-groups-100b-v2/zh-Hans/COMPLETED ]]; do
    tmux has-session -t public-token-shards-zh-100b-v2 2>/dev/null || exit 1
    sleep 60
  done
  python -m atomllm.training.public_token_shards \
    --plan configs/data/public-pretraining-plan-100b-v3.yaml \
    --output-root artifacts/training-data/public-token-groups-100b-v2 \
    --assemble-output-dir artifacts/training-data/public-token-shards-100b-v2 \
    2>&1 | tee artifacts/logs/public-token-shards-assemble-100b-v2.log
"

ensure_session public-pretraining-release-100b-v2 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/training-data/public-token-shards-100b-v2/COMPLETED ]]; do
    tmux has-session -t public-token-shards-assemble-100b-v2 2>/dev/null || exit 1
    sleep 60
  done
  python -m atomllm.training.public_pretraining_release \
    --tokenizer-selection-dir artifacts/tokenizer-selections/atom-public-en-zh-084pct-h100-v2-gpu-v1 \
    --token-shards-dir artifacts/training-data/public-token-shards-100b-v2 \
    --base-model-config configs/model/atom-base-300m.yaml \
    --output-dir artifacts/training-releases/atom-base-300m-public-100b-v2 \
    2>&1 | tee artifacts/logs/public-pretraining-release-100b-v2.log
"

ensure_session public-pretraining-smoke-30-v2 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/training-releases/atom-base-300m-public-100b-v2/COMPLETED ]]; do
    tmux has-session -t public-pretraining-release-100b-v2 2>/dev/null || exit 1
    sleep 60
  done
  smoke_run=artifacts/training-runs/atom-base-300m-public-100b-smoke-v2
  resume_args=()
  if [[ -d \"\$smoke_run\" ]]; then
    if [[ ! -f \"\$smoke_run/checkpoints/latest.json\" ]]; then
      echo \"smoke run exists without a resumable checkpoint: \$smoke_run\" >&2
      exit 1
    fi
    resume_args=(--resume)
  fi
  ulimit -n 65536
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 \
    -m atomllm.training.trainer \
    --config artifacts/training-releases/atom-base-300m-public-100b-v2/training.yaml \
    --training-data artifacts/training-data/public-token-shards-100b-v2 \
    --steps 30 --output-dir artifacts/training-runs \
    --run-id atom-base-300m-public-100b-smoke-v2 --checkpoint-every 10 \
    \"\${resume_args[@]}\" \
    2>&1 | tee -a artifacts/logs/public-pretraining-smoke-30-v2.log
  python -m atomllm.training.public_smoke_gate \
    --run-dir artifacts/training-runs/atom-base-300m-public-100b-smoke-v2 \
    --training-config artifacts/training-releases/atom-base-300m-public-100b-v2/training.yaml \
    --output-dir artifacts/training-smoke-gates/atom-base-300m-public-100b-v2 \
    --expected-steps 30 --maximum-reserved-gib 23.5 \
    2>&1 | tee -a artifacts/logs/public-pretraining-smoke-30-v2.log
"

ensure_session public-pretraining-100b-v2 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/training-smoke-gates/atom-base-300m-public-100b-v2/COMPLETED ]]; do
    tmux has-session -t public-pretraining-smoke-30-v2 2>/dev/null || exit 1
    sleep 60
  done
  training_run=artifacts/training-runs/atom-base-300m-public-100b-v2
  resume_args=()
  if [[ -d \"\$training_run\" ]]; then
    if [[ ! -f \"\$training_run/checkpoints/latest.json\" ]]; then
      echo \"training run exists without a resumable checkpoint: \$training_run\" >&2
      exit 1
    fi
    resume_args=(--resume)
  fi
  ulimit -n 65536
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 \
    -m atomllm.training.trainer \
    --config artifacts/training-releases/atom-base-300m-public-100b-v2/training.yaml \
    --training-data artifacts/training-data/public-token-shards-100b-v2 \
    --output-dir artifacts/training-runs --run-id atom-base-300m-public-100b-v2 \
    \"\${resume_args[@]}\" \
    2>&1 | tee -a artifacts/logs/public-pretraining-100b-v2.log
"

ensure_session public-base-evaluation-full-v4 "
  set -o pipefail
  cd '$PROJECT_ROOT'
  source .venv/bin/activate
  while [[ ! -f artifacts/training-runs/atom-base-300m-public-100b-v2/COMPLETED ]]; do
    tmux has-session -t public-pretraining-100b-v2 2>/dev/null || exit 1
    sleep 60
  done
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 \
    -m atomllm.training.public_base_evaluation \
    --run-dir artifacts/training-runs/atom-base-300m-public-100b-v2 \
    --release-dir artifacts/training-releases/atom-base-300m-public-100b-v2 \
    --tokenizer-selection-dir artifacts/tokenizer-selections/atom-public-en-zh-084pct-h100-v2-gpu-v1 \
    --suite-dir artifacts/evaluation-data/atom-base-public-zero-shot-full-v3 \
    --output-dir artifacts/evaluations/atom-base-300m-public-100b-full-v4 \
    2>&1 | tee artifacts/logs/public-base-evaluation-full-v4.log
"

tmux list-sessions -F '#{session_name}' | sort
