#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One-time official BeIR/SciFact TEST evaluation for A (baseline) and all
# B/C final seeds. Opened only after the B/C protocol was frozen from pilot +
# validation results. No training, no tuning after this.
set -euo pipefail

cd "$HOME/scientific-retrieval-finetuning"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DATA_AUDIT="$HOME/scientific-retrieval-evidence/data-audit-20261017T214239Z.wuW5KT"
TEST_OPEN="$HOME/scientific-retrieval-evidence/official-test-open-20261018T210000Z.claude01"
SPLIT_LOCK="projects/scientific-retrieval/split-lock.json"
BASELINE_SNAPSHOT="$HOME/scientific-retrieval-artifacts/hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

declare -a LABELS=("A-original" "B-seed13" "B-seed42" "B-seed2026" "C-seed13" "C-seed42" "C-seed2026")
declare -a MODEL_DIRS=(
  "$BASELINE_SNAPSHOT"
  "$HOME/scientific-retrieval-artifacts/final-B-seed13/model"
  "$HOME/scientific-retrieval-artifacts/final-B-seed42/model"
  "$HOME/scientific-retrieval-artifacts/final-B-seed2026/model"
  "$HOME/scientific-retrieval-artifacts/final-C-seed13/model"
  "$HOME/scientific-retrieval-artifacts/final-C-seed42/model"
  "$HOME/scientific-retrieval-artifacts/final-C-seed2026/model"
)

for i in "${!LABELS[@]}"; do
  LABEL="${LABELS[$i]}"
  MODEL_DIR="${MODEL_DIRS[$i]}"
  echo "=== official test: ${LABEL} ==="
  .venv/bin/python projects/scientific-retrieval/evaluate_official_test.py \
    --label "$LABEL" \
    --model-dir "$MODEL_DIR" \
    --split-lock "$SPLIT_LOCK" \
    --data-audit "$DATA_AUDIT" \
    --test-open "$TEST_OPEN" \
    --output-dir "$HOME/scientific-retrieval-artifacts/official-test-${LABEL}"
  echo
done

echo "=== Official TEST summary (300 queries, 339 qrels; single open, no tuning after) ==="
for LABEL in "${LABELS[@]}"; do
  .venv/bin/python -c "
import json
d = json.load(open('$HOME/scientific-retrieval-artifacts/official-test-${LABEL}/summary.json'))
print('${LABEL}:', 'nDCG@10=%.4f' % d['metrics']['nDCG@10'], 'Recall@10=%.4f' % d['metrics']['Recall@10'])
"
done
