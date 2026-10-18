#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the final 3-seed training + validation evaluation for one arm (B or C),
# using the shared config chosen from the 4 pilot runs: lr=1e-5, batch 16,
# AdamW wd=0.01, warmup 10%, TripletLoss margin 0.2 (cosine), 3 epochs.
set -euo pipefail

VARIANT="${1:?Usage: run_final.sh <B|C> <training-data-jsonl>}"
TRAINING_DATA="${2:?Usage: run_final.sh <B|C> <training-data-jsonl>}"

[ "$VARIANT" = "B" ] || [ "$VARIANT" = "C" ] || { echo "Variant must be B or C"; exit 1; }

cd "$HOME/scientific-retrieval-finetuning"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CACHE_DIR="$HOME/scientific-retrieval-artifacts/hf-cache"
TOKEN_AUDIT="$HOME/scientific-retrieval-evidence/token-audit-cache-fix-20261017T225329Z.gAjsX3"
DATA_AUDIT="$HOME/scientific-retrieval-evidence/data-audit-20261017T214239Z.wuW5KT"
SPLIT_LOCK="projects/scientific-retrieval/split-lock.json"
TEXT_POLICY="projects/scientific-retrieval/text-policy.json"

SEEDS=(13 42 2026)

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="$HOME/scientific-retrieval-artifacts/final-${VARIANT}-seed${SEED}"
  echo "=== ${VARIANT} seed=${SEED}: training ==="
  .venv/bin/python projects/scientific-retrieval/finetune.py \
    --variant "$VARIANT" \
    --training-data "$TRAINING_DATA" \
    --cache-dir "$CACHE_DIR" \
    --token-audit "$TOKEN_AUDIT" \
    --output-dir "$RUN_DIR" \
    --seed "$SEED" \
    --epochs 3 \
    --learning-rate 1e-5

  echo "=== ${VARIANT} seed=${SEED}: validation evaluation ==="
  .venv/bin/python projects/scientific-retrieval/evaluate_finetuned.py \
    --label "final-${VARIANT}-seed${SEED}" \
    --model-dir "$RUN_DIR/model" \
    --split-lock "$SPLIT_LOCK" \
    --text-policy "$TEXT_POLICY" \
    --data-audit "$DATA_AUDIT" \
    --output-dir "$RUN_DIR/eval"
  echo
done

echo "=== Summary: ${VARIANT}, lr=1e-5, 3 epochs, seeds ${SEEDS[*]} ==="
for SEED in "${SEEDS[@]}"; do
  .venv/bin/python -c "
import json
d = json.load(open('$HOME/scientific-retrieval-artifacts/final-${VARIANT}-seed${SEED}/eval/summary.json'))
print('seed ${SEED}:', 'nDCG@10=%.4f' % d['metrics']['nDCG@10'], 'Recall@10=%.4f' % d['metrics']['Recall@10'])
"
done
