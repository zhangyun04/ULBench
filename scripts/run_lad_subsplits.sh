#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="scripts/model_list_3day.txt"
RESULTS_DIR="experiments/results"
NUM_GPUS=8
IMAGE_ROOT="data/LAD"

SPLIT_ORDER=(
  lad_habitat_rk3_s42
  lad_habitat_ground
  lad_behaviour_fly
  lad_behaviour_swim
  lad_shape_rk3_s42
  lad_shape_long
  lad_size_big
  lad_size_small
)

TOTAL_SPLITS=${#SPLIT_ORDER[@]}
DONE=0
FAILED=0

echo "========================================================"
echo "  LAD sub-splits: ${TOTAL_SPLITS} splits × all models (skip_existing)"
echo "  Models : $MODELS"
echo "  Results: $RESULTS_DIR"
echo "  GPUs   : $NUM_GPUS"
echo "  Started: $(date)"
echo "========================================================"

for split_name in "${SPLIT_ORDER[@]}"; do
  split_dir="experiments/splits/$split_name"

  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  Split: $split_name"
  echo "╚══════════════════════════════════════════════════════╝"

  if bash scripts/run_batch_eval.sh \
      --split_dir  "$split_dir" \
      --image_root "$IMAGE_ROOT" \
      --models     "$MODELS" \
      --results_dir "$RESULTS_DIR" \
      --size_tier  small \
      --num_gpus   $NUM_GPUS \
      --skip_existing; then
    DONE=$((DONE + 1))
  else
    FAILED=$((FAILED + 1))
    echo "WARNING: Some models failed for $split_name"
  fi
done

echo ""
echo "========================================================"
echo "  Pipeline Complete: $(date)"
echo "  Splits done   : $DONE / $TOTAL_SPLITS"
echo "  Splits failed : $FAILED"
echo "========================================================"
