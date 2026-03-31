#!/usr/bin/env bash
# =============================================================================
# run_remaining_splits.sh
# Runs the 3 remaining splits (mit_indoor67, lad_color, spatialmqa) only.
# Skips coco and aid (already have results).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # always run from /ULBench

MODELS="scripts/model_list_3day.txt"
RESULTS_DIR="experiments/results"
NUM_GPUS=8

declare -A SPLITS=(
  ["mit_indoor67_rk10_s42"]="data/mit_indoor67"
  ["lad_color_rk3_s42"]="data/LAD"
  ["spatialmqa_rk3_s42"]="data/coco"
)

SPLIT_ORDER=(
  "mit_indoor67_rk10_s42"
  "lad_color_rk3_s42"
  "spatialmqa_rk3_s42"
)

TOTAL_SPLITS=${#SPLIT_ORDER[@]}
DONE=0
FAILED=0

echo "========================================================"
echo "  Remaining-splits pipeline: ${TOTAL_SPLITS} splits × 14 models"
echo "  Models : $MODELS"
echo "  Results: $RESULTS_DIR"
echo "  GPUs   : $NUM_GPUS"
echo "  Started: $(date)"
echo "========================================================"

for split_name in "${SPLIT_ORDER[@]}"; do
  image_root="${SPLITS[$split_name]}"
  split_dir="experiments/splits/$split_name"

  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  Split: $split_name"
  echo "║  Image: $image_root"
  echo "╚══════════════════════════════════════════════════════╝"

  if bash scripts/run_batch_eval.sh \
      --split_dir  "$split_dir" \
      --image_root "$image_root" \
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
