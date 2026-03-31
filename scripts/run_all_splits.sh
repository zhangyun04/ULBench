#!/usr/bin/env bash
# Runs all splits with skip_existing.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="scripts/model_list_3day.txt"
RESULTS_DIR="experiments/results"
NUM_GPUS=8

declare -A SPLITS=(
  ["coco_rk10_s42"]="data/coco"
  ["coco_sbk10_s42"]="data/coco"
  ["aid_rk5_s42"]="data/AID"
  ["mit_indoor67_rk10_s42"]="data/mit_indoor67"
  ["lad_color_rk3_s42"]="data/LAD"
  ["lad_colour_black"]="data/LAD"
  ["lad_color_black"]="data/LAD"
  ["lad_color_brown"]="data/LAD"
  ["lad_habitat_rk3_s42"]="data/LAD"
  ["lad_habitat_ground"]="data/LAD"
  ["lad_shape_rk3_s42"]="data/LAD"
  ["lad_shape_long"]="data/LAD"
  ["lad_size_big"]="data/LAD"
  ["lad_size_small"]="data/LAD"
  ["lad_behaviour_fly"]="data/LAD"
  ["lad_behaviour_swim"]="data/LAD"
  ["logo2kplus_rk10_s42"]="data/LOGO-2K+/Logo-2K+"
  ["celebrity_rk5_s42"]="data/celebrity_faces/Celebrity Faces Dataset"
  ["spatialmqa_rk3_s42"]="data/spatialmqa_images"
  ["spatialmqa_leftof"]="data/spatialmqa_images"
)

# Only run splits that actually exist
SPLIT_ORDER=()
for s in coco_rk10_s42 coco_sbk10_s42 aid_rk5_s42 mit_indoor67_rk10_s42 \
          lad_color_rk3_s42 lad_color_black lad_color_brown \
          lad_habitat_rk3_s42 lad_habitat_ground \
          lad_shape_rk3_s42 lad_shape_long \
          lad_size_big lad_size_small \
          lad_behaviour_fly lad_behaviour_swim \
          logo2kplus_rk10_s42 celebrity_rk5_s42 \
          spatialmqa_rk3_s42 spatialmqa_leftof; do
  if [ -d "experiments/splits/$s" ] && [ -n "${SPLITS[$s]+x}" ]; then
    SPLIT_ORDER+=("$s")
  fi
done

TOTAL_SPLITS=${#SPLIT_ORDER[@]}
DONE=0
FAILED=0

echo "========================================================"
echo "  Full pipeline: ${TOTAL_SPLITS} splits × all models (skip_existing)"
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
