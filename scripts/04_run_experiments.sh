#!/usr/bin/env bash
# =============================================================================
# 04_run_experiments.sh
# Run batch evaluation across ALL splits defined in split_registry.yaml.
#
# For each split directory under experiments/splits/, calls run_batch_eval.sh
# with the correct --image_root from the registry.
#
# Usage:
#   bash scripts/04_run_experiments.sh                          # all splits, all models
#   bash scripts/04_run_experiments.sh --size_tier small        # only small models
#   bash scripts/04_run_experiments.sh --only coco,aid          # only matching splits
#   bash scripts/04_run_experiments.sh --conditions BASELINE_NORMAL,ORACLE_HARD
#   bash scripts/04_run_experiments.sh --dry_run
#   bash scripts/04_run_experiments.sh --skip_existing
# =============================================================================
set -euo pipefail

REGISTRY="scripts/split_registry.yaml"
SIZE_TIER="all"
CONDITIONS="all"
MODELS_FILE="scripts/model_list.txt"
RESULTS_DIR="experiments/results"
MAX_SAMPLES=""
ONLY=""
DRY_RUN=false
SKIP_EXISTING=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry)       REGISTRY="$2";      shift 2 ;;
        --size_tier)      SIZE_TIER="$2";      shift 2 ;;
        --conditions)     CONDITIONS="$2";     shift 2 ;;
        --models)         MODELS_FILE="$2";    shift 2 ;;
        --results_dir)    RESULTS_DIR="$2";    shift 2 ;;
        --max_samples)    MAX_SAMPLES="$2";    shift 2 ;;
        --only)           ONLY="$2";           shift 2 ;;
        --dry_run)        DRY_RUN=true;        shift ;;
        --skip_existing)  SKIP_EXISTING=true;  shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo "  Full Experiment Pipeline"
echo "  Registry    : $REGISTRY"
echo "  Size tier   : $SIZE_TIER"
echo "  Conditions  : $CONDITIONS"
echo "  Models      : $MODELS_FILE"
echo "  Results dir : $RESULTS_DIR"
echo "  Filter      : ${ONLY:-all}"
echo "  Dry run     : $DRY_RUN"
echo "  Skip exist  : $SKIP_EXISTING"
echo "================================================================"
echo ""

# Parse the registry to get (split_name, image_root) pairs.
# We use Python to read the YAML — more robust than bash parsing.
SPLIT_ENTRIES=$(python3 -c "
import yaml, sys
with open('$REGISTRY') as f:
    reg = yaml.safe_load(f)
only = set('$ONLY'.split(',')) if '$ONLY' else None
for ds_key, ds_cfg in reg.get('datasets', {}).items():
    if only and ds_key not in only:
        # Also check if any experiment name matches
        exp_match = any(e['name'] in only for e in ds_cfg.get('experiments', []))
        if not exp_match:
            continue
    image_root = ds_cfg['image_root']
    for exp in ds_cfg.get('experiments', []):
        name = exp['name']
        if only and ds_key not in only and name not in only:
            continue
        print(f'{name}\t{image_root}')
")

if [[ -z "$SPLIT_ENTRIES" ]]; then
    echo "ERROR: No split entries found. Check registry and --only filter."
    exit 1
fi

TOTAL_SPLITS=0
DONE_SPLITS=0
FAILED_SPLITS=0

while IFS=$'\t' read -r split_name image_root; do
    split_dir="experiments/splits/$split_name"

    if [[ ! -f "$split_dir/test_forget.jsonl" ]]; then
        echo "SKIP $split_name: $split_dir/test_forget.jsonl not found (run 03_make_splits.py first)"
        continue
    fi

    TOTAL_SPLITS=$((TOTAL_SPLITS + 1))

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Split: $split_name"
    echo "║  Image root: $image_root"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # Build flags for run_batch_eval.sh
    EXTRA_FLAGS=""
    if [[ "$DRY_RUN" == true ]]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --dry_run"
    fi
    if [[ "$SKIP_EXISTING" == true ]]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --skip_existing"
    fi
    if [[ -n "$MAX_SAMPLES" ]]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --max_samples $MAX_SAMPLES"
    fi

    cmd="bash scripts/run_batch_eval.sh \
        --split_dir $split_dir \
        --image_root $image_root \
        --conditions $CONDITIONS \
        --size_tier $SIZE_TIER \
        --models $MODELS_FILE \
        --results_dir $RESULTS_DIR \
        $EXTRA_FLAGS"

    echo "CMD: $cmd"
    echo ""

    if eval "$cmd"; then
        DONE_SPLITS=$((DONE_SPLITS + 1))
    else
        FAILED_SPLITS=$((FAILED_SPLITS + 1))
        echo "WARNING: Batch eval for $split_name had failures."
    fi

done <<< "$SPLIT_ENTRIES"

echo ""
echo "================================================================"
echo "  Full Pipeline Complete"
echo "  Total splits : $TOTAL_SPLITS"
echo "  Succeeded    : $DONE_SPLITS"
echo "  Failed       : $FAILED_SPLITS"
echo "================================================================"
