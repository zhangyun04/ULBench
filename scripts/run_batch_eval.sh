#!/usr/bin/env bash
# =============================================================================
# run_batch_eval.sh
# Batch evaluation of multiple VLMs on an experiment split.
#
# Usage:
#   bash scripts/run_batch_eval.sh [OPTIONS]
#
# Required:
#   --split_dir    DIR     Path to split directory (contains test_forget.jsonl etc.)
#   --image_root   DIR     Root directory for resolving image paths
#
# Optional:
#   --conditions   STR     Comma-separated conditions to run (default: all)
#                          Options: BASELINE_NORMAL,UNLEARN_SOFT,UNLEARN_MEDIUM,
#                                   ORACLE_HARD,ORACLE_REVERSE
#   --results_dir  DIR     Output root (default: experiments/results)
#   --models       FILE    Path to model list file, one HF model ID per line
#                          (default: scripts/model_list.txt)
#   --size_tier    STR     Filter by tier: small|medium|large|all (default: all)
#   --max_samples  N       Limit samples per split per model (for debugging)
#   --dry_run              Print commands without executing
#   --skip_existing        Skip if results dir already has metrics.json
#
# Examples:
#   # Run all conditions on all models, celebrity faces
#   bash scripts/run_batch_eval.sh \
#     --split_dir experiments/splits/celebrity_tom_cruise \
#     --image_root data/celebrity_faces
#
#   # Run only small models with baseline + oracle_hard
#   bash scripts/run_batch_eval.sh \
#     --split_dir experiments/splits/coco_randomk10_seed123 \
#     --image_root data/coco \
#     --size_tier small \
#     --conditions BASELINE_NORMAL,ORACLE_HARD
#
#   # Dry run to preview commands
#   bash scripts/run_batch_eval.sh \
#     --split_dir experiments/splits/coco_dog \
#     --image_root data/coco \
#     --dry_run
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SPLIT_DIR=""
IMAGE_ROOT=""
CONDITIONS="all"
RESULTS_DIR="experiments/results"
MODELS_FILE="scripts/model_list.txt"
SIZE_TIER="all"
MAX_SAMPLES=""
DRY_RUN=false
SKIP_EXISTING=false

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --split_dir)    SPLIT_DIR="$2";    shift 2 ;;
        --image_root)   IMAGE_ROOT="$2";   shift 2 ;;
        --conditions)   CONDITIONS="$2";   shift 2 ;;
        --results_dir)  RESULTS_DIR="$2";  shift 2 ;;
        --models)       MODELS_FILE="$2";  shift 2 ;;
        --size_tier)    SIZE_TIER="$2";    shift 2 ;;
        --max_samples)  MAX_SAMPLES="$2";  shift 2 ;;
        --dry_run)      DRY_RUN=true;      shift ;;
        --skip_existing) SKIP_EXISTING=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Validate required args ────────────────────────────────────────────────────
if [[ -z "$SPLIT_DIR" || -z "$IMAGE_ROOT" ]]; then
    echo "ERROR: --split_dir and --image_root are required."
    echo "Run: bash scripts/run_batch_eval.sh --help"
    exit 1
fi

if [[ ! -f "$SPLIT_DIR/test_forget.jsonl" ]]; then
    echo "ERROR: $SPLIT_DIR/test_forget.jsonl not found."
    exit 1
fi

# ── Build condition flags ─────────────────────────────────────────────────────
COND_FLAGS=""
if [[ "$CONDITIONS" == "all" ]]; then
    COND_FLAGS="--run_all"
else
    IFS=',' read -ra COND_LIST <<< "$CONDITIONS"
    for cond in "${COND_LIST[@]}"; do
        case "$cond" in
            BASELINE_NORMAL) ;;   # always runs, no flag needed
            UNLEARN_SOFT)    COND_FLAGS="$COND_FLAGS --run_unlearn_soft" ;;
            UNLEARN_MEDIUM)  COND_FLAGS="$COND_FLAGS --run_unlearn_medium" ;;
            ORACLE_HARD)     COND_FLAGS="$COND_FLAGS --run_oracle_hard" ;;
            ORACLE_REVERSE)  COND_FLAGS="$COND_FLAGS --run_oracle_reverse" ;;
            *) echo "WARNING: Unknown condition '$cond', skipping." ;;
        esac
    done
fi

# ── Optional forget_classes_json ──────────────────────────────────────────────
FC_FLAG=""
if [[ -f "$SPLIT_DIR/forget_classes.json" ]]; then
    FC_FLAG="--forget_classes_json $SPLIT_DIR/forget_classes.json"
fi

# ── Optional max_samples ──────────────────────────────────────────────────────
MS_FLAG=""
if [[ -n "$MAX_SAMPLES" ]]; then
    MS_FLAG="--max_samples_per_split $MAX_SAMPLES"
fi

# ── Infer split name from dir ─────────────────────────────────────────────────
SPLIT_NAME=$(basename "$SPLIT_DIR")

# ── Summary ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Batch VLM Evaluation"
echo "  Split     : $SPLIT_NAME  ($SPLIT_DIR)"
echo "  Image root: $IMAGE_ROOT"
echo "  Conditions: $CONDITIONS"
echo "  Size tier : $SIZE_TIER"
echo "  Results   : $RESULTS_DIR"
echo "  Dry run   : $DRY_RUN"
echo "================================================================"
echo ""

# ── Load model list ───────────────────────────────────────────────────────────
if [[ ! -f "$MODELS_FILE" ]]; then
    echo "ERROR: Model list not found: $MODELS_FILE"
    echo "       Create it or pass --models <path>"
    exit 1
fi

# Filter by size tier (based on tags in model_list.txt)
mapfile -t ALL_LINES < "$MODELS_FILE"
MODELS=()
for line in "${ALL_LINES[@]}"; do
    # Skip blank lines and comments
    [[ -z "$line" || "$line" == \#* ]] && continue
    # Extract model id (first field) and optional tier tag (second field)
    model_id=$(echo "$line" | awk '{print $1}')
    tier=$(echo "$line" | awk '{print $2}')  # e.g. #small, #medium, #large
    tier="${tier#\#}"  # strip leading #

    if [[ "$SIZE_TIER" == "all" || "$SIZE_TIER" == "$tier" ]]; then
        MODELS+=("$model_id")
    fi
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: No models matched tier='$SIZE_TIER'"
    exit 1
fi

echo "Models to evaluate (${#MODELS[@]}):"
for m in "${MODELS[@]}"; do echo "  $m"; done
echo ""

# ── Run loop ──────────────────────────────────────────────────────────────────
PASSED=0
SKIPPED=0
FAILED=0
FAILED_MODELS=()

for model_id in "${MODELS[@]}"; do
    # Sanitise model name for directory: replace / and : with _
    model_slug=$(echo "$model_id" | tr '/:' '__')
    out_dir="$RESULTS_DIR/${SPLIT_NAME}__${model_slug}"

    echo "────────────────────────────────────────────────────────────"
    echo "  Model : $model_id"
    echo "  Out   : $out_dir"

    # Skip if already done
    if [[ "$SKIP_EXISTING" == true && -f "$out_dir/metrics.json" ]]; then
        echo "  Status: SKIPPED (metrics.json exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    cmd="python -m experiments.intext_unlearning \
  --test_forget_jsonl $SPLIT_DIR/test_forget.jsonl \
  --test_retain_jsonl $SPLIT_DIR/test_retain.jsonl \
  $FC_FLAG \
  --image_root $IMAGE_ROOT \
  --model_name $model_id \
  $COND_FLAGS \
  $MS_FLAG \
  --out_dir $out_dir"

    echo "  CMD: $cmd"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo "  Status: DRY RUN"
        continue
    fi

    START=$(date +%s)
    if $cmd; then
        END=$(date +%s)
        echo "  Status: DONE  ($(( END - START ))s)"
        PASSED=$((PASSED + 1))
    else
        END=$(date +%s)
        echo "  Status: FAILED ($(( END - START ))s)"
        FAILED=$((FAILED + 1))
        FAILED_MODELS+=("$model_id")
    fi
    echo ""
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Batch complete"
echo "  Passed : $PASSED"
echo "  Skipped: $SKIPPED"
echo "  Failed : $FAILED"
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo "  Failed models:"
    for m in "${FAILED_MODELS[@]}"; do echo "    $m"; done
fi
echo "================================================================"
