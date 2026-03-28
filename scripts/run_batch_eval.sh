#!/usr/bin/env bash
# =============================================================================
# run_batch_eval.sh
# Batch evaluation of multiple VLMs on an experiment split.
#
# Data-parallel MapReduce (8×B200, 192 GB each):
#   For each model, spawn N workers (one per GPU for small models).
#   Each worker loads the same model independently on its own GPU and
#   processes 1/N of the data.  Results are merged automatically.
#
#   small  (≤8B)  → 8 workers × 1 GPU each  → batch=16  (full 8-GPU parallel)
#   medium (8-32B) → 4 workers × 2 GPUs each → batch=8
#   large  (>32B)  → 2 workers × 4 GPUs each → batch=4
#
#   Models are run ONE AT A TIME (sequentially), but each model uses ALL
#   GPUs via data-parallel workers for maximum throughput per model.
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
#   --results_dir  DIR     Output root (default: experiments/results)
#   --models       FILE    Path to model list file (default: scripts/model_list.txt)
#   --size_tier    STR     Filter by tier: small|medium|large|all (default: all)
#   --num_gpus     N       Total number of GPUs available (default: 8)
#   --batch_size   N       Override batch size per worker (default: auto by tier)
#   --max_samples  N       Limit samples per split per model (for debugging)
#   --dry_run              Print commands without executing
#   --skip_existing        Skip if results dir already has metrics.json
#
# Examples:
#   # Run all conditions on all models (auto data-parallel)
#   bash scripts/run_batch_eval.sh \
#     --split_dir experiments/splits/coco_rk10_s42 \
#     --image_root data/coco
#
#   # Run only small models
#   bash scripts/run_batch_eval.sh \
#     --split_dir experiments/splits/coco_rk10_s42 \
#     --image_root data/coco \
#     --size_tier small
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
NUM_GPUS=8
BATCH_SIZE_OVERRIDE=""
MAX_SAMPLES=""
DRY_RUN=false
SKIP_EXISTING=false

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --split_dir)    SPLIT_DIR="$2";           shift 2 ;;
        --image_root)   IMAGE_ROOT="$2";          shift 2 ;;
        --conditions)   CONDITIONS="$2";          shift 2 ;;
        --results_dir)  RESULTS_DIR="$2";         shift 2 ;;
        --models)       MODELS_FILE="$2";         shift 2 ;;
        --size_tier)    SIZE_TIER="$2";           shift 2 ;;
        --num_gpus)     NUM_GPUS="$2";            shift 2 ;;
        --batch_size)   BATCH_SIZE_OVERRIDE="$2"; shift 2 ;;
        --max_samples)  MAX_SAMPLES="$2";         shift 2 ;;
        --dry_run)      DRY_RUN=true;             shift ;;
        --skip_existing) SKIP_EXISTING=true;      shift ;;
        # Keep --sequential for backward compat (ignored, all models are sequential now)
        --sequential)   shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Validate required args ────────────────────────────────────────────────────
if [[ -z "$SPLIT_DIR" || -z "$IMAGE_ROOT" ]]; then
    echo "ERROR: --split_dir and --image_root are required."
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
            BASELINE_NORMAL) ;;
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

# ── GPU allocation per tier (B200 192GB) ──────────────────────────────────────
# Each model uses ALL GPUs via --num_workers data-parallel.
# GPUs per worker determines how many workers fit.
#   small:  1 GPU/worker  → 8 workers on 8 GPUs
#   medium: 2 GPUs/worker → 4 workers on 8 GPUs
#   large:  4 GPUs/worker → 2 workers on 8 GPUs
declare -A TIER_GPUS_PER_WORKER=( [small]=1 [medium]=2 [large]=4 )
declare -A TIER_BATCH=( [small]=16 [medium]=8 [large]=4 )

# ── Summary ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Batch VLM Evaluation (MapReduce data-parallel)"
echo "  Split      : $SPLIT_NAME  ($SPLIT_DIR)"
echo "  Image root : $IMAGE_ROOT"
echo "  Conditions : $CONDITIONS"
echo "  Size tier  : $SIZE_TIER"
echo "  Num GPUs   : $NUM_GPUS"
echo "  Results    : $RESULTS_DIR"
echo "  Dry run    : $DRY_RUN"
echo "================================================================"
echo ""

# ── Load model list ───────────────────────────────────────────────────────────
if [[ ! -f "$MODELS_FILE" ]]; then
    echo "ERROR: Model list not found: $MODELS_FILE"
    exit 1
fi

# Parse models with their tier
declare -a MODEL_IDS=()
declare -a MODEL_TIERS=()

while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    model_id=$(echo "$line" | awk '{print $1}')
    tier=$(echo "$line" | awk '{print $2}')
    tier="${tier#\#}"

    if [[ "$SIZE_TIER" != "all" && "$SIZE_TIER" != "$tier" ]]; then
        continue
    fi

    MODEL_IDS+=("$model_id")
    MODEL_TIERS+=("${tier:-small}")
done < "$MODELS_FILE"

if [[ ${#MODEL_IDS[@]} -eq 0 ]]; then
    echo "ERROR: No models matched tier='$SIZE_TIER'"
    exit 1
fi

echo "Models to evaluate (${#MODEL_IDS[@]}):"
for i in "${!MODEL_IDS[@]}"; do
    local_tier="${MODEL_TIERS[$i]}"
    local_gpw="${TIER_GPUS_PER_WORKER[$local_tier]:-1}"
    local_nw=$(( NUM_GPUS / local_gpw ))
    echo "  ${MODEL_IDS[$i]}  (${local_tier}, ${local_nw} workers × ${local_gpw} GPU)"
done
echo ""

# ── Run models sequentially, each with data-parallel workers ──────────────────
PASSED=0
SKIPPED=0
FAILED=0
declare -a FAILED_MODELS=()

for i in "${!MODEL_IDS[@]}"; do
    model_id="${MODEL_IDS[$i]}"
    tier="${MODEL_TIERS[$i]}"
    model_slug=$(echo "$model_id" | tr '/:' '__')
    out_dir="$RESULTS_DIR/${SPLIT_NAME}__${model_slug}"

    # Compute workers and batch size for this tier
    gpus_per_worker="${TIER_GPUS_PER_WORKER[$tier]:-1}"
    num_workers=$(( NUM_GPUS / gpus_per_worker ))
    if [[ $num_workers -lt 1 ]]; then
        num_workers=1
    fi
    batch_size="${TIER_BATCH[$tier]:-8}"
    if [[ -n "$BATCH_SIZE_OVERRIDE" ]]; then
        batch_size="$BATCH_SIZE_OVERRIDE"
    fi

    # Build GPU list for --gpu_ids (all available)
    gpu_list=""
    for (( g=0; g<NUM_GPUS; g++ )); do
        if [[ -n "$gpu_list" ]]; then
            gpu_list="$gpu_list,$g"
        else
            gpu_list="$g"
        fi
    done

    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Model   : $model_id"
    echo "║  Tier    : $tier  |  ${num_workers} workers × ${gpus_per_worker} GPU  |  batch=$batch_size"
    echo "║  Output  : $out_dir"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # Skip if already done
    if [[ "$SKIP_EXISTING" == true && -f "$out_dir/metrics.json" ]]; then
        echo "  SKIPPED (metrics.json exists)"
        SKIPPED=$((SKIPPED + 1))
        echo ""
        continue
    fi

    cmd="python -m experiments.intext_unlearning \
  --test_forget_jsonl $SPLIT_DIR/test_forget.jsonl \
  --test_retain_jsonl $SPLIT_DIR/test_retain.jsonl \
  $FC_FLAG \
  --image_root \"$IMAGE_ROOT\" \
  --model_name $model_id \
  --gpu_ids $gpu_list \
  --batch_size $batch_size \
  --num_workers $num_workers \
  $COND_FLAGS \
  $MS_FLAG \
  --out_dir $out_dir"

    echo "  CMD: $cmd"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  Status: DRY RUN"
        echo ""
        continue
    fi

    mkdir -p "$out_dir"
    log_file="$out_dir/run.log"

    START=$(date +%s)
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        END=$(date +%s)
        echo "  ✓ DONE: $model_id  ($(( END - START ))s)"
        PASSED=$((PASSED + 1))
    else
        END=$(date +%s)
        echo "  ✗ FAILED: $model_id  ($(( END - START ))s)  — see $log_file"
        FAILED=$((FAILED + 1))
        FAILED_MODELS+=("$model_id")
    fi
    echo ""
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
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
