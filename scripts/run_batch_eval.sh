#!/usr/bin/env bash
# =============================================================================
# run_batch_eval.sh
# Batch evaluation of multiple VLMs on an experiment split.
#
# GPU Allocation (8×B200, 192 GB each):
#   small  (≤8B)  → 1 GPU  → up to 8 models in parallel
#   medium (8-32B) → 2 GPUs → up to 4 models in parallel
#   large  (>32B)  → 4 GPUs → up to 2 models in parallel
#
# Models of the SAME tier are grouped and dispatched in parallel waves.
# Models of DIFFERENT tiers are run sequentially by tier (small → medium → large).
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
#   --batch_size   N       Override batch size (default: auto by tier)
#   --max_samples  N       Limit samples per split per model (for debugging)
#   --dry_run              Print commands without executing
#   --skip_existing        Skip if results dir already has metrics.json
#   --sequential           Disable parallel dispatch, run one model at a time
#
# Examples:
#   # Run all conditions on all models (parallel by tier)
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
SEQUENTIAL=false

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
        --sequential)   SEQUENTIAL=true;          shift ;;
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
# small (≤8B):  1 GPU,  batch=16
# medium(8-32B): 2 GPUs, batch=8
# large (>32B):  4 GPUs, batch=4
declare -A TIER_GPUS=( [small]=1 [medium]=2 [large]=4 )
declare -A TIER_BATCH=( [small]=16 [medium]=8 [large]=4 )

# ── Summary ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Batch VLM Evaluation (GPU-parallel)"
echo "  Split      : $SPLIT_NAME  ($SPLIT_DIR)"
echo "  Image root : $IMAGE_ROOT"
echo "  Conditions : $CONDITIONS"
echo "  Size tier  : $SIZE_TIER"
echo "  Num GPUs   : $NUM_GPUS"
echo "  Results    : $RESULTS_DIR"
echo "  Sequential : $SEQUENTIAL"
echo "  Dry run    : $DRY_RUN"
echo "================================================================"
echo ""

# ── Load model list ───────────────────────────────────────────────────────────
if [[ ! -f "$MODELS_FILE" ]]; then
    echo "ERROR: Model list not found: $MODELS_FILE"
    exit 1
fi

# Parse models grouped by tier
declare -a SMALL_MODELS=()
declare -a MEDIUM_MODELS=()
declare -a LARGE_MODELS=()

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip blank lines and comments
    [[ -z "$line" || "$line" == \#* ]] && continue
    model_id=$(echo "$line" | awk '{print $1}')
    tier=$(echo "$line" | awk '{print $2}')
    tier="${tier#\#}"  # strip leading #

    if [[ "$SIZE_TIER" != "all" && "$SIZE_TIER" != "$tier" ]]; then
        continue
    fi

    case "$tier" in
        small)  SMALL_MODELS+=("$model_id") ;;
        medium) MEDIUM_MODELS+=("$model_id") ;;
        large)  LARGE_MODELS+=("$model_id") ;;
        *)      SMALL_MODELS+=("$model_id") ;;  # default to small
    esac
done < "$MODELS_FILE"

TOTAL_MODELS=$(( ${#SMALL_MODELS[@]} + ${#MEDIUM_MODELS[@]} + ${#LARGE_MODELS[@]} ))
if [[ $TOTAL_MODELS -eq 0 ]]; then
    echo "ERROR: No models matched tier='$SIZE_TIER'"
    exit 1
fi

echo "Models by tier:"
echo "  small  (1 GPU each) : ${#SMALL_MODELS[@]}"
echo "  medium (2 GPUs each): ${#MEDIUM_MODELS[@]}"
echo "  large  (4 GPUs each): ${#LARGE_MODELS[@]}"
echo ""

# ── Helper: dispatch a wave of models in parallel ─────────────────────────────
PASSED=0
SKIPPED=0
FAILED=0
declare -a FAILED_MODELS=()

# Track PIDs → model_id mapping for reporting
declare -A PID_MODEL=()
# Track PIDs → log_file mapping
declare -A PID_LOG=()

run_wave() {
    # Args: tier_name gpus_per_model batch_size model_ids...
    local tier_name="$1"; shift
    local gpus_per_model="$1"; shift
    local batch_size="$1"; shift
    local models=("$@")

    if [[ ${#models[@]} -eq 0 ]]; then
        return
    fi

    # Override batch size if user specified
    if [[ -n "$BATCH_SIZE_OVERRIDE" ]]; then
        batch_size="$BATCH_SIZE_OVERRIDE"
    fi

    local max_parallel=$(( NUM_GPUS / gpus_per_model ))
    if [[ $max_parallel -lt 1 ]]; then
        max_parallel=1
    fi

    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Tier: $tier_name  |  ${gpus_per_model} GPU/model  |  batch=$batch_size"
    echo "║  Models: ${#models[@]}  |  Max parallel: $max_parallel"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    local gpu_cursor=0
    local running_pids=()
    PID_MODEL=()
    PID_LOG=()

    for model_id in "${models[@]}"; do
        local model_slug=$(echo "$model_id" | tr '/:' '__')
        local out_dir="$RESULTS_DIR/${SPLIT_NAME}__${model_slug}"
        local log_file="$out_dir/run.log"

        echo "  ── $model_id ──"

        # Skip if already done
        if [[ "$SKIP_EXISTING" == true && -f "$out_dir/metrics.json" ]]; then
            echo "     SKIPPED (metrics.json exists)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Wait for a GPU slot if all are occupied
        while [[ ${#running_pids[@]} -ge $max_parallel ]]; do
            _wait_for_any_pid
        done

        # Allocate GPUs: assign next available slice
        local gpu_start=$gpu_cursor
        local gpu_list=""
        for (( g=0; g<gpus_per_model; g++ )); do
            local gpu_idx=$(( (gpu_start + g) % NUM_GPUS ))
            if [[ -n "$gpu_list" ]]; then
                gpu_list="$gpu_list,$gpu_idx"
            else
                gpu_list="$gpu_idx"
            fi
        done
        gpu_cursor=$(( (gpu_start + gpus_per_model) % NUM_GPUS ))

        local cmd="python -m experiments.intext_unlearning \
  --test_forget_jsonl $SPLIT_DIR/test_forget.jsonl \
  --test_retain_jsonl $SPLIT_DIR/test_retain.jsonl \
  $FC_FLAG \
  --image_root $IMAGE_ROOT \
  --model_name $model_id \
  --gpu_ids $gpu_list \
  --batch_size $batch_size \
  $COND_FLAGS \
  $MS_FLAG \
  --out_dir $out_dir"

        echo "     GPUs: $gpu_list  |  Out: $out_dir"
        echo "     CMD: $cmd"

        if [[ "$DRY_RUN" == true ]]; then
            echo "     Status: DRY RUN"
            echo ""
            continue
        fi

        # Launch in background (or foreground if sequential)
        mkdir -p "$out_dir"

        if [[ "$SEQUENTIAL" == true ]]; then
            echo ""
            local START=$(date +%s)
            if CUDA_VISIBLE_DEVICES="$gpu_list" eval "$cmd" 2>&1 | tee "$log_file"; then
                local END=$(date +%s)
                echo "     Status: DONE ($(( END - START ))s)"
                PASSED=$((PASSED + 1))
            else
                local END=$(date +%s)
                echo "     Status: FAILED ($(( END - START ))s)"
                FAILED=$((FAILED + 1))
                FAILED_MODELS+=("$model_id")
            fi
        else
            CUDA_VISIBLE_DEVICES="$gpu_list" eval "$cmd" > "$log_file" 2>&1 &
            local pid=$!
            running_pids+=($pid)
            PID_MODEL[$pid]="$model_id"
            PID_LOG[$pid]="$log_file"
            echo "     PID: $pid → log: $log_file"
        fi
        echo ""
    done

    # Wait for all remaining background jobs
    for pid in "${running_pids[@]}"; do
        _wait_for_pid "$pid"
    done
}

_wait_for_any_pid() {
    # Wait for any one of the running pids to finish
    while true; do
        for i in "${!running_pids[@]}"; do
            local pid="${running_pids[$i]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                _collect_pid "$pid"
                unset 'running_pids[$i]'
                running_pids=("${running_pids[@]}")  # reindex
                return
            fi
        done
        sleep 2
    done
}

_wait_for_pid() {
    local pid="$1"
    wait "$pid" 2>/dev/null
    _collect_pid "$pid"
}

_collect_pid() {
    local pid="$1"
    local model_id="${PID_MODEL[$pid]:-unknown}"
    local log_file="${PID_LOG[$pid]:-/dev/null}"

    # Check if the process succeeded by looking for metrics.json
    local model_slug=$(echo "$model_id" | tr '/:' '__')
    local out_dir="$RESULTS_DIR/${SPLIT_NAME}__${model_slug}"

    if [[ -f "$out_dir/metrics.json" ]]; then
        echo "  ✓ DONE: $model_id (log: $log_file)"
        PASSED=$((PASSED + 1))
    else
        echo "  ✗ FAILED: $model_id (log: $log_file)"
        FAILED=$((FAILED + 1))
        FAILED_MODELS+=("$model_id")
    fi

    unset "PID_MODEL[$pid]"
    unset "PID_LOG[$pid]"
}

# ── Dispatch by tier (small first, then medium, then large) ───────────────────
run_wave "small"  "${TIER_GPUS[small]}"  "${TIER_BATCH[small]}"  "${SMALL_MODELS[@]}"
run_wave "medium" "${TIER_GPUS[medium]}" "${TIER_BATCH[medium]}" "${MEDIUM_MODELS[@]}"
run_wave "large"  "${TIER_GPUS[large]}"  "${TIER_BATCH[large]}"  "${LARGE_MODELS[@]}"

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
