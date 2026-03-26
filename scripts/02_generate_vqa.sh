#!/usr/bin/env bash
# =============================================================================
# 02_generate_vqa.sh
# Generate VQA JSONL for all (or selected) dataset configs.
#
# Usage:
#   bash scripts/02_generate_vqa.sh                       # all configs
#   bash scripts/02_generate_vqa.sh --only coco,aid       # specific datasets
#   bash scripts/02_generate_vqa.sh --skip_existing        # skip if all.jsonl exists
#   bash scripts/02_generate_vqa.sh --max_samples 100      # limit per config (debug)
#   bash scripts/02_generate_vqa.sh --dry_run
# =============================================================================
set -euo pipefail

CONFIGS_DIR="vqa_gen/configs"
ONLY=""
SKIP_EXISTING=false
MAX_SAMPLES=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)          ONLY="$2";          shift 2 ;;
        --skip_existing) SKIP_EXISTING=true; shift ;;
        --max_samples)   MAX_SAMPLES="$2";   shift 2 ;;
        --dry_run)       DRY_RUN=true;       shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Build filter set
declare -A ONLY_SET
if [[ -n "$ONLY" ]]; then
    IFS=',' read -ra NAMES <<< "$ONLY"
    for n in "${NAMES[@]}"; do ONLY_SET["$n"]=1; done
fi

echo "================================================================"
echo "  VQA Generation Pipeline"
echo "  Configs dir  : $CONFIGS_DIR"
echo "  Filter       : ${ONLY:-all}"
echo "  Skip existing: $SKIP_EXISTING"
echo "  Max samples  : ${MAX_SAMPLES:-unlimited}"
echo "  Dry run      : $DRY_RUN"
echo "================================================================"
echo ""

PASSED=0
SKIPPED=0
FAILED=0

for config in "$CONFIGS_DIR"/*.yaml; do
    basename=$(basename "$config" .yaml)
    # Extract a short name: strip _mvp suffix, e.g. "coco_identity_mvp" → "coco_identity"
    short_name="${basename%_mvp}"

    # Filter check
    if [[ -n "$ONLY" ]]; then
        match=false
        for key in "${!ONLY_SET[@]}"; do
            if [[ "$short_name" == *"$key"* ]]; then
                match=true
                break
            fi
        done
        if [[ "$match" == false ]]; then
            continue
        fi
    fi

    # Get output root from config
    output_root=$(python3 -c "
import yaml, sys
with open('$config') as f:
    cfg = yaml.safe_load(f)
print(cfg['paths']['output_root'])
" 2>/dev/null || echo "")

    if [[ -z "$output_root" ]]; then
        echo "WARNING: Cannot parse output_root from $config, skipping."
        continue
    fi

    echo "────────────────────────────────────────────────────────────"
    echo "  Config: $basename"
    echo "  Output: $output_root"

    # Skip if already generated
    if [[ "$SKIP_EXISTING" == true && -f "$output_root/all.jsonl" ]]; then
        echo "  Status: SKIPPED (all.jsonl exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    MS_FLAG=""
    if [[ -n "$MAX_SAMPLES" ]]; then
        MS_FLAG="--max_samples $MAX_SAMPLES"
    fi

    cmd="python -m vqa_gen.pipeline.build --config $config $MS_FLAG"
    echo "  CMD: $cmd"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  Status: DRY RUN"
        continue
    fi

    START=$(date +%s)
    if eval "$cmd"; then
        END=$(date +%s)
        echo "  Status: DONE  ($((END - START))s)"
        PASSED=$((PASSED + 1))
    else
        END=$(date +%s)
        echo "  Status: FAILED ($((END - START))s)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo ""
echo "================================================================"
echo "  VQA Generation Complete"
echo "  Passed : $PASSED"
echo "  Skipped: $SKIPPED"
echo "  Failed : $FAILED"
echo "================================================================"
