#!/usr/bin/env bash
# =============================================================================
# 05_aggregate_all.sh
# Aggregate results for all splits into per-split CSV files and one combined CSV.
#
# Usage:
#   bash scripts/05_aggregate_all.sh                         # all splits
#   bash scripts/05_aggregate_all.sh --only coco_rk10_s42    # specific split
#   bash scripts/05_aggregate_all.sh --results_dir experiments/results
# =============================================================================
set -euo pipefail

RESULTS_DIR="experiments/results"
OUT_DIR="experiments/results/summaries"
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
        --out_dir)     OUT_DIR="$2";     shift 2 ;;
        --only)        ONLY="$2";        shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

echo "================================================================"
echo "  Results Aggregation"
echo "  Results dir : $RESULTS_DIR"
echo "  Output dir  : $OUT_DIR"
echo "  Filter      : ${ONLY:-all}"
echo "================================================================"
echo ""

# Discover split names from result directories.
# Result dirs follow the pattern: {split_name}__{model_slug}
# We extract unique split names.
SPLIT_NAMES=$(ls -d "$RESULTS_DIR"/*__* 2>/dev/null | xargs -n1 basename | sed 's/__.*$//' | sort -u)

if [[ -z "$SPLIT_NAMES" ]]; then
    echo "No result directories found in $RESULTS_DIR."
    exit 0
fi

AGGREGATED=0

for split_name in $SPLIT_NAMES; do
    # Filter
    if [[ -n "$ONLY" ]]; then
        match=false
        IFS=',' read -ra FILTERS <<< "$ONLY"
        for f in "${FILTERS[@]}"; do
            if [[ "$split_name" == *"$f"* ]]; then
                match=true
                break
            fi
        done
        if [[ "$match" == false ]]; then
            continue
        fi
    fi

    out_csv="$OUT_DIR/${split_name}_summary.csv"
    echo "  Aggregating: $split_name → $out_csv"

    if python scripts/aggregate_results.py \
        --results_dir "$RESULTS_DIR" \
        --split_name "$split_name" \
        --out "$out_csv"; then
        AGGREGATED=$((AGGREGATED + 1))
    else
        echo "    WARNING: aggregation failed for $split_name"
    fi
done

echo ""
echo "================================================================"
echo "  Aggregated $AGGREGATED split(s) into $OUT_DIR/"
echo "================================================================"

# Also produce a combined mega-CSV if we have multiple splits
if [[ $AGGREGATED -gt 1 ]]; then
    COMBINED="$OUT_DIR/all_splits_combined.csv"
    echo ""
    echo "  Generating combined CSV: $COMBINED"

    # Concatenate all CSVs, keeping header from first file only
    FIRST=true
    for csv_file in "$OUT_DIR"/*_summary.csv; do
        if [[ "$FIRST" == true ]]; then
            # Add a "split" column header + full content
            head -1 "$csv_file" | sed 's/^/split,/'
            tail -n +2 "$csv_file" | sed "s/^/$(basename "$csv_file" _summary.csv),/"
            FIRST=false
        else
            tail -n +2 "$csv_file" | sed "s/^/$(basename "$csv_file" _summary.csv),/"
        fi
    done > "$COMBINED"

    echo "  Done: $COMBINED"
fi
