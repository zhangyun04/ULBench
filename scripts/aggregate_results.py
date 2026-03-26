"""Aggregate metrics.json files from batch evaluation into a single CSV/table.

Usage:
    python scripts/aggregate_results.py \
        --results_dir experiments/results \
        --split_name celebrity_tom_cruise \
        --out experiments/results/summary.csv

    # Print to stdout instead of saving:
    python scripts/aggregate_results.py \
        --results_dir experiments/results \
        --split_name celebrity_tom_cruise
"""

import argparse
import json
import csv
import sys
from pathlib import Path

# Metrics to extract per condition
_CONDITIONS = [
    "baseline_normal",
    "unlearn_soft",
    "unlearn_medium",
    "oracle_hard",
    "oracle_reverse",
]

_METRICS_PER_COND = [
    "forget_macro_acc",
    "forget_micro_acc",
    "retain_acc",
    "invalid_rate_forget",
    "invalid_rate_retain",
    "forget_pred_entropy",
]


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open() as f:
        return json.load(f)


def extract_row(model_slug: str, metrics: dict) -> dict:
    row = {"model": model_slug, "K": metrics.get("K", "?")}
    for cond in _CONDITIONS:
        prefix = cond
        for metric in _METRICS_PER_COND:
            key = f"{prefix}__{metric}"
            val = metrics.get(key, "")
            row[f"{cond}__{metric}"] = val
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate batch evaluation metrics into a CSV table."
    )
    parser.add_argument(
        "--results_dir", default="experiments/results",
        help="Root results directory.",
    )
    parser.add_argument(
        "--split_name", required=True,
        help="Split name prefix (e.g. 'celebrity_tom_cruise').",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output CSV path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--sort_by", default="baseline_normal__forget_macro_acc",
        help="Column to sort rows by (descending).",
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    pattern = f"{args.split_name}__*"
    run_dirs = sorted(results_root.glob(pattern))

    if not run_dirs:
        print(f"No results found matching: {results_root / pattern}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"  SKIP (no metrics.json): {run_dir.name}", file=sys.stderr)
            continue
        # Extract model slug: strip the split_name prefix
        model_slug = run_dir.name.removeprefix(f"{args.split_name}__")
        metrics = load_metrics(metrics_path)
        rows.append(extract_row(model_slug, metrics))

    if not rows:
        print("No valid results to aggregate.", file=sys.stderr)
        sys.exit(1)

    # Sort
    def sort_key(r):
        v = r.get(args.sort_by, "")
        try:
            return float(v)
        except (ValueError, TypeError):
            return -1.0

    rows.sort(key=sort_key, reverse=True)

    # Build column list
    fieldnames = ["model", "K"]
    for cond in _CONDITIONS:
        for metric in _METRICS_PER_COND:
            fieldnames.append(f"{cond}__{metric}")

    # Write
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {args.out}")
    else:
        # Pretty-print to stdout
        writer = csv.DictWriter(
            sys.stdout, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
