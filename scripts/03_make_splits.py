#!/usr/bin/env python3
"""Generate experiment splits for all datasets defined in split_registry.yaml.

Reads the registry, merges per-experiment overrides with defaults, and calls
``make_explicit_splits`` for each experiment entry.

Usage:
    python scripts/03_make_splits.py                         # all datasets
    python scripts/03_make_splits.py --only coco,aid         # specific datasets
    python scripts/03_make_splits.py --skip_existing         # skip if split dir exists
    python scripts/03_make_splits.py --dry_run               # print commands only
    python scripts/03_make_splits.py --registry path/to/custom_registry.yaml
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


SPLITS_ROOT = Path("experiments/splits")


def _load_registry(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_params(defaults: dict, experiment: dict) -> dict:
    """Merge experiment-level overrides onto defaults."""
    merged = dict(defaults)
    for key in ("n_train_per_forget_class", "n_test_per_forget_class",
                "n_retain_train", "n_retain_test", "seed"):
        if key in experiment:
            merged[key] = experiment[key]
    return merged


def _build_command(input_jsonl: str, experiment: dict, params: dict,
                   out_dir: Path) -> list[str]:
    """Build the CLI command for make_explicit_splits."""
    cmd = [
        sys.executable, "-m", "vqa_gen.tools.make_explicit_splits",
        "--input_jsonl", input_jsonl,
        "--k", str(experiment["k"]),
        "--mode", experiment["mode"],
        "--n_train_per_forget_class", str(params["n_train_per_forget_class"]),
        "--n_test_per_forget_class", str(params["n_test_per_forget_class"]),
        "--n_retain_train", str(params["n_retain_train"]),
        "--n_retain_test", str(params["n_retain_test"]),
        "--seed", str(params["seed"]),
        "--out_dir", str(out_dir),
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment splits from split_registry.yaml.",
    )
    parser.add_argument(
        "--registry", default="scripts/split_registry.yaml",
        help="Path to split registry YAML.",
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated dataset keys to process (default: all).",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip if split output directory already has test_forget.jsonl.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    registry = _load_registry(args.registry)
    defaults = registry.get("defaults", {})
    datasets = registry.get("datasets", {})

    # Filter
    if args.only:
        only_set = set(args.only.split(","))
        datasets = {k: v for k, v in datasets.items() if k in only_set}
        missing = only_set - set(datasets.keys())
        if missing:
            print(f"WARNING: Unknown dataset keys: {missing}", file=sys.stderr)

    print("=" * 62)
    print("  Split Generation Pipeline")
    print(f"  Registry     : {args.registry}")
    print(f"  Datasets     : {sorted(datasets.keys())}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Dry run      : {args.dry_run}")
    print("=" * 62)
    print()

    passed, skipped, failed = 0, 0, 0

    for ds_key, ds_cfg in sorted(datasets.items()):
        input_jsonl = ds_cfg["input_jsonl"]

        if not Path(input_jsonl).exists():
            print(f"  SKIP {ds_key}: {input_jsonl} not found (run 02_generate_vqa.sh first)")
            skipped += 1
            continue

        for exp in ds_cfg.get("experiments", []):
            name = exp["name"]
            out_dir = SPLITS_ROOT / name
            params = _merge_params(defaults, exp)

            print(f"── {name} ──")
            print(f"   Input : {input_jsonl}")
            print(f"   Output: {out_dir}")
            print(f"   K={exp['k']}, mode={exp['mode']}, seed={params['seed']}")

            if args.skip_existing and (out_dir / "test_forget.jsonl").exists():
                print(f"   Status: SKIPPED (already exists)")
                skipped += 1
                continue

            cmd = _build_command(input_jsonl, exp, params, out_dir)
            print(f"   CMD: {' '.join(cmd)}")

            if args.dry_run:
                print(f"   Status: DRY RUN")
                continue

            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - start

            if result.returncode == 0:
                print(f"   Status: DONE ({elapsed:.1f}s)")
                if result.stdout:
                    # Print last few lines (summary)
                    lines = result.stdout.strip().split("\n")
                    for line in lines[-6:]:
                        print(f"   {line}")
                passed += 1
            else:
                print(f"   Status: FAILED ({elapsed:.1f}s)")
                if result.stderr:
                    for line in result.stderr.strip().split("\n")[-5:]:
                        print(f"   ERR: {line}")
                failed += 1

            print()

    print("=" * 62)
    print(f"  Splits Complete:  passed={passed}  skipped={skipped}  failed={failed}")
    print("=" * 62)


if __name__ == "__main__":
    main()
