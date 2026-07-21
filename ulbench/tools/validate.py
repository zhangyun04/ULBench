"""Validate ULBench v1 items, run manifests, or a split registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ulbench.schema import BenchmarkItem, RunManifest, SchemaValidationError
from ulbench.validation import validate_benchmark_items, validate_split_registry_file


def _load_items(paths: list[str]) -> list[BenchmarkItem]:
    items = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    items.append(BenchmarkItem.from_dict(payload))
                except (json.JSONDecodeError, SchemaValidationError) as exc:
                    raise SchemaValidationError(
                        "benchmark JSONL", [f"{path}:{line_number}: {exc}"]
                    ) from exc
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    item_parser = subparsers.add_parser("items")
    item_parser.add_argument("paths", nargs="+")
    item_parser.add_argument("--require-core-controls", action="store_true")
    item_parser.add_argument("--skip-answer-balance-check", action="store_true")

    registry_parser = subparsers.add_parser("registry")
    registry_parser.add_argument("path")
    registry_parser.add_argument("--base-dir", default=".")
    registry_parser.add_argument("--skip-path-check", action="store_true")

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("path")

    args = parser.parse_args(argv)
    try:
        if args.command == "items":
            items = _load_items(args.paths)
            validate_benchmark_items(
                items,
                check_answer_balance=not args.skip_answer_balance_check,
                require_core_controls=args.require_core_controls,
            )
            print(f"Validated {len(items)} benchmark items")
        elif args.command == "registry":
            validate_split_registry_file(
                args.path,
                check_paths=not args.skip_path_check,
                base_dir=Path(args.base_dir),
            )
            print(f"Validated split registry: {args.path}")
        else:
            path = Path(args.path)
            RunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
            print(f"Validated run manifest: {path}")
    except (OSError, json.JSONDecodeError, SchemaValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
