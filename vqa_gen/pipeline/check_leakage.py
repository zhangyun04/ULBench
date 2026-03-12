"""Check for class-level leakage across target splits in generated JSONL."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def check_leakage(output_root: str) -> bool:
    """Return True if no leakage detected, False otherwise."""
    root = Path(output_root)

    synsets_by_split: dict[str, set[str]] = defaultdict(set)

    for jsonl_file in sorted(root.glob("*.jsonl")):
        for item in _load_jsonl(jsonl_file):
            target_split = item.get("target_split")
            synset = item.get("meta", {}).get("synset")
            if target_split and synset:
                synsets_by_split[target_split].add(synset)

    splits = sorted(synsets_by_split.keys())
    leaks: list[tuple[str, str, set[str]]] = []

    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            overlap = synsets_by_split[s1] & synsets_by_split[s2]
            if overlap:
                leaks.append((s1, s2, overlap))

    if leaks:
        print("LEAKAGE DETECTED:")
        for s1, s2, overlap in leaks:
            print(f"  {s1} <-> {s2}: {len(overlap)} overlapping classes")
            for synset in sorted(overlap)[:10]:
                print(f"    - {synset}")
            if len(overlap) > 10:
                print(f"    ... and {len(overlap) - 10} more")
        return False

    print("No leakage detected.")
    for split in splits:
        print(f"  {split}: {len(synsets_by_split[split])} classes")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check for class leakage across target splits."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to output directory containing JSONL files.",
    )
    args = parser.parse_args()

    ok = check_leakage(args.output_root)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
