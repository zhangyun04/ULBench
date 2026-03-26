"""Build explicit experiment splits from a pre-generated identity JSONL.

Supports two modes:

  SINGLE-TARGET  (--forget_class dog)
    Original behaviour: one forget class, flat pool slicing.

  MULTI-TARGET   (--k 10 --mode random_k  *or*  --forget_classes_json ...)
    K forget classes selected via random sampling, superclass-balanced
    sampling, or an explicit JSON list.  Per-class stratified sampling
    guarantees every forget class appears exactly n_test_per_forget_class
    times in test_forget.

Example usage
-------------
Single-target (unchanged):
  python -m vqa_gen.tools.make_explicit_splits \\
    --input_jsonl vqa_gen/output/output_coco/all.jsonl \\
    --forget_class dog \\
    --n_forget_test 50 --n_retain_test 50 \\
    --out_dir experiments/splits/dog/

Multi-target random_k:
  python -m vqa_gen.tools.make_explicit_splits \\
    --input_jsonl vqa_gen/output/output_coco/all.jsonl \\
    --k 10 --mode random_k \\
    --n_train_per_forget_class 50 --n_test_per_forget_class 50 \\
    --n_retain_train 2000 --n_retain_test 500 \\
    --seed 123 \\
    --out_dir experiments/splits/randomk10_seed123/

Multi-target superclass_balanced_k:
  python -m vqa_gen.tools.make_explicit_splits \\
    --input_jsonl vqa_gen/output/output_coco/all.jsonl \\
    --k 10 --mode superclass_balanced_k \\
    --n_train_per_forget_class 50 --n_test_per_forget_class 50 \\
    --n_retain_train 2000 --n_retain_test 500 \\
    --seed 123 \\
    --out_dir experiments/splits/balancedk10_seed123/

Manual forget-class list:
  python -m vqa_gen.tools.make_explicit_splits \\
    --input_jsonl vqa_gen/output/output_coco/all.jsonl \\
    --forget_classes_json path/to/forget_classes.json \\
    --n_train_per_forget_class 50 --n_test_per_forget_class 50 \\
    --n_retain_train 2000 --n_retain_test 500 \\
    --seed 123 \\
    --out_dir experiments/splits/manual/
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ── I/O helpers ─────────────────────────────────────────────────────────

def _load_jsonl(path):
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(path, items):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _check_no_overlap(named_splits):
    """Abort if any ID appears in more than one split."""
    seen: dict[str, str] = {}
    for name, items in named_splits:
        for item in items:
            iid = item["id"]
            if iid in seen:
                print(
                    f"OVERLAP: id={iid} appears in both '{seen[iid]}' and '{name}'",
                    file=sys.stderr,
                )
                sys.exit(1)
            seen[iid] = name


# ── Deterministic hashing ──────────────────────────────────────────────

def _stable_hash(s: str) -> int:
    """Deterministic integer hash of a string (seed-independent)."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


# ── Forget-class selection (multi-target) ──────────────────────────────

def _eligible_categories(items_by_class, min_count):
    """Return sorted list of category names with at least *min_count* items."""
    return sorted(c for c, v in items_by_class.items() if len(v) >= min_count)


def _sample_random_k(eligible, k, rng):
    pool = list(eligible)
    rng.shuffle(pool)
    if len(pool) < k:
        print(
            f"ERROR: only {len(pool)} eligible categories but requested k={k}",
            file=sys.stderr,
        )
        sys.exit(1)
    return sorted(pool[:k])


def _sample_superclass_balanced_k(eligible, k, superclass_of, rng):
    """Round-robin allocation of K slots across superclasses."""
    by_super: dict[str, list[str]] = defaultdict(list)
    for cat in eligible:
        by_super[superclass_of.get(cat, "unknown")].append(cat)
    for cats in by_super.values():
        rng.shuffle(cats)

    super_names = sorted(by_super.keys())
    rng.shuffle(super_names)

    allocation: dict[str, int] = {s: 0 for s in super_names}
    remaining = k
    active = list(super_names)

    while remaining > 0 and active:
        per = max(1, remaining // len(active))
        next_active = []
        for s in active:
            can_give = len(by_super[s]) - allocation[s]
            give = min(per, can_give, remaining)
            allocation[s] += give
            remaining -= give
            if allocation[s] < len(by_super[s]):
                next_active.append(s)
            if remaining == 0:
                break
        if next_active == active:
            # per was 1 and all supplied — force round-robin one at a time
            for s in active:
                if remaining == 0:
                    break
                can_give = len(by_super[s]) - allocation[s]
                if can_give > 0:
                    allocation[s] += 1
                    remaining -= 1
            break
        active = next_active

    if remaining > 0:
        total_eligible = sum(len(v) for v in by_super.values())
        print(
            f"ERROR: cannot allocate k={k} across superclasses "
            f"(total eligible={total_eligible})",
            file=sys.stderr,
        )
        sys.exit(1)

    selected: list[str] = []
    for s in sorted(allocation):
        selected.extend(by_super[s][: allocation[s]])

    return sorted(selected), {s: allocation[s] for s in sorted(allocation) if allocation[s] > 0}


def _load_forget_classes_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return sorted(data)
    if isinstance(data, dict) and "forget_classes" in data:
        return sorted(data["forget_classes"])
    print(
        f"ERROR: {path} must be a JSON list or object with 'forget_classes' key.",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Single-target mode (original logic, unchanged) ────────────────────

def _run_single_target(items, args):
    """Exact original behaviour for --forget_class."""
    forget_pool = [i for i in items if i["meta"]["synset"] == args.forget_class]
    retain_pool = [i for i in items if i["meta"]["synset"] != args.forget_class]
    print(f"Forget pool ({args.forget_class}): {len(forget_pool)}")
    print(f"Retain pool (other classes):       {len(retain_pool)}")

    rng = random.Random(args.seed)
    rng.shuffle(forget_pool)
    rng.shuffle(retain_pool)

    n_forget_test = args.n_forget_test
    n_forget_train = args.n_forget_train
    if n_forget_train is None:
        n_forget_train = len(forget_pool) - n_forget_test

    if n_forget_train < 0:
        n_forget_train = 0

    need_forget = n_forget_train + n_forget_test
    need_retain = args.n_retain_train + args.n_retain_test

    if len(forget_pool) < need_forget:
        print(
            f"ERROR: forget pool ({len(forget_pool)}) < "
            f"n_forget_train ({n_forget_train}) + n_forget_test ({n_forget_test}) = {need_forget}",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(retain_pool) < need_retain:
        print(
            f"ERROR: retain pool ({len(retain_pool)}) < "
            f"n_retain_train ({args.n_retain_train}) + n_retain_test ({args.n_retain_test}) = {need_retain}",
            file=sys.stderr,
        )
        sys.exit(1)

    train_forget = forget_pool[:n_forget_train]
    test_forget = forget_pool[n_forget_train: n_forget_train + n_forget_test]
    train_retain = retain_pool[: args.n_retain_train]
    test_retain = retain_pool[args.n_retain_train: args.n_retain_train + args.n_retain_test]

    _check_no_overlap([
        ("train_forget", train_forget),
        ("train_retain", train_retain),
        ("test_forget", test_forget),
        ("test_retain", test_retain),
    ])

    out = Path(args.out_dir)
    _write_jsonl(out / "train_forget.jsonl", train_forget)
    _write_jsonl(out / "train_retain.jsonl", train_retain)
    _write_jsonl(out / "test_forget.jsonl", test_forget)
    _write_jsonl(out / "test_retain.jsonl", test_retain)
    _write_jsonl(out / "combined_train.jsonl", train_forget + train_retain)
    _write_jsonl(out / "combined_test.jsonl", test_forget + test_retain)

    stats = {
        "forget_class": args.forget_class,
        "seed": args.seed,
        "input_total": len(items),
        "forget_pool_size": len(forget_pool),
        "retain_pool_size": len(retain_pool),
        "counts": {
            "train_forget": len(train_forget),
            "train_retain": len(train_retain),
            "test_forget": len(test_forget),
            "test_retain": len(test_retain),
        },
        "class_distribution": {
            "train_forget": dict(Counter(i["meta"]["synset"] for i in train_forget)),
            "test_forget": dict(Counter(i["meta"]["synset"] for i in test_forget)),
            "train_retain": dict(Counter(i["meta"]["synset"] for i in train_retain).most_common(10)),
            "test_retain": dict(Counter(i["meta"]["synset"] for i in test_retain).most_common(10)),
        },
    }
    with (out / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    _print_summary(out)


# ── Multi-target mode ─────────────────────────────────────────────────

def _run_multi_target(items, forget_classes, mode_name, superclass_allocation,
                      args):
    """Per-class stratified splitting for K forget classes."""
    n_train_fc = args.n_train_per_forget_class
    n_test_fc = args.n_test_per_forget_class
    need_per_class = n_train_fc + n_test_fc

    forget_set = set(forget_classes)

    items_by_class: dict[str, list] = defaultdict(list)
    for item in items:
        items_by_class[item["meta"]["synset"]].append(item)

    # Validate every requested forget class has enough items
    for cls in forget_classes:
        avail = len(items_by_class.get(cls, []))
        if avail < need_per_class:
            print(
                f"ERROR: forget class '{cls}' has {avail} items but needs "
                f"{n_train_fc} (train) + {n_test_fc} (test) = {need_per_class}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Per-class stratified sampling for forget splits
    train_forget: list = []
    test_forget: list = []
    used_forget_ids: set[str] = set()

    per_class_counts_train: dict[str, int] = {}
    per_class_counts_test: dict[str, int] = {}

    for cls in forget_classes:
        class_items = list(items_by_class[cls])
        class_rng = random.Random(args.seed + _stable_hash(cls))
        class_rng.shuffle(class_items)

        test_slice = class_items[:n_test_fc]
        train_slice = class_items[n_test_fc: n_test_fc + n_train_fc]

        test_forget.extend(test_slice)
        train_forget.extend(train_slice)
        used_forget_ids.update(i["id"] for i in test_slice)
        used_forget_ids.update(i["id"] for i in train_slice)

        per_class_counts_test[cls] = len(test_slice)
        per_class_counts_train[cls] = len(train_slice)

    # Verify macro guarantee
    for cls in forget_classes:
        if per_class_counts_test[cls] != n_test_fc:
            print(
                f"ERROR: macro guarantee failed for '{cls}': "
                f"got {per_class_counts_test[cls]} test items, expected {n_test_fc}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Retain pool: everything NOT in a forget class
    retain_pool = [i for i in items if i["meta"]["synset"] not in forget_set]
    rng = random.Random(args.seed)
    rng.shuffle(retain_pool)

    need_retain = args.n_retain_train + args.n_retain_test
    if len(retain_pool) < need_retain:
        print(
            f"ERROR: retain pool ({len(retain_pool)}) < "
            f"n_retain_train ({args.n_retain_train}) + "
            f"n_retain_test ({args.n_retain_test}) = {need_retain}",
            file=sys.stderr,
        )
        sys.exit(1)

    train_retain = retain_pool[: args.n_retain_train]
    test_retain = retain_pool[args.n_retain_train: args.n_retain_train + args.n_retain_test]

    _check_no_overlap([
        ("train_forget", train_forget),
        ("train_retain", train_retain),
        ("test_forget", test_forget),
        ("test_retain", test_retain),
    ])

    # ── Write outputs ──────────────────────────────────────────────
    out = Path(args.out_dir)
    _write_jsonl(out / "train_forget.jsonl", train_forget)
    _write_jsonl(out / "train_retain.jsonl", train_retain)
    _write_jsonl(out / "test_forget.jsonl", test_forget)
    _write_jsonl(out / "test_retain.jsonl", test_retain)
    _write_jsonl(out / "combined_train.jsonl", train_forget + train_retain)
    _write_jsonl(out / "combined_test.jsonl", test_forget + test_retain)

    # ── forget_classes.json ────────────────────────────────────────
    superclass_of: dict[str, str] = {}
    for item in items:
        c = item["meta"]["synset"]
        if c not in superclass_of:
            superclass_of[c] = item["meta"].get("superclass", "unknown")

    forget_classes_info = []
    for cls in forget_classes:
        forget_classes_info.append({
            "class": cls,
            "superclass": superclass_of.get(cls, "unknown"),
            "count": len(items_by_class.get(cls, [])),
        })

    fc_doc = {
        "seed": args.seed,
        "mode": mode_name,
        "k": len(forget_classes),
        "forget_classes": forget_classes_info,
        "superclass_allocation": superclass_allocation or {},
    }
    with (out / "forget_classes.json").open("w", encoding="utf-8") as f:
        json.dump(fc_doc, f, indent=2)

    # ── split_stats.json ──────────────────────────────────────────
    stats = {
        "mode": mode_name,
        "seed": args.seed,
        "k": len(forget_classes),
        "selected_forget_classes": forget_classes,
        "input_total": len(items),
        "forget_pool_size": sum(len(items_by_class[c]) for c in forget_classes),
        "retain_pool_size": len(retain_pool),
        "counts": {
            "train_forget": len(train_forget),
            "train_retain": len(train_retain),
            "test_forget": len(test_forget),
            "test_retain": len(test_retain),
        },
        "per_forget_class_counts": {
            "train_forget": per_class_counts_train,
            "test_forget": per_class_counts_test,
        },
        "n_train_per_forget_class": n_train_fc,
        "n_test_per_forget_class": n_test_fc,
        "macro_guarantee": (
            f"every forget class has exactly {n_test_fc} samples in test_forget"
        ),
        "class_distribution": {
            "train_forget": dict(Counter(i["meta"]["synset"] for i in train_forget).most_common()),
            "test_forget": dict(Counter(i["meta"]["synset"] for i in test_forget).most_common()),
            "train_retain": dict(Counter(i["meta"]["synset"] for i in train_retain).most_common(10)),
            "test_retain": dict(Counter(i["meta"]["synset"] for i in test_retain).most_common(10)),
        },
    }
    with (out / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    _print_summary(out, extra_files=["forget_classes"])


# ── Console summary ───────────────────────────────────────────────────

def _print_summary(out, extra_files=None):
    names = [
        "train_forget", "train_retain", "test_forget", "test_retain",
        "combined_train", "combined_test", "split_stats",
    ]
    if extra_files:
        names.extend(extra_files)
    print(f"\nWrote splits to {out}/")
    for name in names:
        suffix = ".json" if name in ("split_stats", "forget_classes") else ".jsonl"
        print(f"  {name}{suffix}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build explicit forget/retain × train/test splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_jsonl", required=True,
        help="Source JSONL (e.g. output/output_coco/all.jsonl).",
    )

    # ── Single-target args (preserved) ─────────────────────────────
    parser.add_argument(
        "--forget_class", default=None,
        help="Single forget category (e.g. 'dog'). For single-target mode.",
    )
    parser.add_argument("--n_forget_train", type=int, default=None)
    parser.add_argument("--n_forget_test", type=int, default=50)

    # ── Multi-target args (new) ────────────────────────────────────
    parser.add_argument(
        "--k", type=int, default=None,
        help="Number of forget classes for multi-target.",
    )
    parser.add_argument(
        "--mode", default=None, choices=["random_k", "superclass_balanced_k"],
        help="Forget-class selection mode (requires --k).",
    )
    parser.add_argument(
        "--forget_classes_json", default=None,
        help="JSON file with explicit forget class list (overrides --k/--mode).",
    )
    parser.add_argument("--n_train_per_forget_class", type=int, default=50)
    parser.add_argument("--n_test_per_forget_class", type=int, default=50)

    # ── Shared args ────────────────────────────────────────────────
    parser.add_argument("--n_retain_train", type=int, default=200)
    parser.add_argument("--n_retain_test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    # ── Determine mode ─────────────────────────────────────────────
    is_multi = args.forget_classes_json is not None or args.k is not None

    if is_multi and args.forget_class is not None:
        print(
            "WARNING: --forget_class is ignored when using multi-target mode "
            "(--k / --forget_classes_json).",
            file=sys.stderr,
        )

    if not is_multi and args.forget_class is None:
        print(
            "ERROR: must specify --forget_class (single-target) or "
            "--k/--forget_classes_json (multi-target).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.k is not None and args.mode is None and args.forget_classes_json is None:
        print(
            "ERROR: --k requires --mode (random_k | superclass_balanced_k).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Warn if single-target per-class args are mixed with multi-target
    if is_multi and (args.n_forget_train is not None or args.n_forget_test != 50):
        print(
            "WARNING: --n_forget_train / --n_forget_test are ignored in "
            "multi-target mode. Use --n_train_per_forget_class / "
            "--n_test_per_forget_class instead.",
            file=sys.stderr,
        )

    # ── Load data ──────────────────────────────────────────────────
    items = _load_jsonl(args.input_jsonl)
    print(f"Loaded {len(items)} items from {args.input_jsonl}")

    if not is_multi:
        _run_single_target(items, args)
        return

    # ── Multi-target: resolve forget classes ───────────────────────
    items_by_class: dict[str, list] = defaultdict(list)
    superclass_of: dict[str, str] = {}
    for item in items:
        c = item["meta"]["synset"]
        items_by_class[c].append(item)
        if c not in superclass_of:
            superclass_of[c] = item["meta"].get("superclass", "unknown")

    min_count = args.n_train_per_forget_class + args.n_test_per_forget_class
    eligible = _eligible_categories(items_by_class, min_count)
    print(f"Eligible categories (>= {min_count} items): {len(eligible)}")

    rng = random.Random(args.seed)
    superclass_allocation: dict[str, int] | None = None

    if args.forget_classes_json is not None:
        forget_classes = _load_forget_classes_json(args.forget_classes_json)
        mode_name = "explicit_json"
        # Validate all requested classes exist in the data
        for cls in forget_classes:
            if cls not in items_by_class:
                print(f"ERROR: class '{cls}' from JSON not found in data.", file=sys.stderr)
                sys.exit(1)
    elif args.mode == "random_k":
        forget_classes = _sample_random_k(eligible, args.k, rng)
        mode_name = "random_k"
    elif args.mode == "superclass_balanced_k":
        forget_classes, superclass_allocation = _sample_superclass_balanced_k(
            eligible, args.k, superclass_of, rng,
        )
        mode_name = "superclass_balanced_k"
    else:
        print("ERROR: unreachable mode state.", file=sys.stderr)
        sys.exit(1)

    print(f"Selected {len(forget_classes)} forget classes ({mode_name}): {forget_classes}")
    if superclass_allocation:
        print(f"Superclass allocation: {superclass_allocation}")

    _run_multi_target(items, forget_classes, mode_name, superclass_allocation, args)


if __name__ == "__main__":
    main()
