"""End-to-end smoke of the new benchmark stack on a real model.

Composes the full v1 pipeline on a small sample:

    legacy split JSONL → schema migration → MCQ option randomization →
    P0 control variants → short-answer twins → EvaluationRunner artifacts

This is a construct smoke, not a release run: migrated items carry a
"pilot_smoke" provenance and the run manifest is scoped "pilot".

Usage (GPU node):

    python -m ulbench.tools.pilot_smoke \\
        --model Qwen/Qwen3-VL-2B-Instruct \\
        --split_jsonl experiments/splits/coco_rk10_s42/test_forget.jsonl \\
        --image_root data/coco \\
        --out_dir experiments/results/pilot_smoke/qwen3vl2b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ulbench.ids import slug
from ulbench.methods.noop import NoOpMethod
from ulbench.methods.prompt_suppression import PromptSuppressionMethod
from ulbench.models import create_adapter
from ulbench.probes.controls import build_control_variants
from ulbench.probes.direct import derive_short_answer_items, randomize_mcq_options
from ulbench.runner import EvaluationRunner, RunConfig
from ulbench.schema import Split
from ulbench.tools.migrate_legacy_jsonl import (
    MigrationContext,
    migrate_legacy_records,
)


def build_items(split_jsonl: str, n: int, seed: int):
    # Round-robin across concepts: legacy splits are class-grouped, and the
    # shuffled_image derangement needs balanced cross-concept blocks.
    by_concept: dict[str, list] = {}
    with Path(split_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            meta = record.get("meta", {})
            concept = meta.get("forget_concept") or meta.get("class_name")
            by_concept.setdefault(concept, []).append(record)

    records = []
    index = 0
    while len(records) < n:
        added = False
        for concept in sorted(by_concept):
            pool = by_concept[concept]
            if index < len(pool) and len(records) < n:
                records.append(pool[index])
                added = True
        if not added:
            break
        index += 1

    context = MigrationContext(
        dataset_id="pilot_smoke",
        source_version="legacy",
        split=Split.TEST_FORGET,
        license_id="unreviewed",
        redistribution="internal_only",
        provenance_note="pilot smoke sample; not for release",
    )
    migrated, report = migrate_legacy_records(records, context)
    if report["rejected_count"]:
        raise SystemExit(f"migration rejected records: {report['rejections']}")

    randomized, _ = randomize_mcq_options(migrated, seed=seed)
    with_controls, control_report = build_control_variants(randomized, seed=seed)
    short_twins = derive_short_answer_items(randomized)
    return with_controls + short_twins, control_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split_jsonl", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu_ids", default=None)
    parser.add_argument("--method", default="no_op",
                        choices=["no_op", "prompt_suppression_soft",
                                 "prompt_suppression_medium"])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args(argv)

    items, control_report = build_items(args.split_jsonl, args.n, args.seed)
    concepts = sorted({item.concept_name for item in items})
    print(f"Items: {len(items)}  (concepts: {', '.join(concepts)})")

    if args.method == "no_op":
        method = NoOpMethod()
    else:
        variant = args.method.rsplit("_", 1)[-1]
        method = PromptSuppressionMethod(variant, concepts)

    gpu_ids = ([int(part) for part in args.gpu_ids.split(",")]
               if args.gpu_ids else None)
    adapter = create_adapter(args.model, gpu_ids=gpu_ids)

    run_id = f"pilot_smoke__{slug(args.model)}__{args.method}__{int(time.time())}"
    config = RunConfig(
        run_id=run_id,
        out_dir=args.out_dir,
        image_root=args.image_root,
        seed=args.seed,
        notes={"control_report": control_report},
    )
    manifest = EvaluationRunner(adapter, method, config).run(items)

    print(f"Run {manifest['status']}: {run_id}")
    metrics = json.loads((Path(args.out_dir) / "metrics.json").read_text())
    print(f"Overall access_rate={metrics['overall']['access_rate']}  "
          f"coverage={metrics['overall']['coverage']}")
    for label, cell in metrics["by_condition"].items():
        print(f"  {label:45s} access={cell['access_rate']}  "
              f"n={cell['attempted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
