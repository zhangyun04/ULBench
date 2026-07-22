"""Explicitly migrate legacy ULBench/VQA JSONL records to schema v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ulbench.ids import option_order_hash as _option_order_hash
from ulbench.ids import slug as _slug
from ulbench.ids import stable_id as _stable_id
from ulbench.schema import (
    SCHEMA_VERSION,
    BenchmarkItem,
    InputCondition,
    ProbeFamily,
    QuestionFormat,
    SchemaValidationError,
    Split,
)
from ulbench.validation import validate_benchmark_items


@dataclass(frozen=True)
class MigrationContext:
    dataset_id: str
    source_version: str
    split: Split
    license_id: str
    redistribution: str
    provenance_note: str
    probe_id: str = "legacy.mcq.identity.v1"
    prompt_variant_id: str = "legacy_default"


def _migrate_record(
    record: dict[str, Any], context: MigrationContext, fallbacks: Counter
) -> BenchmarkItem:
    required_legacy = {
        "id", "image", "question", "choices", "answer_index",
        "forgetting_level", "concept_axis", "meta",
    }
    missing = sorted(required_legacy - set(record))
    if missing:
        raise SchemaValidationError(
            "legacy record", [f"missing required legacy fields: {', '.join(missing)}"]
        )
    if not isinstance(record["meta"], dict):
        raise SchemaValidationError("legacy record", ["meta must be an object"])

    legacy_split = record.get("target_split")
    if legacy_split not in (None, "all", context.split.value):
        raise SchemaValidationError(
            "legacy record",
            [
                f"target_split={legacy_split!r} conflicts with explicit "
                f"split={context.split.value!r}"
            ],
        )

    choices = record["choices"]
    answer_index = record["answer_index"]
    if not isinstance(choices, list) or not isinstance(answer_index, int) or not (
        0 <= answer_index < len(choices)
    ):
        raise SchemaValidationError(
            "legacy record", ["choices/answer_index do not identify a valid answer"]
        )
    answer = choices[answer_index]
    if not isinstance(answer, str) or not answer.strip():
        raise SchemaValidationError("legacy record", ["indexed answer must be non-empty"])

    meta = record["meta"]
    concept_name = meta.get("forget_concept")
    if not isinstance(concept_name, str) or not concept_name.strip():
        concept_name = meta.get("class_name") or meta.get("synset")
        fallbacks["concept_name_from_class_or_synset"] += 1
    if not isinstance(concept_name, str) or not concept_name.strip():
        raise SchemaValidationError(
            "legacy record", ["cannot resolve concept name from meta"]
        )

    sample_id = str(record["id"])
    image = record["image"]
    if not isinstance(image, str) or not image.strip():
        raise SchemaValidationError("legacy record", ["image must be non-empty"])
    image_id = _stable_id("image", context.dataset_id, image)
    concept_id = f"{context.dataset_id}:{_slug(concept_name)}"
    option_order_hash = _option_order_hash(choices)
    pairing_id = _stable_id(
        "pair",
        sample_id,
        context.probe_id,
        context.prompt_variant_id,
        option_order_hash,
    )
    item_id = _stable_id(
        "item",
        sample_id,
        context.probe_id,
        context.prompt_variant_id,
        InputCondition.NORMAL_IMAGE.value,
        option_order_hash,
    )

    fallbacks["concept_id_from_dataset_and_normalized_name"] += 1
    fallbacks["concept_aliases_empty"] += 1
    fallbacks["image_id_from_dataset_and_image_reference"] += 1
    return BenchmarkItem.from_dict({
        "schema_version": SCHEMA_VERSION,
        "item_id": item_id,
        "sample_id": sample_id,
        "dataset_id": context.dataset_id,
        "image_id": image_id,
        "image": image,
        "concept_id": concept_id,
        "concept_name": concept_name,
        "concept_aliases": [],
        "forgetting_level": record["forgetting_level"],
        "concept_axis": record["concept_axis"],
        "split": context.split.value,
        "probe_id": context.probe_id,
        "probe_family": ProbeFamily.P1_DIRECT.value,
        "question_format": QuestionFormat.MCQ.value,
        "prompt_variant_id": context.prompt_variant_id,
        "input_condition": InputCondition.NORMAL_IMAGE.value,
        "question": record["question"],
        "choices": choices,
        "accepted_answers": [answer],
        "answer_index": answer_index,
        "matched_retain_id": None,
        "source": {
            "dataset_id": context.dataset_id,
            "record_id": sample_id,
            "version": context.source_version,
        },
        "license": {
            "id": context.license_id,
            "redistribution": context.redistribution,
        },
        "provenance_note": context.provenance_note,
        "metadata": {
            "pairing_id": pairing_id,
            "option_order_hash": option_order_hash,
            "legacy_meta": meta,
            "migration": {
                "source_schema": "legacy_dataset_item",
                "target_split_original": legacy_split,
            },
        },
    })


def migrate_legacy_records(
    records: Iterable[dict[str, Any]], context: MigrationContext
) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    """Migrate records without hiding failures; rejected rows stay in the report."""

    items: list[BenchmarkItem] = []
    rejections: list[dict[str, Any]] = []
    fallbacks: Counter = Counter()
    input_count = 0
    for index, record in enumerate(records):
        input_count += 1
        try:
            items.append(_migrate_record(record, context, fallbacks))
        except (SchemaValidationError, TypeError, ValueError) as exc:
            rejections.append({
                "record_index": index,
                "legacy_id": record.get("id") if isinstance(record, dict) else None,
                "error": str(exc),
            })
    report = {
        "source_schema": "legacy_dataset_item",
        "target_schema_version": SCHEMA_VERSION,
        "dataset_id": context.dataset_id,
        "explicit_split": context.split.value,
        "input_count": input_count,
        "migrated_count": len(items),
        "rejected_count": len(rejections),
        "fallbacks": dict(sorted(fallbacks.items())),
        "rejections": rejections,
    }
    return items, report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(
                    "legacy JSONL", [f"line {line_number}: {exc}"]
                ) from exc
            if not isinstance(payload, dict):
                raise SchemaValidationError(
                    "legacy JSONL", [f"line {line_number}: record must be an object"]
                )
            records.append(payload)
    return records


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl_atomic(path: Path, items: list[BenchmarkItem]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy DatasetItem JSONL into ULBench schema v1.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--split", required=True, choices=[member.value for member in Split])
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--redistribution", required=True)
    parser.add_argument("--provenance-note", required=True)
    parser.add_argument("--probe-id", default="legacy.mcq.identity.v1")
    parser.add_argument("--prompt-variant-id", default="legacy_default")
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--allow-rejects",
        action="store_true",
        help="Write valid migrated rows while preserving rejects in the report.",
    )
    parser.add_argument(
        "--skip-answer-balance-check",
        action="store_true",
        help="Migration diagnostic only; output is not release-conformant.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_suffix(
        output_path.suffix + ".migration_report.json"
    )
    context = MigrationContext(
        dataset_id=args.dataset_id,
        source_version=args.source_version,
        split=Split(args.split),
        license_id=args.license_id,
        redistribution=args.redistribution,
        provenance_note=args.provenance_note,
        probe_id=args.probe_id,
        prompt_variant_id=args.prompt_variant_id,
    )

    try:
        records = _read_jsonl(input_path)
        items, report = migrate_legacy_records(records, context)
        report["input_sha256"] = _file_hash(input_path)
        if report["rejected_count"] and not args.allow_rejects:
            raise SchemaValidationError(
                "legacy migration",
                [
                    f"{report['rejected_count']} records rejected; inspect {report_path} "
                    "or rerun with --allow-rejects"
                ],
            )
        validate_benchmark_items(
            items,
            check_answer_balance=not args.skip_answer_balance_check,
        )
        _write_jsonl_atomic(output_path, items)
        report["output_sha256"] = _file_hash(output_path)
        report["validation_status"] = (
            "diagnostic_balance_skipped"
            if args.skip_answer_balance_check else "passed"
        )
    except (OSError, SchemaValidationError) as exc:
        report = locals().get("report", {
            "source_schema": "legacy_dataset_item",
            "target_schema_version": SCHEMA_VERSION,
        })
        report["validation_status"] = "failed"
        report["fatal_error"] = str(exc)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(str(exc), file=sys.stderr)
        print(f"Migration report: {report_path}", file=sys.stderr)
        return 2

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"Migrated {len(items)} records to {output_path}")
    print(f"Migration report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
