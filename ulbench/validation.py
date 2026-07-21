"""Cross-record and repository configuration validation for ULBench."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from ulbench.schema import (
    BenchmarkItem,
    InputCondition,
    QuestionFormat,
    SchemaValidationError,
    Split,
)


CORE_CONTROLS = {
    InputCondition.NORMAL_IMAGE,
    InputCondition.NO_IMAGE,
    InputCondition.SHUFFLED_IMAGE,
    InputCondition.OPTION_ONLY,
}
_MULTI_SPLIT_NAME = re.compile(r"_(?P<kind>rk|sbk)(?P<k>\d+)_s(?P<seed>\d+)$")


def validate_benchmark_items(
    items: Iterable[BenchmarkItem],
    *,
    check_answer_balance: bool = True,
    require_core_controls: bool = False,
) -> None:
    """Validate item-level schemas plus cross-record invariants.

    Raises :class:`SchemaValidationError` with every detected problem. It never
    drops bad records or repairs data silently.
    """

    materialized = list(items)
    errors: list[str] = []
    for index, item in enumerate(materialized):
        if not isinstance(item, BenchmarkItem):
            errors.append(f"record {index} is not a BenchmarkItem")
            continue
        try:
            item.validate()
        except SchemaValidationError as exc:
            errors.extend(f"record {index}: {error}" for error in exc.errors)

    if errors:
        raise SchemaValidationError("benchmark dataset", errors)

    item_ids = Counter(item.item_id for item in materialized)
    duplicates = sorted(item_id for item_id, count in item_ids.items() if count > 1)
    if duplicates:
        errors.append(f"duplicate item_id values: {', '.join(duplicates[:10])}")

    sample_invariants: dict[str, tuple[str, str, str, Split]] = {}
    for item in materialized:
        current = (item.dataset_id, item.image_id, item.concept_id, item.split)
        prior = sample_invariants.setdefault(item.sample_id, current)
        if prior != current:
            errors.append(
                f"sample_id {item.sample_id!r} has inconsistent dataset/image/concept/split: "
                f"{prior!r} vs {current!r}"
            )

    train_items = [item for item in materialized if item.split in {
        Split.TRAIN_FORGET, Split.TRAIN_RETAIN
    }]
    test_items = [item for item in materialized if item.split in {
        Split.TEST_FORGET, Split.TEST_RETAIN
    }]
    train_sample_ids = {item.sample_id for item in train_items}
    test_sample_ids = {item.sample_id for item in test_items}
    sample_overlap = sorted(train_sample_ids & test_sample_ids)
    if sample_overlap:
        errors.append(
            "train/test sample overlap: " + ", ".join(sample_overlap[:10])
        )

    train_image_ids = {item.image_id for item in train_items}
    test_image_ids = {item.image_id for item in test_items}
    image_overlap = sorted(train_image_ids & test_image_ids)
    if image_overlap:
        errors.append(
            "train/test source image overlap: " + ", ".join(image_overlap[:10])
        )

    forget_concepts = {
        item.concept_id for item in materialized
        if item.split in {Split.TRAIN_FORGET, Split.TEST_FORGET}
    }
    retain_concepts = {
        item.concept_id for item in materialized
        if item.split in {Split.TRAIN_RETAIN, Split.TEST_RETAIN}
    }
    concept_overlap = sorted(forget_concepts & retain_concepts)
    if concept_overlap:
        errors.append(
            "forget/retain concept overlap: " + ", ".join(concept_overlap[:10])
        )

    if check_answer_balance:
        _validate_answer_position_balance(materialized, errors)
    if require_core_controls:
        _validate_core_controls(materialized, errors)

    if errors:
        raise SchemaValidationError("benchmark dataset", errors)


def _validate_answer_position_balance(
    items: list[BenchmarkItem], errors: list[str]
) -> None:
    blocks: dict[tuple[str, Split, str, str, int], list[int]] = defaultdict(list)
    for item in items:
        if item.question_format != QuestionFormat.MCQ or item.choices is None:
            continue
        key = (
            item.dataset_id,
            item.split,
            item.probe_id,
            item.prompt_variant_id,
            len(item.choices),
        )
        blocks[key].append(item.answer_index)  # type: ignore[arg-type]

    for key, positions in blocks.items():
        choice_count = key[-1]
        counts = Counter(positions)
        per_position = [counts.get(index, 0) for index in range(choice_count)]
        if max(per_position, default=0) - min(per_position, default=0) > 1:
            errors.append(
                f"answer-position imbalance in block {key!r}: {per_position}; "
                "counts must differ by at most one"
            )


def _validate_core_controls(
    items: list[BenchmarkItem], errors: list[str]
) -> None:
    groups: dict[str, set[InputCondition]] = defaultdict(set)
    for item in items:
        groups[item.metadata["pairing_id"]].add(item.input_condition)
    for pairing_id, conditions in sorted(groups.items()):
        missing = sorted(condition.value for condition in CORE_CONTROLS - conditions)
        if missing:
            errors.append(
                f"pairing_id {pairing_id!r} missing core controls: {', '.join(missing)}"
            )


def validate_split_registry_data(
    registry: dict[str, Any],
    *,
    check_paths: bool = True,
    base_dir: Optional[Path] = None,
) -> None:
    """Validate split names, parameters, uniqueness, and optional input paths."""

    errors: list[str] = []
    if not isinstance(registry, dict):
        raise SchemaValidationError("split registry", ["registry must be an object"])
    defaults = registry.get("defaults", {})
    datasets = registry.get("datasets")
    if not isinstance(defaults, dict):
        errors.append("defaults must be an object")
        defaults = {}
    if not isinstance(datasets, dict) or not datasets:
        errors.append("datasets must be a non-empty object")
        datasets = {}

    names: set[str] = set()
    root = base_dir or Path.cwd()
    for dataset_key, dataset in datasets.items():
        if not isinstance(dataset, dict):
            errors.append(f"datasets.{dataset_key} must be an object")
            continue
        input_jsonl = dataset.get("input_jsonl")
        if not isinstance(input_jsonl, str) or not input_jsonl:
            errors.append(f"datasets.{dataset_key}.input_jsonl must be non-empty")
        elif check_paths and not (root / input_jsonl).is_file():
            errors.append(
                f"datasets.{dataset_key}.input_jsonl not found: {root / input_jsonl}"
            )

        experiments = dataset.get("experiments", [])
        if not isinstance(experiments, list) or not experiments:
            errors.append(f"datasets.{dataset_key}.experiments must be non-empty")
            continue
        for experiment in experiments:
            if not isinstance(experiment, dict):
                errors.append(f"datasets.{dataset_key} experiment must be an object")
                continue
            name = experiment.get("name")
            mode = experiment.get("mode")
            if not isinstance(name, str) or not name:
                errors.append(f"datasets.{dataset_key} experiment name must be non-empty")
                continue
            if name in names:
                errors.append(f"duplicate experiment name: {name}")
            names.add(name)
            seed = experiment.get("seed", defaults.get("seed"))
            if not isinstance(seed, int):
                errors.append(f"{name}: seed must be an integer")

            if mode == "single_target":
                if not isinstance(experiment.get("forget_class"), str):
                    errors.append(f"{name}: single_target requires forget_class")
                if "k" in experiment:
                    errors.append(f"{name}: single_target must not define k")
                continue
            if mode not in {"random_k", "superclass_balanced_k"}:
                errors.append(f"{name}: unsupported mode {mode!r}")
                continue
            k = experiment.get("k")
            if not isinstance(k, int) or k < 1:
                errors.append(f"{name}: k must be a positive integer")
                continue
            match = _MULTI_SPLIT_NAME.search(name)
            if match is None:
                errors.append(f"{name}: multi-target name must encode k and seed")
                continue
            encoded_kind = match.group("kind")
            expected_kind = "rk" if mode == "random_k" else "sbk"
            if encoded_kind != expected_kind:
                errors.append(
                    f"{name}: encodes mode {encoded_kind!r}, expected {expected_kind!r}"
                )
            encoded_k = int(match.group("k"))
            encoded_seed = int(match.group("seed"))
            if encoded_k != k:
                errors.append(f"{name}: encodes k={encoded_k}, configured k={k}")
            if encoded_seed != seed:
                errors.append(
                    f"{name}: encodes seed={encoded_seed}, configured seed={seed}"
                )

            for parameter in (
                "n_train_per_forget_class", "n_test_per_forget_class",
                "n_retain_train", "n_retain_test",
            ):
                value = experiment.get(parameter, defaults.get(parameter))
                if not isinstance(value, int) or value < 0:
                    errors.append(f"{name}: {parameter} must be a non-negative integer")

    if errors:
        raise SchemaValidationError("split registry", errors)


def validate_split_registry_file(
    path: str | Path, *, check_paths: bool = True, base_dir: Optional[Path] = None
) -> None:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    validate_split_registry_data(
        registry,
        check_paths=check_paths,
        base_dir=base_dir,
    )
