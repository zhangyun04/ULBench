"""P0 visual-grounding control construction (spec §7.1, roadmap B-P0.4).

Given ``normal_image`` items, derive the paired core controls:

- ``no_image``        image removed, question and choices unchanged
- ``shuffled_image``  registered cross-concept derangement of images
- ``option_only``     image and question removed, choices unchanged
- ``question_only``   optional audit condition (image and choices removed)

All variants keep the source item's ``pairing_id`` so per-item paired
statistics work, and every derived permutation is deterministic from the
registered control seed. Construction fails loudly when a valid cross-concept
derangement cannot exist; it never silently degrades to same-concept donors.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import Any

from ulbench.ids import stable_id
from ulbench.schema import (
    BenchmarkItem,
    InputCondition,
    ProbeFamily,
    QuestionFormat,
    SchemaValidationError,
)

CORE_CONTROL_CONDITIONS = (
    InputCondition.NO_IMAGE,
    InputCondition.SHUFFLED_IMAGE,
    InputCondition.OPTION_ONLY,
)


def _variant_item_id(source: BenchmarkItem, condition: InputCondition) -> str:
    return stable_id(
        "item",
        source.sample_id,
        source.probe_id,
        source.prompt_variant_id,
        condition.value,
        source.metadata.get("option_order_hash", ""),
    )


def _base_payload(source: BenchmarkItem, condition: InputCondition,
                  seed: int) -> dict[str, Any]:
    payload = source.to_dict()
    payload["item_id"] = _variant_item_id(source, condition)
    payload["input_condition"] = condition.value
    payload["probe_family"] = ProbeFamily.P0_CONTROL.value
    payload["metadata"] = copy.deepcopy(payload["metadata"])
    payload["metadata"]["control"] = {
        "control_seed": seed,
        "derived_from_item_id": source.item_id,
    }
    return payload


def build_shuffled_image_assignment(
    items: list[BenchmarkItem], seed: int
) -> dict[str, BenchmarkItem]:
    """Return a deterministic cross-concept derangement ``item_id → donor``.

    Donors come from the same construction group (dataset, split, probe,
    format — the caller groups before calling) and always carry a different
    ``concept_id`` and a different ``image_id`` than the receiving item.

    Construction: order concept blocks largest-first, shuffle items inside
    each block with the seeded RNG, then rotate the concatenated sequence by
    the largest block size. With ``2 * max_block <= n`` this provably assigns
    every item a donor from another concept block.
    """
    if len(items) < 2:
        raise SchemaValidationError(
            "shuffled_image", ["need at least two items to build a derangement"]
        )

    blocks: dict[str, list[BenchmarkItem]] = defaultdict(list)
    for item in items:
        blocks[item.concept_id].append(item)

    if len(blocks) < 2:
        raise SchemaValidationError(
            "shuffled_image",
            ["all items share one concept_id; cross-concept donors are impossible"],
        )
    max_block = max(len(block) for block in blocks.values())
    if 2 * max_block > len(items):
        raise SchemaValidationError(
            "shuffled_image",
            [
                f"largest concept block has {max_block} of {len(items)} items; "
                "a cross-concept derangement cannot exist"
            ],
        )

    rng = random.Random(seed)
    ordered_blocks = sorted(
        blocks.items(), key=lambda pair: (-len(pair[1]), pair[0])
    )
    sequence: list[BenchmarkItem] = []
    for _, block in ordered_blocks:
        block = sorted(block, key=lambda item: item.item_id)
        rng.shuffle(block)
        sequence.extend(block)

    shift = max_block
    assignment: dict[str, BenchmarkItem] = {}
    problems: list[str] = []
    for index, item in enumerate(sequence):
        donor = sequence[(index + shift) % len(sequence)]
        if donor.concept_id == item.concept_id:
            problems.append(
                f"{item.item_id}: donor shares concept {item.concept_id}"
            )
        if donor.image_id == item.image_id:
            problems.append(
                f"{item.item_id}: donor shares image {item.image_id}"
            )
        assignment[item.item_id] = donor
    if problems:
        raise SchemaValidationError("shuffled_image", problems)
    return assignment


def build_control_variants(
    items: list[BenchmarkItem],
    *,
    seed: int,
    include_question_only: bool = False,
) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    """Derive P0 control variants for *items* and return (all_items, report).

    *items* must be ``normal_image`` records. The result contains the source
    items followed by their variants; every pairing then satisfies
    ``validate_benchmark_items(..., require_core_controls=True)``.
    """
    errors = []
    seen_ids = set()
    for item in items:
        if item.input_condition != InputCondition.NORMAL_IMAGE:
            errors.append(
                f"{item.item_id}: input_condition must be normal_image, "
                f"got {item.input_condition.value}"
            )
        if item.item_id in seen_ids:
            errors.append(f"duplicate item_id {item.item_id}")
        seen_ids.add(item.item_id)
    if errors:
        raise SchemaValidationError("control construction", errors)

    groups: dict[tuple, list[BenchmarkItem]] = defaultdict(list)
    for item in items:
        key = (item.dataset_id, item.split, item.probe_id,
               item.question_format, item.prompt_variant_id)
        groups[key].append(item)

    donor_by_item: dict[str, BenchmarkItem] = {}
    for key in sorted(groups, key=repr):
        donor_by_item.update(build_shuffled_image_assignment(groups[key], seed))

    variants: list[BenchmarkItem] = []
    for item in items:
        no_image = _base_payload(item, InputCondition.NO_IMAGE, seed)
        no_image["image"] = None
        variants.append(BenchmarkItem.from_dict(no_image))

        option_only = _base_payload(item, InputCondition.OPTION_ONLY, seed)
        option_only["image"] = None
        option_only["question"] = ""
        variants.append(BenchmarkItem.from_dict(option_only))

        donor = donor_by_item[item.item_id]
        shuffled = _base_payload(item, InputCondition.SHUFFLED_IMAGE, seed)
        shuffled["image"] = donor.image
        shuffled["metadata"]["donor_image_id"] = donor.image_id
        shuffled["metadata"]["control"].update({
            "donor_sample_id": donor.sample_id,
            "donor_concept_id": donor.concept_id,
        })
        variants.append(BenchmarkItem.from_dict(shuffled))

        if include_question_only:
            if item.question_format == QuestionFormat.MCQ:
                raise SchemaValidationError(
                    "control construction",
                    [
                        f"{item.item_id}: question_only removes choices and is "
                        "not defined for mcq items"
                    ],
                )
            question_only = _base_payload(item, InputCondition.QUESTION_ONLY, seed)
            question_only["image"] = None
            question_only["choices"] = None
            question_only["answer_index"] = None
            variants.append(BenchmarkItem.from_dict(question_only))

    report = {
        "control_seed": seed,
        "source_count": len(items),
        "variant_count": len(variants),
        "conditions": [condition.value for condition in CORE_CONTROL_CONDITIONS]
        + (["question_only"] if include_question_only else []),
        "derangement_groups": {
            repr(key): len(group) for key, group in sorted(groups.items(), key=repr)
        },
    }
    return list(items) + variants, report
