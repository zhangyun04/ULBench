"""P1 direct probes: MCQ option randomization and short-answer derivation
(spec §7.2, §8; roadmap B-P0.4 P1).

MCQ permutations are deterministic from the registered option-order seed,
stored per item, and answer positions are balanced within each spec §8
balancing block (dataset × split × probe × prompt variant × choice count) by
construction. Randomize *before* deriving P0 controls so every pairing shares
one option order.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import Any, Optional

from vqa_gen.ontology.normalize import normalize_choice_text

from ulbench.ids import option_order_hash, stable_id
from ulbench.schema import (
    BenchmarkItem,
    InputCondition,
    ProbeFamily,
    QuestionFormat,
    SchemaValidationError,
)

SHORT_ANSWER_PROBE_ID = "direct.short_answer.v1"


def _rebuild_ids(payload: dict[str, Any]) -> None:
    ooh = option_order_hash(payload["choices"]) if payload["choices"] else ""
    payload["metadata"]["option_order_hash"] = ooh
    payload["metadata"]["pairing_id"] = stable_id(
        "pair", payload["sample_id"], payload["probe_id"],
        payload["prompt_variant_id"], ooh,
    )
    payload["item_id"] = stable_id(
        "item", payload["sample_id"], payload["probe_id"],
        payload["prompt_variant_id"], payload["input_condition"], ooh,
    )


def randomize_mcq_options(
    items: list[BenchmarkItem], *, seed: int
) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    """Return MCQ items with deterministic, position-balanced option orders.

    Within each balancing block the target answer positions are a shuffled
    round-robin over ``range(choice_count)``, so per-position counts differ by
    at most one (spec §8). Distractor order is drawn from the same seeded RNG.
    The permutation and seed are stored in item metadata; item and pairing IDs
    are recomputed because the option order changes.
    """
    errors = []
    for item in items:
        if item.question_format != QuestionFormat.MCQ:
            errors.append(f"{item.item_id}: not an mcq item")
        elif item.input_condition != InputCondition.NORMAL_IMAGE:
            errors.append(
                f"{item.item_id}: randomize before deriving controls; "
                f"got {item.input_condition.value}"
            )
    if errors:
        raise SchemaValidationError("mcq randomization", errors)

    blocks: dict[tuple, list[BenchmarkItem]] = defaultdict(list)
    for item in items:
        key = (item.dataset_id, item.split, item.probe_id,
               item.prompt_variant_id, len(item.choices or []))
        blocks[key].append(item)

    rng = random.Random(seed)
    randomized: list[BenchmarkItem] = []
    block_report = {}
    for key in sorted(blocks, key=repr):
        block = sorted(blocks[key], key=lambda item: item.item_id)
        choice_count = key[-1]
        targets = [index % choice_count for index in range(len(block))]
        rng.shuffle(targets)
        position_counts = [0] * choice_count
        for item, target in zip(block, targets):
            payload = item.to_dict()
            payload["metadata"] = copy.deepcopy(payload["metadata"])
            choices = payload["choices"]
            answer = choices[payload["answer_index"]]
            distractors = [choice for index, choice in enumerate(choices)
                           if index != payload["answer_index"]]
            rng.shuffle(distractors)
            new_choices = distractors[:target] + [answer] + distractors[target:]
            payload["choices"] = new_choices
            payload["answer_index"] = target
            payload["metadata"]["option_order"] = {
                "seed": seed,
                "source_item_id": item.item_id,
                "permutation": [choices.index(choice) for choice in new_choices],
            }
            _rebuild_ids(payload)
            randomized.append(BenchmarkItem.from_dict(payload))
            position_counts[target] += 1
        block_report[repr(key)] = position_counts

    report = {
        "option_order_seed": seed,
        "item_count": len(randomized),
        "answer_position_counts": block_report,
    }
    return randomized, report


def derive_short_answer_items(
    items: list[BenchmarkItem],
    *,
    probe_id: str = SHORT_ANSWER_PROBE_ID,
    question_override: Optional[str] = None,
) -> list[BenchmarkItem]:
    """Derive short-answer twins of direct MCQ items (spec §7.2).

    Accepted answers become the canonical concept name plus registered
    aliases, deduplicated after normalization. The MCQ item stays untouched;
    the twin gets its own probe_id and IDs. Question naturalness still goes
    through pilot human QC (B-P0.10) before release.
    """
    errors = []
    for item in items:
        if item.question_format != QuestionFormat.MCQ:
            errors.append(f"{item.item_id}: short-answer twin needs an mcq source")
        elif item.input_condition != InputCondition.NORMAL_IMAGE:
            errors.append(f"{item.item_id}: derive twins from normal_image items")
    if errors:
        raise SchemaValidationError("short-answer derivation", errors)

    derived = []
    for item in items:
        payload = item.to_dict()
        payload["metadata"] = copy.deepcopy(payload["metadata"])
        payload["probe_id"] = probe_id
        payload["probe_family"] = ProbeFamily.P1_DIRECT.value
        payload["question_format"] = QuestionFormat.SHORT_ANSWER.value
        payload["choices"] = None
        payload["answer_index"] = None
        if question_override:
            payload["question"] = question_override

        accepted: list[str] = []
        seen = set()
        for answer in [item.concept_name, *item.concept_aliases]:
            normalized = normalize_choice_text(answer)
            if normalized and normalized not in seen:
                seen.add(normalized)
                accepted.append(answer)
        payload["accepted_answers"] = accepted
        payload["metadata"]["short_answer"] = {
            "source_item_id": item.item_id,
            "source_probe_id": item.probe_id,
        }
        _rebuild_ids(payload)
        derived.append(BenchmarkItem.from_dict(payload))
    return derived
