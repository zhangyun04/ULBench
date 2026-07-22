import unittest

from ulbench.ids import option_order_hash, stable_id
from ulbench.probes.controls import (
    build_control_variants,
    build_shuffled_image_assignment,
)
from ulbench.schema import (
    SCHEMA_VERSION,
    BenchmarkItem,
    InputCondition,
    ProbeFamily,
    SchemaValidationError,
)
from ulbench.validation import validate_benchmark_items


def make_item(index: int, concept: str) -> BenchmarkItem:
    sample_id = f"coco_test_{concept}_{index:03d}"
    choices = ["cat", "dog", "horse", "pizza"]
    ooh = option_order_hash(choices)
    return BenchmarkItem.from_dict({
        "schema_version": SCHEMA_VERSION,
        "item_id": stable_id("item", sample_id, "legacy.mcq.identity.v1",
                             "legacy_default", "normal_image", ooh),
        "sample_id": sample_id,
        "dataset_id": "coco@2017",
        "image_id": f"coco:{concept}/{index}.jpg",
        "image": f"test2017/{concept}_{index}.jpg",
        "concept_id": f"coco:{concept}",
        "concept_name": concept,
        "concept_aliases": [],
        "forgetting_level": "object",
        "concept_axis": "identity",
        "split": "test_forget",
        "probe_id": "legacy.mcq.identity.v1",
        "probe_family": ProbeFamily.P1_DIRECT.value,
        "question_format": "mcq",
        "prompt_variant_id": "legacy_default",
        "input_condition": "normal_image",
        "question": "What is the main object in this image?",
        "choices": choices,
        "accepted_answers": [concept],
        "answer_index": choices.index(concept),
        "matched_retain_id": None,
        "source": {"dataset_id": "coco@2017", "record_id": sample_id,
                   "version": "v1"},
        "license": {"id": "cc-by-4.0", "redistribution": "metadata_only"},
        "provenance_note": "test fixture",
        "metadata": {
            "pairing_id": stable_id("pair", sample_id, "legacy.mcq.identity.v1",
                                    "legacy_default", ooh),
            "option_order_hash": ooh,
        },
    })


def make_items():
    items = []
    for concept in ("cat", "dog", "horse"):
        for index in range(3):
            items.append(make_item(index, concept))
    return items


class ShuffledAssignmentTest(unittest.TestCase):

    def test_derangement_properties(self):
        items = make_items()
        assignment = build_shuffled_image_assignment(items, seed=7)
        self.assertEqual(set(assignment), {item.item_id for item in items})
        by_id = {item.item_id: item for item in items}
        for item_id, donor in assignment.items():
            item = by_id[item_id]
            self.assertNotEqual(donor.concept_id, item.concept_id)
            self.assertNotEqual(donor.image_id, item.image_id)

    def test_deterministic_given_seed(self):
        items = make_items()
        first = build_shuffled_image_assignment(items, seed=7)
        second = build_shuffled_image_assignment(items, seed=7)
        self.assertEqual(
            {k: v.item_id for k, v in first.items()},
            {k: v.item_id for k, v in second.items()},
        )

    def test_single_concept_fails_loudly(self):
        items = [make_item(index, "cat") for index in range(4)]
        with self.assertRaises(SchemaValidationError):
            build_shuffled_image_assignment(items, seed=7)

    def test_dominant_concept_fails_loudly(self):
        items = [make_item(index, "cat") for index in range(5)]
        items.append(make_item(0, "dog"))
        with self.assertRaises(SchemaValidationError):
            build_shuffled_image_assignment(items, seed=7)


class ControlVariantTest(unittest.TestCase):

    def test_full_suite_passes_core_control_validation(self):
        items, report = build_control_variants(make_items(), seed=7)
        validate_benchmark_items(
            items, check_answer_balance=False, require_core_controls=True
        )
        self.assertEqual(report["source_count"], 9)
        self.assertEqual(report["variant_count"], 27)

    def test_variant_fields(self):
        source = make_items()
        items, _ = build_control_variants(source, seed=7)
        derived = [item for item in items
                   if item.input_condition != InputCondition.NORMAL_IMAGE]
        for item in derived:
            self.assertEqual(item.probe_family, ProbeFamily.P0_CONTROL)
            self.assertEqual(item.metadata["control"]["control_seed"], 7)
        no_image = [item for item in derived
                    if item.input_condition == InputCondition.NO_IMAGE]
        option_only = [item for item in derived
                       if item.input_condition == InputCondition.OPTION_ONLY]
        shuffled = [item for item in derived
                    if item.input_condition == InputCondition.SHUFFLED_IMAGE]
        self.assertTrue(all(item.image is None for item in no_image))
        self.assertTrue(all(item.question for item in no_image))
        self.assertTrue(all(item.question == "" for item in option_only))
        for item in shuffled:
            self.assertNotEqual(item.metadata["donor_image_id"], item.image_id)
            self.assertNotEqual(
                item.metadata["control"]["donor_concept_id"], item.concept_id
            )

    def test_pairing_and_ground_truth_preserved(self):
        source = make_items()
        items, _ = build_control_variants(source, seed=7)
        by_pairing = {}
        for item in items:
            by_pairing.setdefault(item.metadata["pairing_id"], []).append(item)
        for pairing_items in by_pairing.values():
            self.assertEqual(len(pairing_items), 4)
            answers = {tuple(item.accepted_answers) for item in pairing_items}
            self.assertEqual(len(answers), 1)
            indices = {item.answer_index for item in pairing_items}
            self.assertEqual(len(indices), 1)

    def test_rejects_non_normal_input(self):
        items, _ = build_control_variants(make_items(), seed=7)
        with self.assertRaises(SchemaValidationError):
            build_control_variants(items, seed=7)

    def test_question_only_rejected_for_mcq(self):
        with self.assertRaises(SchemaValidationError):
            build_control_variants(
                make_items(), seed=7, include_question_only=True
            )


if __name__ == "__main__":
    unittest.main()
