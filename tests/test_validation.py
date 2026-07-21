import copy
import unittest

from tests.test_schema import valid_item_payload
from ulbench.schema import BenchmarkItem, SchemaValidationError
from ulbench.validation import validate_benchmark_items, validate_split_registry_data


def balanced_items():
    items = []
    for answer_index in range(4):
        payload = valid_item_payload(
            item_id=f"item-{answer_index}",
            sample_id=f"sample-{answer_index}",
            image_id=f"image-{answer_index}",
            image=f"images/{answer_index}.jpg",
            choices=["dog", "cat", "bus", "chair"],
            answer_index=answer_index,
            accepted_answers=[["dog", "cat", "bus", "chair"][answer_index]],
            source={
                "dataset_id": "coco",
                "record_id": str(answer_index),
                "version": "2017",
            },
            metadata={"pairing_id": f"sample-{answer_index}:legacy"},
        )
        items.append(BenchmarkItem.from_dict(payload))
    return items


class DatasetValidationTest(unittest.TestCase):
    def test_balanced_non_overlapping_items_pass(self):
        validate_benchmark_items(balanced_items())

    def test_answer_position_imbalance_fails(self):
        items = balanced_items()
        for item in items[1:]:
            item.choices[0], item.choices[item.answer_index] = (
                item.choices[item.answer_index],
                item.choices[0],
            )
            item.answer_index = 0
        with self.assertRaisesRegex(SchemaValidationError, "answer-position imbalance"):
            validate_benchmark_items(items)

    def test_train_test_sample_overlap_fails(self):
        items = balanced_items()
        duplicate = copy.deepcopy(items[0])
        duplicate.item_id = "test-copy"
        duplicate.split = duplicate.split.__class__.TRAIN_FORGET
        with self.assertRaisesRegex(SchemaValidationError, "train/test sample overlap"):
            validate_benchmark_items(items + [duplicate], check_answer_balance=False)


class RegistryValidationTest(unittest.TestCase):
    def test_wrong_encoded_k_fails(self):
        registry = {
            "defaults": {"seed": 42},
            "datasets": {
                "coco": {
                    "input_jsonl": "missing.jsonl",
                    "experiments": [
                        {"name": "coco_rk10_s42", "mode": "random_k", "k": 1}
                    ],
                }
            },
        }
        with self.assertRaisesRegex(SchemaValidationError, "encodes k=10"):
            validate_split_registry_data(registry, check_paths=False)


if __name__ == "__main__":
    unittest.main()
