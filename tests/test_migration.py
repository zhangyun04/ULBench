import json
import tempfile
import unittest
from pathlib import Path

from ulbench.schema import Split
from ulbench.tools.migrate_legacy_jsonl import (
    MigrationContext,
    main as migration_main,
    migrate_legacy_records,
)
from ulbench.validation import validate_benchmark_items


class LegacyMigrationTest(unittest.TestCase):
    @staticmethod
    def _legacy_records():
        records = []
        choices = ["dog", "cat", "bus", "chair"]
        for answer_index in range(4):
            records.append({
                "id": f"legacy-{answer_index}",
                "image": f"train2017/{answer_index}.jpg",
                "question": "What object is shown?",
                "choices": choices,
                "answer_index": answer_index,
                "forgetting_level": "object",
                "concept_axis": "identity",
                "target_split": "all",
                "meta": {
                    "forget_concept": choices[answer_index],
                    "class_name": choices[answer_index],
                },
            })
        return records

    def test_explicit_migration_produces_balanced_v1_items(self):
        records = self._legacy_records()

        context = MigrationContext(
            dataset_id="coco@2017",
            source_version="2017",
            split=Split.TEST_FORGET,
            license_id="CC-BY-4.0",
            redistribution="allowed",
            provenance_note="Dataset provenance only; no pretraining claim.",
            probe_id="legacy.mcq.identity.v1",
            prompt_variant_id="legacy_default",
        )
        items, report = migrate_legacy_records(records, context)

        self.assertEqual(len(items), 4)
        self.assertEqual(report["migrated_count"], 4)
        self.assertEqual(report["rejected_count"], 0)
        self.assertEqual(items[0].concept_aliases, [])
        validate_benchmark_items(items)

    def test_cli_writes_validated_output_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "legacy.jsonl"
            output_path = root / "v1.jsonl"
            input_path.write_text(
                "".join(
                    json.dumps(record) + "\n" for record in self._legacy_records()
                ),
                encoding="utf-8",
            )
            exit_code = migration_main([
                "--input", str(input_path),
                "--output", str(output_path),
                "--dataset-id", "coco@2017",
                "--source-version", "2017",
                "--split", "test_forget",
                "--license-id", "CC-BY-4.0",
                "--redistribution", "allowed",
                "--provenance-note",
                "Dataset provenance only; no pretraining claim.",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.is_file())
            report = json.loads(
                output_path.with_suffix(".jsonl.migration_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["validation_status"], "passed")
            self.assertEqual(report["migrated_count"], 4)


if __name__ == "__main__":
    unittest.main()
