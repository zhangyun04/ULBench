import unittest

from ulbench.eligibility import (
    CI_BELOW_FLOOR,
    INSUFFICIENT_SAMPLES,
    LOW_NORMAL_ACCESS,
    LOW_VISUAL_GAP,
    MISSING_CONTROL,
    VARIANT_INSTABILITY,
    EligibilityThresholds,
    build_eligibility_manifest,
    summarize_coverage,
    write_eligibility_manifest,
)
from ulbench.schema import QuestionFormat, SchemaValidationError


def context():
    return {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "model_revision": "abc123",
        "processor_revision": "abc123",
        "benchmark_version": "v1",
        "probe_bank_hash": "hash-p",
        "prompt_bank_hash": "hash-b",
        "scorer_versions": {"mcq_exact": "1.0.0"},
        "data_hashes": {"items": "hash-d"},
        "code_git_sha": "deadbeef",
    }


def cell_records(concept, question_format, condition, n_correct, n_wrong,
                 variant="v1", choice_count=4):
    records = []
    for status, count in (("correct", n_correct), ("incorrect", n_wrong)):
        for _ in range(count):
            records.append({
                "concept_id": concept,
                "question_format": question_format,
                "input_condition": condition,
                "prompt_variant_id": variant,
                "response_status": status,
                "choice_count": choice_count if question_format == "mcq" else None,
            })
    return records


def concept_records(concept, normal_acc=0.9, control_acc=0.25, n=60,
                    formats=("mcq", "short_answer")):
    records = []
    correct = round(normal_acc * n)
    control_correct = round(control_acc * n)
    for fmt in formats:
        records += cell_records(concept, fmt, "normal_image",
                                correct, n - correct)
        for condition in ("no_image", "option_only", "shuffled_image"):
            records += cell_records(concept, fmt, condition,
                                    control_correct, n - control_correct)
    return records


class EligibilityGateTest(unittest.TestCase):

    def manifest(self, records, **kwargs):
        return build_eligibility_manifest(
            records, context=context(), seed=13, **kwargs
        )

    def test_strong_concept_is_eligible(self):
        manifest = self.manifest(concept_records("coco:dog"))
        concept = manifest["concepts"]["coco:dog"]
        self.assertTrue(concept["eligible"])
        self.assertEqual(concept["failure_codes"], [])
        self.assertEqual(manifest["eligible_count"], 1)
        self.assertEqual(manifest["coverage"], 1.0)

    def test_low_normal_access_fails(self):
        manifest = self.manifest(concept_records("coco:cat", normal_acc=0.40))
        codes = manifest["concepts"]["coco:cat"]["failure_codes"]
        self.assertIn(f"mcq:{LOW_NORMAL_ACCESS}", codes)
        self.assertIn(f"short_answer:{LOW_NORMAL_ACCESS}", codes)

    def test_low_visual_gap_fails(self):
        manifest = self.manifest(
            concept_records("coco:cat", normal_acc=0.9, control_acc=0.85)
        )
        codes = manifest["concepts"]["coco:cat"]["failure_codes"]
        self.assertIn(f"mcq:{LOW_VISUAL_GAP}", codes)

    def test_insufficient_samples_fails(self):
        manifest = self.manifest(concept_records("coco:cat", n=20))
        codes = manifest["concepts"]["coco:cat"]["failure_codes"]
        self.assertIn(f"mcq:{INSUFFICIENT_SAMPLES}", codes)

    def test_missing_control_fails(self):
        records = cell_records("coco:cat", "mcq", "normal_image", 54, 6)
        records += cell_records("coco:cat", "mcq", "no_image", 15, 45)
        records += cell_records("coco:cat", "mcq", "option_only", 15, 45)
        # shuffled_image entirely absent
        manifest = self.manifest(
            records, required_formats=(QuestionFormat.MCQ,)
        )
        codes = manifest["concepts"]["coco:cat"]["failure_codes"]
        self.assertIn(f"mcq:{MISSING_CONTROL}", codes)

    def test_ci_below_floor_fails_with_binary_choices(self):
        # 2 choices → chance floor 0.5; accuracy 0.62 passes tau but its
        # bootstrap lower bound stays below the floor at n=60.
        records = []
        records += cell_records("coco:cat", "mcq", "normal_image", 37, 23,
                                choice_count=2)
        for condition in ("no_image", "option_only", "shuffled_image"):
            records += cell_records("coco:cat", "mcq", condition, 5, 55,
                                    choice_count=2)
        manifest = self.manifest(records, required_formats=(QuestionFormat.MCQ,))
        codes = manifest["concepts"]["coco:cat"]["failure_codes"]
        self.assertIn(f"mcq:{CI_BELOW_FLOOR}", codes)
        self.assertNotIn(f"mcq:{LOW_NORMAL_ACCESS}", codes)

    def test_variant_instability_fails(self):
        records = []
        records += cell_records("coco:cat", "mcq", "normal_image", 55, 5,
                                variant="v1")
        records += cell_records("coco:cat", "mcq", "normal_image", 34, 26,
                                variant="v2")  # 0.57 vs 0.92: spread > 0.15
        for condition in ("no_image", "option_only", "shuffled_image"):
            records += cell_records("coco:cat", "mcq", condition, 10, 110)
        manifest = self.manifest(records, required_formats=(QuestionFormat.MCQ,))
        codes = manifest["concepts"]["coco:cat"]["failure_codes"]
        self.assertIn(f"mcq:{VARIANT_INSTABILITY}", codes)

    def test_refusals_do_not_count_as_access(self):
        records = concept_records("coco:dog")
        for record in records:
            if (record["input_condition"] == "normal_image"
                    and record["response_status"] == "correct"):
                record["response_status"] = "refusal"
        manifest = self.manifest(records)
        self.assertFalse(manifest["concepts"]["coco:dog"]["eligible"])


class ManifestContractTest(unittest.TestCase):

    def test_missing_context_key_fails_loudly(self):
        broken = context()
        del broken["code_git_sha"]
        with self.assertRaises(SchemaValidationError):
            build_eligibility_manifest(
                concept_records("coco:dog"), context=broken, seed=13
            )

    def test_deterministic_given_seed(self):
        first = build_eligibility_manifest(
            concept_records("coco:dog"), context=context(), seed=13
        )
        second = build_eligibility_manifest(
            concept_records("coco:dog"), context=context(), seed=13
        )
        first.pop("created_at")
        second.pop("created_at")
        self.assertEqual(first, second)

    def test_coverage_summary_and_write(self):
        import tempfile
        from pathlib import Path

        records = concept_records("coco:dog") + concept_records(
            "coco:cat", normal_acc=0.30
        )
        manifest = build_eligibility_manifest(
            records, context=context(), seed=13
        )
        summary = summarize_coverage(manifest)
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["eligible_count"], 1)
        self.assertIn(f"mcq:{LOW_NORMAL_ACCESS}",
                      summary["failure_code_counts"])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_eligibility_manifest(manifest, tmp)
            self.assertEqual(
                path, Path(tmp) / "eligible_concepts" / "abc123.json"
            )
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
