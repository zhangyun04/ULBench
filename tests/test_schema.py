import copy
import unittest

from ulbench.schema import (
    AccessRegime,
    BenchmarkItem,
    InputCondition,
    MethodSpec,
    ModelCapabilities,
    ProbeFamily,
    ProbeSpec,
    QuestionFormat,
    RunManifest,
    SchemaValidationError,
    Split,
    validate_method_model_compatibility,
)


def valid_item_payload(**overrides):
    payload = {
        "schema_version": "1.0.0",
        "item_id": "coco_train_dog_001__legacy_mcq__normal",
        "sample_id": "coco_train_dog_001",
        "dataset_id": "coco@2017",
        "image_id": "coco:train2017/1.jpg",
        "image": "train2017/1.jpg",
        "concept_id": "coco:dog",
        "concept_name": "dog",
        "concept_aliases": ["domestic dog"],
        "forgetting_level": "object",
        "concept_axis": "identity",
        "split": "test_forget",
        "probe_id": "legacy.mcq.identity.v1",
        "probe_family": "P1_DIRECT",
        "question_format": "mcq",
        "prompt_variant_id": "legacy_default",
        "input_condition": "normal_image",
        "question": "What object is shown?",
        "choices": ["dog", "cat", "bus", "chair"],
        "accepted_answers": ["dog", "domestic dog"],
        "answer_index": 0,
        "matched_retain_id": None,
        "source": {
            "dataset_id": "coco",
            "record_id": "1",
            "version": "2017",
        },
        "license": {"id": "CC-BY-4.0", "redistribution": "allowed"},
        "provenance_note": "Dataset provenance only; no pretraining claim.",
        "metadata": {"pairing_id": "coco_train_dog_001:legacy.mcq.identity.v1"},
    }
    payload.update(overrides)
    return payload


class BenchmarkItemSchemaTest(unittest.TestCase):
    def test_round_trip_valid_item(self):
        item = BenchmarkItem.from_dict(valid_item_payload())
        self.assertEqual(item.split, Split.TEST_FORGET)
        self.assertEqual(item.probe_family, ProbeFamily.P1_DIRECT)
        self.assertEqual(item.question_format, QuestionFormat.MCQ)
        self.assertEqual(item.input_condition, InputCondition.NORMAL_IMAGE)
        self.assertEqual(BenchmarkItem.from_dict(item.to_dict()), item)

    def test_missing_required_field_fails_loudly(self):
        payload = valid_item_payload()
        del payload["concept_id"]
        with self.assertRaisesRegex(SchemaValidationError, "concept_id"):
            BenchmarkItem.from_dict(payload)

    def test_invalid_output_is_not_a_benchmark_item_status(self):
        payload = valid_item_payload(response_status="invalid_format")
        with self.assertRaisesRegex(SchemaValidationError, "unknown fields"):
            BenchmarkItem.from_dict(payload)

    def test_no_image_rejects_an_attached_image(self):
        payload = valid_item_payload(input_condition="no_image")
        with self.assertRaisesRegex(SchemaValidationError, "must be null"):
            BenchmarkItem.from_dict(payload)


class CapabilityTest(unittest.TestCase):
    def test_missing_capability_is_explicit(self):
        capabilities = ModelCapabilities(
            supports_images=True,
            supports_logits=False,
            supports_hidden_states=False,
            supports_gradients=False,
            supports_weight_write=False,
            supports_system_prompt=True,
            supports_multi_turn=True,
            is_closed_api=True,
            constraints={},
        )
        method = MethodSpec(
            schema_version="1.0.0",
            method_id="gradient_ascent",
            method_version="1",
            access_regime=AccessRegime.R2_WHITE_BOX,
            method_family="weight_update",
            semantic_label="weight_intervention",
            required_capabilities=["supports_gradients", "supports_weight_write"],
            requires_forget_set=True,
            requires_retain_set=False,
            uses_external_models=False,
            modifies_persistent_state=True,
            tunable_hyperparameters={"learning_rate": [1e-5]},
            selected_hyperparameters={"learning_rate": 1e-5},
            tuning={"split": "train_forget", "budget": 1, "checkpoint_rule": "last"},
            cost_fields=["gpu_hours"],
            metadata={},
        )
        missing = validate_method_model_compatibility(method, capabilities)
        self.assertEqual(missing, ["supports_gradients", "supports_weight_write"])


class ContractSchemaTest(unittest.TestCase):
    def test_probe_spec_round_trip(self):
        probe = ProbeSpec.from_dict({
            "schema_version": "1.0.0",
            "probe_id": "direct.mcq.identity.v1",
            "probe_version": "1",
            "probe_family": "P1_DIRECT",
            "question_format": "mcq",
            "input_condition": "normal_image",
            "allowed_modalities": ["image", "text"],
            "prompt_template": "Q: {question}",
            "prompt_variant_id": "identity_v1",
            "scorer_id": "exact_option",
            "scorer_version": "1",
            "answer_space": {"type": "choice_index"},
            "construction_source": "vqa_gen",
            "seed": 42,
            "applicable_concept_axes": ["identity"],
            "reveals_concept_name": False,
            "attack_family": None,
            "attack_unit_cost": 0,
            "expected_control_ids": ["no_image", "shuffled_image", "option_only"],
            "known_limitations": [],
        })
        self.assertEqual(ProbeSpec.from_dict(probe.to_dict()), probe)

    def test_main_run_manifest_requires_eligibility(self):
        payload = {
            "schema_version": "1.0.0",
            "run_id": "run-1",
            "run_scope": "main",
            "status": "planned",
            "created_at": "2026-07-20T00:00:00Z",
            "completed_at": None,
            "repository": {"url": "https://example.invalid/repo", "git_sha": "abc", "dirty": False},
            "data": {
                "benchmark_version": "1",
                "dataset_version": "1",
                "dataset_hash": "a",
                "split_hash": "b",
                "concept_registry_hash": "c",
                "probe_bank_hash": "d",
                "prompt_bank_hash": "e",
            },
            "model": {
                "model_id": "org/model",
                "model_revision": "rev",
                "processor_revision": "rev",
                "loader": "huggingface",
                "chat_template": "default",
            },
            "model_capabilities": {
                "supports_images": True,
                "supports_logits": True,
                "supports_hidden_states": True,
                "supports_gradients": True,
                "supports_weight_write": True,
                "supports_system_prompt": True,
                "supports_multi_turn": True,
                "is_closed_api": False,
                "constraints": {},
            },
            "model_state_input": "M0",
            "model_state_output": "M_u",
            "method": {"method_id": "noop", "method_version": "1", "config_hash": "f"},
            "access_regime": "R0_BLACK_BOX",
            "capability_validation": {"supported": True, "missing_capabilities": []},
            "seeds": {
                "data_selection": 1,
                "option_order": 2,
                "controls": 3,
                "inference": 4,
                "training": 5,
            },
            "inference": {"decoding": {}, "dtype": "bf16", "device": "cuda"},
            "training": {
                "steps": 0,
                "updated_parameter_count": 0,
                "updated_parameter_fraction": 0.0,
                "tuning_budget": 0,
            },
            "cost": {
                "wall_seconds": 0,
                "gpu_hours": 0,
                "peak_memory_bytes": 0,
                "inference_latency_ms": 0,
            },
            "environment": {},
            "eligibility": {},
            "artifacts": {},
            "failure_counts": {},
        }
        with self.assertRaisesRegex(SchemaValidationError, "manifest_id"):
            RunManifest.from_dict(payload)

        payload["eligibility"] = {
            "manifest_id": "eligibility-1",
            "manifest_hash": "hash",
            "candidate_count": 10,
            "eligible_count": 8,
        }
        manifest = RunManifest.from_dict(payload)
        self.assertEqual(RunManifest.from_dict(manifest.to_dict()), manifest)


if __name__ == "__main__":
    unittest.main()
