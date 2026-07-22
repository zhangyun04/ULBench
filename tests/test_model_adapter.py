import unittest

from ulbench.models import MODEL_INFO_KEYS, ModelAdapter, check_adapter_contract
from ulbench.schema import ModelCapabilities, ResponseStatus
from ulbench.types import CapabilityMismatchError, ProbeRequest, ProbeResponse


def _capabilities(**overrides):
    payload = dict(
        supports_images=True, supports_logits=False,
        supports_hidden_states=False, supports_gradients=False,
        supports_weight_write=False, supports_system_prompt=True,
        supports_multi_turn=True, is_closed_api=False, constraints={},
    )
    payload.update(overrides)
    return ModelCapabilities(**payload)


class FakeAdapter(ModelAdapter):
    """Minimal conformant adapter used for contract testing."""

    def capabilities(self):
        return _capabilities()

    def model_info(self):
        return {
            "model_id": "fake/model", "model_revision": "abc123",
            "processor_revision": "abc123", "loader": "fake",
            "chat_template": "none",
        }

    def generate(self, requests):
        return [
            ProbeResponse(request_id=request.request_id, raw_output="0")
            for request in requests
        ]


class DroppingAdapter(FakeAdapter):
    def generate(self, requests):
        return super().generate(requests)[:-1]


class SilentFailureAdapter(FakeAdapter):
    def generate(self, requests):
        # Empty output without an error status = silent fallback.
        return [ProbeResponse(request_id=request.request_id) for request in requests]


def ten_requests():
    return [
        ProbeRequest(
            request_id=f"r{i}", item_id=f"i{i}", question="q",
            choices=["a", "b", "c", "d"], image_path="x.jpg",
            prompt_text="rendered",
        )
        for i in range(10)
    ]


class AdapterContractTest(unittest.TestCase):

    def test_conformant_adapter_passes(self):
        self.assertEqual(check_adapter_contract(FakeAdapter(), ten_requests()), [])

    def test_dropped_response_is_reported(self):
        violations = check_adapter_contract(DroppingAdapter(), ten_requests())
        self.assertTrue(any("9 responses" in v for v in violations))

    def test_silent_empty_output_is_reported(self):
        violations = check_adapter_contract(SilentFailureAdapter(), ten_requests())
        self.assertEqual(len(violations), 10)
        self.assertIn("explicit", violations[0])

    def test_explicit_model_error_is_accepted(self):
        class ErrorAdapter(FakeAdapter):
            def generate(self, requests):
                return [
                    ProbeResponse(
                        request_id=request.request_id,
                        response_status=ResponseStatus.MODEL_ERROR,
                        error="image_not_found",
                    )
                    for request in requests
                ]

        self.assertEqual(check_adapter_contract(ErrorAdapter(), ten_requests()), [])

    def test_score_options_gated_by_logit_capability(self):
        with self.assertRaises(CapabilityMismatchError) as ctx:
            FakeAdapter().score_options(ten_requests())
        self.assertEqual(ctx.exception.missing_capabilities, ["supports_logits"])


class ConcreteAdapterDeclarationTest(unittest.TestCase):
    """Static declarations must be valid without loading any weights."""

    def test_huggingface_adapter_declarations(self):
        from ulbench.models.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter("Qwen/Qwen3-VL-4B-Instruct")
        capabilities = adapter.capabilities()
        capabilities.validate()
        self.assertTrue(capabilities.supports_logits)
        self.assertFalse(capabilities.is_closed_api)
        info = adapter.model_info()
        for key in MODEL_INFO_KEYS:
            self.assertIn(key, info)
        self.assertEqual(info["model_revision"], "unpinned")
        self.assertIsNone(adapter.model)

    def test_internvl_adapter_declarations(self):
        from ulbench.models.internvl import InternVLAdapter

        adapter = InternVLAdapter("OpenGVLab/InternVL3-2B", model_revision="deadbeef")
        capabilities = adapter.capabilities()
        capabilities.validate()
        self.assertFalse(capabilities.supports_logits)
        self.assertEqual(adapter.model_info()["model_revision"], "deadbeef")

    def test_create_adapter_routing(self):
        from ulbench.models import create_adapter
        from ulbench.models.huggingface import HuggingFaceAdapter
        from ulbench.models.internvl import InternVLAdapter

        self.assertIsInstance(
            create_adapter("OpenGVLab/InternVL3-2B"), InternVLAdapter
        )
        adapter = create_adapter("Qwen/Qwen3-VL-4B-Instruct")
        self.assertIsInstance(adapter, HuggingFaceAdapter)
        self.assertNotIsInstance(adapter, InternVLAdapter)

    def test_thinking_detection_matches_legacy(self):
        from ulbench.models.huggingface import HuggingFaceAdapter

        self.assertTrue(
            HuggingFaceAdapter("Qwen/Qwen3-VL-4B-Thinking").is_thinking_checkpoint
        )
        self.assertFalse(
            HuggingFaceAdapter("Qwen/Qwen3-VL-4B-Instruct").is_thinking_checkpoint
        )

    def test_thinking_mode_defaults_to_disabled(self):
        from ulbench.models.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter("Qwen/Qwen3-VL-4B-Thinking")
        # A thinking checkpoint is scored with thinking disabled by default.
        self.assertEqual(adapter.thinking_mode, "disabled")
        self.assertTrue(adapter.is_thinking_checkpoint)
        with self.assertRaises(ValueError):
            HuggingFaceAdapter("Qwen/Qwen3-VL-4B-Thinking", thinking_mode="bogus")

    def test_thinking_mode_routes_through_create_adapter(self):
        from ulbench.models import create_adapter

        adapter = create_adapter(
            "Qwen/Qwen3-VL-4B-Thinking", thinking_mode="enabled"
        )
        self.assertEqual(adapter.thinking_mode, "enabled")


if __name__ == "__main__":
    unittest.main()
