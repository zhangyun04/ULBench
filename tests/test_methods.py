import unittest

from experiments.intext_unlearning import build_prompt
from ulbench.audits.oracle_controls import (
    ORACLE_CONTROLS,
    OracleHardControl,
    OracleReverseControl,
)
from ulbench.methods.base import UnlearningMethod
from ulbench.methods.noop import NoOpMethod
from ulbench.methods.prompt_suppression import (
    LEGACY_CONDITION_MAP,
    PromptSuppressionMethod,
)
from ulbench.schema import (
    SCHEMA_VERSION,
    MethodSpec,
    ModelCapabilities,
)
from ulbench.types import CapabilityMismatchError, ProbeRequest


ITEM = {
    "question": "What is the main object in this image?",
    "choices": ["cat", "dog", "horse", "pizza"],
    "meta": {"class_name": "dog"},
}
FORGET_CLASSES = ["dog", "cat", "horse"]


def injected_lines(condition, is_forget_split, forget_classes):
    """Lines the legacy build_prompt injects for *condition* vs baseline."""
    baseline = build_prompt(ITEM, "BASELINE_NORMAL", is_forget_split,
                            forget_classes).split("\n")
    conditioned = build_prompt(ITEM, condition, is_forget_split,
                               forget_classes).split("\n")
    prefix = 0
    while (prefix < len(baseline) and prefix < len(conditioned)
           and baseline[prefix] == conditioned[prefix]):
        prefix += 1
    suffix = 0
    while (suffix < len(baseline) - prefix
           and baseline[len(baseline) - 1 - suffix]
           == conditioned[len(conditioned) - 1 - suffix]):
        suffix += 1
    return conditioned[prefix:len(conditioned) - suffix]


class PromptSuppressionEquivalenceTest(unittest.TestCase):
    """The migrated instructions must be byte-identical to legacy conditions."""

    def test_soft_matches_legacy_with_class_list(self):
        method = PromptSuppressionMethod("soft", FORGET_CLASSES)
        self.assertEqual(
            injected_lines("UNLEARN_SOFT", True, FORGET_CLASSES),
            [method.instruction_line(), ""],
        )

    def test_medium_matches_legacy_with_class_list(self):
        method = PromptSuppressionMethod("medium", FORGET_CLASSES)
        self.assertEqual(
            injected_lines("UNLEARN_MEDIUM", True, FORGET_CLASSES),
            [method.instruction_line(), ""],
        )

    def test_single_target_fallback_matches_legacy(self):
        # Legacy falls back to the per-item target when no class list given.
        for variant, condition in LEGACY_CONDITION_MAP.items():
            method = PromptSuppressionMethod(variant)
            self.assertEqual(
                injected_lines(condition, True, None),
                [method.instruction_line(target_concept="dog"), ""],
            )

    def test_applies_to_retain_split_like_legacy(self):
        method = PromptSuppressionMethod("soft", FORGET_CLASSES)
        self.assertEqual(
            injected_lines("UNLEARN_SOFT", False, FORGET_CLASSES),
            [method.instruction_line(), ""],
        )

    def test_transform_input_appends_instruction(self):
        method = PromptSuppressionMethod("soft", FORGET_CLASSES)
        request = ProbeRequest(
            request_id="r1", item_id="i1", question="q",
            choices=list(ITEM["choices"]), image_path="x.jpg",
        )
        out = method.transform_input(request)
        self.assertEqual(out.method_instructions, [method.instruction_line()])

    def test_missing_concepts_fails_loudly(self):
        method = PromptSuppressionMethod("soft")
        with self.assertRaises(ValueError):
            method.instruction_line()


class OracleControlEquivalenceTest(unittest.TestCase):

    def test_hard_matches_legacy(self):
        control = OracleHardControl()
        self.assertEqual(
            injected_lines("ORACLE_HARD", True, FORGET_CLASSES),
            [control.instruction_line("dog"), ""],
        )

    def test_reverse_matches_legacy(self):
        control = OracleReverseControl()
        self.assertEqual(
            injected_lines("ORACLE_REVERSE", True, FORGET_CLASSES),
            [control.instruction_line("dog"), ""],
        )

    def test_legacy_adds_nothing_on_retain_split(self):
        for condition in ("ORACLE_HARD", "ORACLE_REVERSE"):
            self.assertEqual(injected_lines(condition, False, FORGET_CLASSES), [])
        for control in ORACLE_CONTROLS.values():
            self.assertTrue(control.forget_split_only)

    def test_transform_input_skips_retain_split(self):
        control = OracleHardControl()
        request = ProbeRequest(
            request_id="r1", item_id="i1", question="q", choices=None,
            image_path="x.jpg",
            context={"split": "test_retain", "target_concept": "dog"},
        )
        self.assertEqual(control.transform_input(request).method_instructions, [])

    def test_oracle_controls_are_not_unlearning_methods(self):
        # Frozen decision #9: oracles never enter method rankings.
        for control in ORACLE_CONTROLS.values():
            self.assertNotIsInstance(control, UnlearningMethod)
            self.assertEqual(control.semantic_label,
                             "instruction_following_control")


class MethodContractTest(unittest.TestCase):

    def test_noop_spec_valid_and_hooks_identity(self):
        method = NoOpMethod()
        method.spec.validate()
        request = ProbeRequest(
            request_id="r1", item_id="i1", question="q", choices=None,
            image_path=None,
        )
        self.assertIs(method.transform_input(request), request)
        self.assertEqual(request.method_instructions, [])
        self.assertEqual(method.metadata()["method_id"], "no_op")

    def test_prompt_suppression_specs_valid(self):
        for variant in ("soft", "medium"):
            spec = PromptSuppressionMethod(variant, FORGET_CLASSES).spec
            spec.validate()
            self.assertEqual(spec.semantic_label, "behavioral_suppression")
            self.assertEqual(spec.access_regime.value, "R0_BLACK_BOX")

    def test_capability_mismatch_produces_spec_record(self):
        spec = MethodSpec.from_dict({
            "schema_version": SCHEMA_VERSION,
            "method_id": "fake_weight_method",
            "method_version": "1.0.0",
            "access_regime": "R2_WHITE_BOX",
            "method_family": "training",
            "semantic_label": "weight_intervention",
            "required_capabilities": ["supports_gradients", "supports_weight_write"],
            "requires_forget_set": True,
            "requires_retain_set": True,
            "uses_external_models": False,
            "modifies_persistent_state": True,
            "tunable_hyperparameters": {},
            "selected_hyperparameters": {},
            "tuning": {},
            "cost_fields": ["gpu_hours"],
            "metadata": {},
        })
        method = UnlearningMethod(spec)
        closed_api = ModelCapabilities(
            supports_images=True, supports_logits=False,
            supports_hidden_states=False, supports_gradients=False,
            supports_weight_write=False, supports_system_prompt=True,
            supports_multi_turn=True, is_closed_api=True, constraints={},
        )
        with self.assertRaises(CapabilityMismatchError) as ctx:
            method.validate_against(closed_api)
        self.assertEqual(ctx.exception.to_record(), {
            "status": "unsupported",
            "reason_code": "CAPABILITY_MISMATCH",
            "missing_capabilities": ["supports_gradients", "supports_weight_write"],
        })

    def test_compatible_combination_passes(self):
        open_weights = ModelCapabilities(
            supports_images=True, supports_logits=True,
            supports_hidden_states=True, supports_gradients=True,
            supports_weight_write=True, supports_system_prompt=True,
            supports_multi_turn=True, is_closed_api=False, constraints={},
        )
        PromptSuppressionMethod("soft", FORGET_CLASSES).validate_against(open_weights)
        NoOpMethod().validate_against(open_weights)


if __name__ == "__main__":
    unittest.main()
