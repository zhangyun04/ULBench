import json
import tempfile
import unittest
from pathlib import Path

from experiments.intext_unlearning import build_prompt
from tests.test_controls import make_item, make_items
from ulbench.methods.noop import NoOpMethod
from ulbench.methods.base import UnlearningMethod
from ulbench.methods.prompt_suppression import PromptSuppressionMethod
from ulbench.models.base import ModelAdapter
from ulbench.probes.controls import build_control_variants
from ulbench.probes.direct import derive_short_answer_items
from ulbench.probes.render import render_prompt
from ulbench.runner import EvaluationRunner, RunConfig
from ulbench.schema import (
    SCHEMA_VERSION,
    MethodSpec,
    ModelCapabilities,
    ResponseStatus,
    RunManifest,
)
from ulbench.types import ProbeResponse


class RenderEquivalenceTest(unittest.TestCase):

    def test_mcq_normal_matches_legacy_baseline(self):
        item = make_item(0, "dog")
        legacy_item = {
            "question": item.question,
            "choices": item.choices,
            "meta": {"class_name": item.concept_name},
        }
        self.assertEqual(
            render_prompt(item),
            build_prompt(legacy_item, "BASELINE_NORMAL", True),
        )

    def test_mcq_with_suppression_matches_legacy_condition(self):
        item = make_item(0, "dog")
        legacy_item = {
            "question": item.question,
            "choices": item.choices,
            "meta": {"class_name": item.concept_name},
        }
        forget = ["dog", "cat"]
        method = PromptSuppressionMethod("soft", forget)
        self.assertEqual(
            render_prompt(item, [method.instruction_line()]),
            build_prompt(legacy_item, "UNLEARN_SOFT", True, forget),
        )

    def test_option_only_has_no_question_line(self):
        items, _ = build_control_variants([make_item(0, "dog"),
                                           make_item(0, "cat")], seed=3)
        option_only = next(
            item for item in items
            if item.input_condition.value == "option_only"
        )
        text = render_prompt(option_only)
        self.assertNotIn("Q:", text)
        self.assertIn("0) cat", text)

    def test_short_answer_render(self):
        twin = derive_short_answer_items([make_item(0, "cat")])[0]
        text = render_prompt(twin)
        self.assertIn("Q:", text)
        self.assertNotIn("0)", text)
        self.assertIn("short name only", text)


class OracleAdapter(ModelAdapter):
    """Answers every request with the ground-truth index (from context)."""

    gpu_ids = None

    def capabilities(self):
        return ModelCapabilities(
            supports_images=True, supports_logits=True,
            supports_hidden_states=False, supports_gradients=False,
            supports_weight_write=False, supports_system_prompt=True,
            supports_multi_turn=False, is_closed_api=False, constraints={},
        )

    def model_info(self):
        return {"model_id": "fake/oracle", "model_revision": "r1",
                "processor_revision": "r1", "loader": "fake",
                "chat_template": "none"}

    def generate(self, requests):
        return [
            ProbeResponse(request_id=request.request_id,
                          raw_output=str(request.context["answer_index"]),
                          latency_ms=1.0)
            for request in requests
        ]

    def score_options(self, requests, option_count=4):
        return [
            ProbeResponse(request_id=request.request_id,
                          raw_output=str(request.context["answer_index"]),
                          prediction=str(request.context["answer_index"]),
                          latency_ms=1.0)
            for request in requests
        ]


def run_config(tmp, run_id="test_run"):
    return RunConfig(run_id=run_id, out_dir=tmp, image_root="/img", seed=7)


class RunnerEndToEndTest(unittest.TestCase):

    def _items(self):
        mcq, _ = build_control_variants(make_items(), seed=3)
        short = derive_short_answer_items(make_items())
        return mcq + short

    def test_artifacts_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = EvaluationRunner(OracleAdapter(), NoOpMethod(),
                                      run_config(tmp))
            manifest = runner.run(self._items())

            out = Path(tmp)
            for name in ("run_manifest.json", "results.jsonl",
                         "metrics.json", "failures.jsonl"):
                self.assertTrue((out / name).exists(), name)
            self.assertTrue((out / "logs").is_dir())

            # Manifest revalidates and reports completion + hashes.
            reloaded = RunManifest.from_dict(
                json.loads((out / "run_manifest.json").read_text())
            )
            self.assertEqual(reloaded.status.value, "completed")
            self.assertEqual(
                manifest["artifacts"]["results"]["path"], "results.jsonl"
            )
            self.assertEqual(manifest["failure_counts"], {})

            records = [json.loads(line) for line in
                       (out / "results.jsonl").read_text().splitlines()]
            # 9 mcq × 4 conditions + 9 short answers
            self.assertEqual(len(records), 45)
            self.assertTrue(all(
                record["response_status"] == "correct" for record in records
                if record["question_format"] == "mcq"
            ))
            metrics = json.loads((out / "metrics.json").read_text())
            self.assertEqual(metrics["overall"]["attempted"], 45)
            self.assertIn("test_forget|mcq|normal_image",
                          metrics["by_condition"])

    def test_short_answer_scored_with_alias_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            # OracleAdapter answers digits; short answers become incorrect,
            # proving format routing to the alias scorer, not digit parsing.
            runner = EvaluationRunner(OracleAdapter(), NoOpMethod(),
                                      run_config(tmp))
            runner.run(derive_short_answer_items(make_items()))
            records = [json.loads(line) for line in
                       (Path(tmp) / "results.jsonl").read_text().splitlines()]
            self.assertTrue(all(
                record["scorer_id"] == "short_answer_alias"
                for record in records
            ))

    def test_method_instructions_reach_prompt(self):
        captured = []

        class CapturingAdapter(OracleAdapter):
            def score_options(self, requests, option_count=4):
                captured.extend(request.prompt_text for request in requests)
                return super().score_options(requests, option_count)

        with tempfile.TemporaryDirectory() as tmp:
            method = PromptSuppressionMethod("soft", ["dog"])
            runner = EvaluationRunner(CapturingAdapter(), method,
                                      run_config(tmp))
            runner.run(make_items())
        self.assertTrue(captured)
        self.assertTrue(all(
            "You should not identify or reveal" in prompt
            for prompt in captured
        ))

    def test_unsupported_combination_writes_manifest_only(self):
        spec = MethodSpec.from_dict({
            "schema_version": SCHEMA_VERSION,
            "method_id": "fake_r2", "method_version": "1.0.0",
            "access_regime": "R2_WHITE_BOX", "method_family": "training",
            "semantic_label": "weight_intervention",
            "required_capabilities": ["supports_gradients"],
            "requires_forget_set": True, "requires_retain_set": True,
            "uses_external_models": False, "modifies_persistent_state": True,
            "tunable_hyperparameters": {}, "selected_hyperparameters": {},
            "tuning": {}, "cost_fields": [], "metadata": {},
        })
        with tempfile.TemporaryDirectory() as tmp:
            runner = EvaluationRunner(OracleAdapter(), UnlearningMethod(spec),
                                      run_config(tmp))
            manifest = runner.run(make_items())
            self.assertEqual(manifest["status"], "unsupported")
            self.assertEqual(
                manifest["capability_validation"]["missing_capabilities"],
                ["supports_gradients"],
            )
            out = Path(tmp)
            self.assertTrue((out / "run_manifest.json").exists())
            # Spec §4.4: no empty metric files for unsupported combinations.
            self.assertFalse((out / "metrics.json").exists())
            self.assertFalse((out / "results.jsonl").exists())

    def test_adapter_errors_recorded_not_raised(self):
        class FailingAdapter(OracleAdapter):
            def score_options(self, requests, option_count=4):
                return [
                    ProbeResponse(request_id=request.request_id,
                                  response_status=ResponseStatus.MODEL_ERROR,
                                  error="boom")
                    for request in requests
                ]

        with tempfile.TemporaryDirectory() as tmp:
            runner = EvaluationRunner(FailingAdapter(), NoOpMethod(),
                                      run_config(tmp))
            manifest = runner.run(make_items())
            self.assertEqual(manifest["failure_counts"], {"model_error": 9})
            failures = [json.loads(line) for line in
                        (Path(tmp) / "failures.jsonl").read_text().splitlines()]
            self.assertEqual(len(failures), 9)
            self.assertTrue(all(f["error"] == "boom" for f in failures))
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())
            self.assertEqual(metrics["overall"]["access_rate"], 0.0)
            self.assertIsNone(metrics["overall"]["conditional_accuracy"])


if __name__ == "__main__":
    unittest.main()
