import unittest

from experiments.intext_unlearning import (
    _apply_chat_template,
    _has_open_think,
    _thinking_label,
    compute_metrics,
)


class FakeThinkingProcessor:
    """Accepts enable_thinking but ignores it — template forces <think>.

    Mirrors the observed Qwen3-VL-2B-Thinking behavior (2026-07-21).
    """

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=None):
        return "<|im_start|>assistant\n<think>\n"


class FakeInstructProcessor:
    """Honors enable_thinking=False: no reasoning block in the template."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=None):
        assert enable_thinking is False
        return "<|im_start|>assistant\n"


class FakeNoFlagProcessor:
    """Older processor without the enable_thinking kwarg (TypeError path)."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        return "<|im_start|>assistant\n"


MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "Q"}]}]


class OpenThinkDetectionTest(unittest.TestCase):

    def test_detects_open_and_closed(self):
        self.assertTrue(_has_open_think("foo<think>\n"))
        self.assertTrue(_has_open_think("a<think>reasoning\nassistant\n"))
        self.assertFalse(_has_open_think("foo<think>\n</think>\n\n"))
        self.assertFalse(_has_open_think("plain assistant\n"))


class ApplyChatTemplateTest(unittest.TestCase):

    def test_thinking_template_gets_empty_block_closed(self):
        text = _apply_chat_template(FakeThinkingProcessor(), MESSAGES)
        self.assertFalse(_has_open_think(text))
        self.assertTrue(text.rstrip().endswith("</think>"))

    def test_instruct_template_untouched(self):
        text = _apply_chat_template(FakeInstructProcessor(), MESSAGES)
        self.assertNotIn("<think>", text)

    def test_missing_flag_falls_through_without_thinking(self):
        text = _apply_chat_template(FakeNoFlagProcessor(), MESSAGES)
        self.assertNotIn("<think>", text)

    def test_rendered_prompt_never_ends_inside_reasoning(self):
        # Whatever the template does, choice_logprob must score the answer
        # position, so the final prompt must not have an open reasoning block.
        for processor in (FakeThinkingProcessor(), FakeInstructProcessor(),
                          FakeNoFlagProcessor()):
            self.assertFalse(
                _has_open_think(_apply_chat_template(processor, MESSAGES))
            )


class ThinkingLabelTest(unittest.TestCase):

    def test_label_only_for_thinking_checkpoints(self):
        self.assertEqual(
            _thinking_label("Qwen/Qwen3-VL-2B-Thinking", "disabled"),
            "Thinking checkpoint (thinking disabled)",
        )
        self.assertEqual(
            _thinking_label("Qwen/Qwen3-VL-2B-Thinking", "enabled"),
            "Thinking checkpoint (thinking enabled)",
        )
        self.assertIsNone(_thinking_label("Qwen/Qwen3-VL-2B-Instruct", "disabled"))
        self.assertIsNone(_thinking_label(None, "disabled"))

    def test_metrics_carry_label_and_scoring_mode(self):
        results = [{
            "split": "test_forget", "condition": "BASELINE_NORMAL",
            "gt_index": 0, "pred_index": 0, "is_correct": True,
            "is_invalid": False,
            "meta_synset": "dog", "meta_superclass": "animal",
        }]
        metrics = compute_metrics(
            results, ["dog"], model_name="Qwen/Qwen3-VL-2B-Thinking",
            thinking_mode="disabled",
        )
        self.assertEqual(metrics["scoring_mode"], "choice_logprob")
        self.assertEqual(metrics["thinking_mode"], "disabled")
        self.assertEqual(metrics["model_variant_note"],
                         "Thinking checkpoint (thinking disabled)")

    def test_instruct_metrics_have_no_variant_note(self):
        metrics = compute_metrics(
            [], [], model_name="Qwen/Qwen3-VL-2B-Instruct",
        )
        self.assertNotIn("model_variant_note", metrics)


if __name__ == "__main__":
    unittest.main()
