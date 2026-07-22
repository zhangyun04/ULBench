import unittest

from tests.test_controls import make_item, make_items
from ulbench.probes.controls import build_control_variants
from ulbench.probes.direct import (
    derive_short_answer_items,
    randomize_mcq_options,
)
from ulbench.probes.scorers import score_mcq, score_short_answer
from ulbench.schema import (
    QuestionFormat,
    ResponseStatus,
    SchemaValidationError,
)
from ulbench.validation import validate_benchmark_items


def make_block(count=8):
    # One balancing block: same dataset/split/probe/variant/choice count.
    items = []
    for concept in ("cat", "dog"):
        for index in range(count // 2):
            items.append(make_item(index, concept))
    return items


class McqRandomizationTest(unittest.TestCase):

    def test_positions_balanced_within_block(self):
        randomized, report = randomize_mcq_options(make_block(8), seed=11)
        counts = list(report["answer_position_counts"].values())[0]
        self.assertEqual(sum(counts), 8)
        self.assertLessEqual(max(counts) - min(counts), 1)
        validate_benchmark_items(randomized, check_answer_balance=True)

    def test_deterministic_and_answer_preserved(self):
        first, _ = randomize_mcq_options(make_block(8), seed=11)
        second, _ = randomize_mcq_options(make_block(8), seed=11)
        self.assertEqual([item.to_dict() for item in first],
                         [item.to_dict() for item in second])
        for item in first:
            self.assertEqual(item.choices[item.answer_index],
                             item.accepted_answers[0])

    def test_permutation_recorded_and_ids_rebuilt(self):
        source = make_block(4)
        randomized, _ = randomize_mcq_options(source, seed=11)
        by_sample = {item.sample_id: item for item in source}
        for item in randomized:
            order = item.metadata["option_order"]
            self.assertEqual(order["seed"], 11)
            original = by_sample[item.sample_id]
            self.assertEqual(order["source_item_id"], original.item_id)
            restored = [item.choices[order["permutation"].index(index)]
                        for index in range(len(item.choices))]
            self.assertEqual(restored, original.choices)
            if item.choices != original.choices:
                self.assertNotEqual(item.item_id, original.item_id)

    def test_controls_compose_after_randomization(self):
        randomized, _ = randomize_mcq_options(make_block(8), seed=11)
        items, _ = build_control_variants(randomized, seed=7)
        validate_benchmark_items(
            items, check_answer_balance=False, require_core_controls=True
        )

    def test_rejects_non_normal_items(self):
        items, _ = build_control_variants(make_items(), seed=7)
        with self.assertRaises(SchemaValidationError):
            randomize_mcq_options(items, seed=11)


class ShortAnswerDerivationTest(unittest.TestCase):

    def test_twin_fields_and_alias_dedup(self):
        source = make_item(0, "cat")
        payload = source.to_dict()
        payload["concept_aliases"] = ["house cat", "Cat", "domestic cat"]
        from ulbench.schema import BenchmarkItem
        source = BenchmarkItem.from_dict(payload)

        twin = derive_short_answer_items([source])[0]
        self.assertEqual(twin.question_format, QuestionFormat.SHORT_ANSWER)
        self.assertIsNone(twin.choices)
        self.assertIsNone(twin.answer_index)
        # "Cat" collapses into canonical "cat" after normalization.
        self.assertEqual(twin.accepted_answers,
                         ["cat", "house cat", "domestic cat"])
        self.assertNotEqual(twin.item_id, source.item_id)
        self.assertNotEqual(twin.metadata["pairing_id"],
                            source.metadata["pairing_id"])
        twin.validate()


class McqScorerTest(unittest.TestCase):

    def setUp(self):
        self.item = make_item(0, "dog")  # answer_index = 1

    def test_digit_answers(self):
        self.assertEqual(score_mcq("1", self.item).response_status,
                         ResponseStatus.CORRECT)
        self.assertEqual(score_mcq("The answer is 1.", self.item).response_status,
                         ResponseStatus.CORRECT)
        self.assertEqual(score_mcq("0", self.item).response_status,
                         ResponseStatus.INCORRECT)

    def test_out_of_range_digit_is_invalid(self):
        result = score_mcq("7", self.item)
        self.assertEqual(result.response_status, ResponseStatus.INVALID_FORMAT)

    def test_choice_text_fallback(self):
        result = score_mcq("Dog", self.item)
        self.assertEqual(result.response_status, ResponseStatus.CORRECT)
        self.assertEqual(result.detail["match_mode"], "choice_text")

    def test_refusal_distinct_from_invalid(self):
        refusal = score_mcq("I cannot identify this image.", self.item)
        self.assertEqual(refusal.response_status, ResponseStatus.REFUSAL)
        invalid = score_mcq("It's a lovely picture!", self.item)
        self.assertEqual(invalid.response_status, ResponseStatus.INVALID_FORMAT)

    def test_parseable_answer_beats_refusal_phrase(self):
        result = score_mcq("I cannot be sure, but 1.", self.item)
        self.assertEqual(result.response_status, ResponseStatus.CORRECT)


class ShortAnswerScorerTest(unittest.TestCase):

    def setUp(self):
        source = make_item(0, "cat")
        payload = source.to_dict()
        payload["concept_aliases"] = ["domestic cat"]
        from ulbench.schema import BenchmarkItem
        self.item = derive_short_answer_items(
            [BenchmarkItem.from_dict(payload)]
        )[0]

    def test_exact_and_normalized_matches(self):
        exact = score_short_answer("cat", self.item)
        self.assertEqual(exact.response_status, ResponseStatus.CORRECT)
        self.assertTrue(exact.detail["exact_match"])
        cased = score_short_answer("  CAT ", self.item)
        self.assertEqual(cased.response_status, ResponseStatus.CORRECT)
        self.assertFalse(cased.detail["exact_match"])
        alias = score_short_answer("Domestic Cat.", self.item)
        self.assertEqual(alias.response_status, ResponseStatus.CORRECT)
        self.assertEqual(alias.detail["normalized_match"], "domestic cat")

    def test_wrong_refusal_and_empty(self):
        wrong = score_short_answer("a small tiger", self.item)
        self.assertEqual(wrong.response_status, ResponseStatus.INCORRECT)
        refusal = score_short_answer("I'm unable to identify animals.", self.item)
        self.assertEqual(refusal.response_status, ResponseStatus.REFUSAL)
        empty = score_short_answer("   ", self.item)
        self.assertEqual(empty.response_status, ResponseStatus.INVALID_FORMAT)


if __name__ == "__main__":
    unittest.main()
