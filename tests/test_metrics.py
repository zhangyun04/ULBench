import unittest

from ulbench.metrics.accounting import accounting_by_group, response_accounting
from ulbench.metrics.forgetting import forgetting_effect, matched_retain_fidelity
from ulbench.metrics.leakage import worst_case_leakage
from ulbench.metrics.statistics import bootstrap_ci, paired_bootstrap_diff_ci
from ulbench.schema import SchemaValidationError


class ResponseAccountingTest(unittest.TestCase):

    def test_two_access_views_and_rates(self):
        statuses = (["correct"] * 4 + ["incorrect"] * 2 + ["refusal"] * 2
                    + ["invalid_format"] + ["model_error"])
        accounting = response_accounting(statuses)
        self.assertEqual(accounting["attempted"], 10)
        self.assertEqual(accounting["scorable"], 6)
        self.assertAlmostEqual(accounting["access_rate"], 0.4)
        self.assertAlmostEqual(accounting["conditional_accuracy"], 4 / 6)
        self.assertAlmostEqual(accounting["coverage"], 0.6)
        self.assertAlmostEqual(accounting["refusal_rate"], 0.2)
        self.assertAlmostEqual(accounting["error_rate"], 0.1)

    def test_coverage_collapse_yields_null_conditional(self):
        accounting = response_accounting(["refusal"] * 5)
        self.assertEqual(accounting["access_rate"], 0.0)
        self.assertIsNone(accounting["conditional_accuracy"])
        self.assertEqual(accounting["coverage"], 0.0)

    def test_grouping(self):
        records = [
            {"probe_id": "p1", "response_status": "correct"},
            {"probe_id": "p1", "response_status": "incorrect"},
            {"probe_id": "p2", "response_status": "refusal"},
        ]
        grouped = accounting_by_group(records, ("probe_id",))
        self.assertAlmostEqual(grouped[("p1",)]["access_rate"], 0.5)
        self.assertEqual(grouped[("p2",)]["refusal_rate"], 1.0)


def attack_record(sample_id, status, probe="direct", cost=1):
    return {
        "sample_id": sample_id,
        "probe_id": probe,
        "response_status": status,
        "attack_unit_cost": cost,
    }


class WorstCaseLeakageTest(unittest.TestCase):

    def test_indicator_categories(self):
        records = [
            # s1 leaks through an adversarial probe despite direct failure.
            attack_record("s1", "incorrect"),
            attack_record("s1", "correct", probe="paraphrase", cost=2),
            # s2 contained: scorable attempts all fail (refusal mixed in).
            attack_record("s2", "incorrect"),
            attack_record("s2", "refusal"),
            # s3 fully unscorable → null.
            attack_record("s3", "refusal"),
            attack_record("s3", "invalid_format"),
        ]
        result = worst_case_leakage(records, attack_set_id="B1")
        self.assertAlmostEqual(result["wcl"], 0.5)
        self.assertEqual(result["samples"], {
            "total": 3, "leaked": 1, "contained": 1, "unscorable_null": 1,
        })
        self.assertEqual(result["null_sample_ids"], ["s3"])
        self.assertAlmostEqual(result["scorable_sample_coverage"], 2 / 3)
        self.assertEqual(result["attack_budget_spent"], 7)
        self.assertEqual(result["attack_set_id"], "B1")

    def test_all_null_gives_null_wcl(self):
        result = worst_case_leakage(
            [attack_record("s1", "refusal")], attack_set_id="B1"
        )
        self.assertIsNone(result["wcl"])

    def test_missing_cost_fails(self):
        record = attack_record("s1", "correct")
        del record["attack_unit_cost"]
        with self.assertRaises(SchemaValidationError):
            worst_case_leakage([record], attack_set_id="B1")


def paired_record(sample_id, status, prediction=None):
    return {
        "sample_id": sample_id,
        "probe_id": "p1",
        "prompt_variant_id": "v1",
        "input_condition": "normal_image",
        "response_status": status,
        "prediction": prediction,
    }


class ForgettingEffectTest(unittest.TestCase):

    def test_access_and_clean_views_diverge_under_refusals(self):
        m0 = [paired_record(f"s{i}", "correct") for i in range(8)]
        m0 += [paired_record("s8", "incorrect"), paired_record("s9", "incorrect")]
        # Mu refuses half the previously-correct items and stays correct on 4.
        mu = [paired_record(f"s{i}", "correct") for i in range(4)]
        mu += [paired_record(f"s{i}", "refusal") for i in range(4, 8)]
        mu += [paired_record("s8", "incorrect"), paired_record("s9", "incorrect")]

        effect = forgetting_effect(m0, mu)
        self.assertEqual(effect["pair_count"], 10)
        self.assertAlmostEqual(effect["fe_access"], 0.8 - 0.4)
        # Clean view only sees pairs scorable in both states: 4/6 vs 4/6...
        self.assertEqual(effect["clean_pair_count"], 6)
        self.assertAlmostEqual(effect["fe_clean"], 4 / 6 - 4 / 6)
        self.assertAlmostEqual(effect["deltas"]["refusal_rate"], 0.4)
        self.assertAlmostEqual(effect["deltas"]["coverage"], -0.4)

    def test_unpaired_records_fail_loudly(self):
        m0 = [paired_record("s1", "correct")]
        mu = [paired_record("s2", "correct")]
        with self.assertRaises(SchemaValidationError):
            forgetting_effect(m0, mu)

    def test_duplicate_pairs_fail_loudly(self):
        m0 = [paired_record("s1", "correct"), paired_record("s1", "incorrect")]
        with self.assertRaises(SchemaValidationError):
            forgetting_effect(m0, [paired_record("s1", "correct")])


class MatchedRetainFidelityTest(unittest.TestCase):

    def test_transitions_and_consistency(self):
        m0 = [
            paired_record("r1", "correct", prediction="cat"),
            paired_record("r2", "correct", prediction="dog"),
            paired_record("r3", "incorrect", prediction="horse"),
            paired_record("r4", "incorrect", prediction="pizza"),
        ]
        mu = [
            paired_record("r1", "correct", prediction="cat"),
            paired_record("r2", "refusal", prediction=None),
            paired_record("r3", "correct", prediction="cow"),
            paired_record("r4", "incorrect", prediction="pizza"),
        ]
        fidelity = matched_retain_fidelity(m0, mu)
        self.assertAlmostEqual(fidelity["retain_fidelity"], 0.5)
        self.assertEqual(fidelity["transitions"], {
            "kept_correct": 1, "lost_correct": 1, "gained_correct": 1,
        })
        self.assertEqual(fidelity["output_consistency"]["comparable_pairs"], 3)
        self.assertAlmostEqual(
            fidelity["output_consistency"]["consistent_rate"], 2 / 3
        )
        self.assertAlmostEqual(fidelity["deltas"]["refusal_rate"], 0.25)


class BootstrapTest(unittest.TestCase):

    def test_ci_brackets_mean_and_is_deterministic(self):
        values = [1.0] * 70 + [0.0] * 30
        first = bootstrap_ci(values, seed=5)
        second = bootstrap_ci(values, seed=5)
        self.assertEqual(first, second)
        self.assertLess(first["low"], 0.7)
        self.assertGreater(first["high"], 0.7)
        self.assertAlmostEqual(first["mean"], 0.7)
        self.assertGreater(first["low"], 0.55)

    def test_paired_diff(self):
        before = [1.0] * 8 + [0.0] * 2
        after = [1.0] * 4 + [0.0] * 6
        result = paired_bootstrap_diff_ci(before, after, seed=5)
        self.assertAlmostEqual(result["mean"], 0.4)
        self.assertAlmostEqual(result["mean_first"], 0.8)
        self.assertAlmostEqual(result["mean_second"], 0.4)
        with self.assertRaises(ValueError):
            paired_bootstrap_diff_ci([1.0], [1.0, 0.0], seed=5)


if __name__ == "__main__":
    unittest.main()
