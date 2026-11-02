import unittest

from validation.evaluation import (
    EVALUATOR_VERSION, answer_is_correct, grade_predictions, normalize_text, summarize_rows,
)


class AnswerGradingTests(unittest.TestCase):
    def test_rejects_empty_and_partial_answers(self):
        for answer in (None, "", "  ", "Par", "ar", "P", "**"):
            with self.subTest(answer=answer):
                self.assertFalse(answer_is_correct(answer, ["Paris"]))

    def test_rejects_negation_and_contradiction(self):
        for answer in ("Não é Paris", "Paris não", "Paris. Mas a resposta correta é Roma.", "Paris ou Roma"):
            with self.subTest(answer=answer):
                self.assertFalse(answer_is_correct(answer, ["Paris"]))

    def test_does_not_match_number_substrings(self):
        self.assertFalse(answer_is_correct("19", ["9"]))
        self.assertFalse(answer_is_correct("900", ["9"]))
        self.assertFalse(answer_is_correct("9 metros", ["9"]))

    def test_no_single_character_alias_substring(self):
        self.assertFalse(answer_is_correct("a resposta é div", ["<a>", "a"]))
        self.assertTrue(answer_is_correct("`<a>`.", ["<a>", "a"]))

    def test_normalizes_historical_answer_formatting(self):
        for answer in ("**Tóquio**", "  TÓQUIO.  ", "**Tóquio", "`Tóquio`", '“Tóquio”'):
            with self.subTest(answer=answer):
                self.assertTrue(answer_is_correct(answer, ["Toquio"]))
        self.assertTrue(answer_is_correct("H₂O", ["H2O"]))

    def test_numeric_equivalence_is_explicit(self):
        self.assertFalse(answer_is_correct("9.0", ["9"]))
        self.assertTrue(answer_is_correct("9.0", ["9"], method="numeric"))
        self.assertFalse(answer_is_correct("19", ["9"], method="numeric"))
        self.assertFalse(answer_is_correct("1,000", ["1"], method="numeric"))
        self.assertFalse(answer_is_correct("nan", ["9"], method="numeric"))

    def test_code_comparison_preserves_case_and_punctuation(self):
        self.assertTrue(answer_is_correct("```js\ntrue\n```", ["true"], method="code"))
        self.assertFalse(answer_is_correct("True", ["true"], method="code"))
        self.assertFalse(answer_is_correct("a", ["<a>"], method="code"))

    def test_rejects_unknown_method_and_string_alias_list(self):
        with self.assertRaises(ValueError):
            answer_is_correct("Paris", ["Paris"], method="substring")
        with self.assertRaises(TypeError):
            answer_is_correct("Paris", "Paris")


class EvaluationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.questions = [{"id": 1, "question": "Capital?", "expected_answer": "Paris", "accepted_answers": ["Paris"]}]
        self.predictions = [{"id": 1, "model_answer": "Paris"}]

    def test_default_score_is_development(self):
        summary = summarize_rows(grade_predictions(self.questions, self.predictions))
        self.assertEqual(summary["evaluation_classification"], "development")
        self.assertFalse(summary["is_held_out_measurement"])
        self.assertEqual(summary["evaluator_version"], EVALUATOR_VERSION)
        self.assertEqual(summary["accuracy"], 1.0)

    def test_refuses_unverified_held_out_claim(self):
        with self.assertRaises(ValueError):
            summarize_rows(grade_predictions(self.questions, self.predictions), classification="held_out")

    def test_held_out_requires_the_verified_test_split(self):
        report = {"verified": True, "classification": "held_out", "evaluated_split": "test", "overlap_count": 0}
        summary = summarize_rows(grade_predictions(self.questions, self.predictions), classification="held_out", split_verification=report)
        self.assertTrue(summary["is_held_out_measurement"])
        report["evaluated_split"] = "development"
        with self.assertRaises(ValueError):
            summarize_rows(grade_predictions(self.questions, self.predictions), classification="held_out", split_verification=report)

    def test_rejects_duplicate_predictions(self):
        with self.assertRaises(ValueError):
            grade_predictions(self.questions, self.predictions * 2)

    def test_rejects_duplicate_questions(self):
        with self.assertRaises(ValueError):
            grade_predictions(self.questions * 2, self.predictions)

    def test_rejects_unknown_and_missing_ids(self):
        with self.assertRaises(ValueError):
            grade_predictions(self.questions, [{"id": 2, "model_answer": "Paris"}])
        with self.assertRaises(ValueError):
            grade_predictions(self.questions, [])

    def test_explicit_incomplete_run_cannot_be_successful(self):
        summary = summarize_rows(grade_predictions(self.questions, [], require_complete=False))
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["accuracy"], 0.0)

    def test_requires_current_boolean_results(self):
        rows = grade_predictions(self.questions, self.predictions)
        rows[0]["is_correct"] = "false"
        with self.assertRaises(ValueError):
            summarize_rows(rows)
        rows[0]["is_correct"] = True
        rows[0]["evaluator_version"] = "legacy-substring"
        with self.assertRaises(ValueError):
            summarize_rows(rows)

    def test_no_empty_set_or_empty_expected_answer(self):
        with self.assertRaises(ValueError):
            grade_predictions([], [])
        with self.assertRaises(ValueError):
            summarize_rows([])
        self.questions[0]["accepted_answers"] = [""]
        with self.assertRaises(ValueError):
            grade_predictions(self.questions, self.predictions)

    def test_legacy_import_reexports_same_evaluator(self):
        from gemma4_theory.utils import eval_utils
        self.assertIs(eval_utils.answer_is_correct, answer_is_correct)
        self.assertIs(eval_utils.normalize_text, normalize_text)


if __name__ == "__main__":
    unittest.main()
