"""
tests/test_evaluator.py
-----------------------
Unit tests for the answer evaluation module.
Tests the fast-path evaluators (no LLM call needed).
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.evaluator import (
    evaluate_true_false_fast,
    evaluate_mcq_fast,
    _fallback_evaluation,
)


class TestTrueFalseEvaluator:
    def test_correct_true(self):
        r = evaluate_true_false_fast("True", "True")
        assert r["is_correct"] is True
        assert r["score"] == 10

    def test_correct_false(self):
        r = evaluate_true_false_fast("False", "False")
        assert r["is_correct"] is True
        assert r["score"] == 10

    def test_wrong_answer(self):
        r = evaluate_true_false_fast("True", "False")
        assert r["is_correct"] is False
        assert r["score"] == 0

    def test_case_insensitive(self):
        r = evaluate_true_false_fast("true", "True")
        assert r["is_correct"] is True

    def test_yes_maps_to_true(self):
        r = evaluate_true_false_fast("yes", "True")
        assert r["is_correct"] is True

    def test_no_maps_to_false(self):
        r = evaluate_true_false_fast("no", "False")
        assert r["is_correct"] is True


class TestMCQEvaluator:
    def test_exact_match(self):
        r = evaluate_mcq_fast("Option B", "Option B")
        assert r["is_correct"] is True
        assert r["score"] == 10

    def test_wrong_option(self):
        r = evaluate_mcq_fast("Option A", "Option B")
        assert r["is_correct"] is False
        assert r["score"] == 0

    def test_case_insensitive(self):
        r = evaluate_mcq_fast("option b", "Option B")
        assert r["is_correct"] is True

    def test_feedback_mentions_correct_on_wrong(self):
        r = evaluate_mcq_fast("Option A", "Option C")
        assert "Option C" in r["feedback"]


class TestFallbackEvaluator:
    def test_identical_answers_score_ten(self):
        r = _fallback_evaluation("the mitochondria", "the mitochondria")
        assert r["score"] == 10
        assert r["is_correct"] is True

    def test_empty_correct_answer(self):
        r = _fallback_evaluation("something", "")
        assert r["score"] == 5  # graceful fallback

    def test_no_overlap_scores_zero(self):
        r = _fallback_evaluation("cats and dogs", "nuclear fission")
        assert r["score"] == 0
        assert r["is_correct"] is False

    def test_partial_overlap(self):
        r = _fallback_evaluation("the nucleus contains DNA", "nucleus stores DNA and RNA")
        assert 0 < r["score"] < 10
