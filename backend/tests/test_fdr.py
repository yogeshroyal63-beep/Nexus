"""
Tests for Benjamini-Hochberg FDR correction.

These tests validate the implementation against KNOWN mathematical
properties of the BH procedure, not just "does it run without crashing".
"""
import numpy as np
import pytest

from app.causal.fdr import benjamini_hochberg, expected_false_positive_rate


class TestBenjaminiHochberg:
    def test_empty_input(self):
        assert benjamini_hochberg([]) == []

    def test_single_pvalue_below_q_is_rejected(self):
        results = benjamini_hochberg([0.01], q=0.05)
        assert len(results) == 1
        assert results[0].rejected is True

    def test_single_pvalue_above_q_is_not_rejected(self):
        results = benjamini_hochberg([0.5], q=0.05)
        assert results[0].rejected is False

    def test_all_significant_pvalues_all_rejected(self):
        # All p-values tiny — should all survive correction.
        p_values = [0.001, 0.002, 0.0005, 0.003]
        results = benjamini_hochberg(p_values, q=0.05)
        assert all(r.rejected for r in results)

    def test_all_insignificant_pvalues_none_rejected(self):
        p_values = [0.6, 0.7, 0.8, 0.9]
        results = benjamini_hochberg(p_values, q=0.05)
        assert not any(r.rejected for r in results)

    def test_preserves_input_order(self):
        # p-values deliberately out of sorted order in the input
        p_values = [0.04, 0.001, 0.5, 0.02]
        results = benjamini_hochberg(p_values, q=0.05)
        assert len(results) == 4
        assert results[0].raw_p_value == 0.04
        assert results[1].raw_p_value == 0.001
        assert results[2].raw_p_value == 0.5
        assert results[3].raw_p_value == 0.02

    def test_textbook_example(self):
        # Classic worked example: 10 p-values, verify against hand-computed BH.
        # p-values sorted: 0.001 0.008 0.039 0.041 0.042 0.06 0.074 0.205 0.212 0.216
        # m=10, q=0.05 -> threshold_i = (i/10)*0.05
        # i=1: 0.005  p=0.001 <= 0.005  YES
        # i=2: 0.010  p=0.008 <= 0.010  YES
        # i=3: 0.015  p=0.039 <= 0.015  NO
        # i=4: 0.020  p=0.041 <= 0.020  NO
        # i=5: 0.025  p=0.042 <= 0.025  NO
        # ... none after i=2 pass, but BH takes the LARGEST passing k.
        # Since only i=1,2 individually pass and there's no larger k where
        # p(k) <= threshold(k), largest_k = 2.
        p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
        results = benjamini_hochberg(p_values, q=0.05)
        rejected_flags = [r.rejected for r in results]
        assert rejected_flags == [True, True, False, False, False, False, False, False, False, False]

    def test_largest_k_rule_rescues_borderline_cases(self):
        # This is the KEY property that distinguishes BH from naive
        # "adjust and compare to q": BH takes the LARGEST k satisfying
        # p(k) <= (k/m)*q, which can reject hypotheses whose raw p-value
        # alone wouldn't clear a Bonferroni-style bar.
        # m=4, q=0.05: thresholds are 0.0125, 0.025, 0.0375, 0.05
        # p-values:     0.01,   0.02,   0.03,   0.20
        # i=1: 0.01 <= 0.0125 YES
        # i=2: 0.02 <= 0.025  YES
        # i=3: 0.03 <= 0.0375 YES  <- would FAIL a flat Bonferroni cutoff of 0.0125
        # i=4: 0.20 <= 0.05   NO
        p_values = [0.01, 0.02, 0.03, 0.20]
        results = benjamini_hochberg(p_values, q=0.05)
        assert results[2].rejected is True, (
            "BH should reject the 3rd-ranked p-value via the step-up rule "
            "even though 0.03 > flat alpha/m = 0.0125"
        )
        assert results[3].rejected is False

    def test_adjusted_pvalues_are_monotone_nondecreasing_in_rank(self):
        # Fundamental BH property: adjusted p-values must be non-decreasing
        # as the raw p-value increases (after sorting).
        p_values = [0.2, 0.001, 0.15, 0.03, 0.5, 0.008]
        results = benjamini_hochberg(p_values)
        sorted_by_raw = sorted(results, key=lambda r: r.raw_p_value)
        adjusted_in_rank_order = [r.adjusted_p_value for r in sorted_by_raw]
        for i in range(len(adjusted_in_rank_order) - 1):
            assert adjusted_in_rank_order[i] <= adjusted_in_rank_order[i + 1] + 1e-9

    def test_adjusted_pvalues_never_exceed_one(self):
        p_values = [0.9, 0.95, 0.99, 0.999]
        results = benjamini_hochberg(p_values)
        assert all(r.adjusted_p_value <= 1.0 for r in results)

    def test_rank_assignment_correct(self):
        p_values = [0.5, 0.01, 0.3]
        results = benjamini_hochberg(p_values)
        # p=0.01 should be rank 1, p=0.3 rank 2, p=0.5 rank 3
        by_pvalue = {r.raw_p_value: r.rank for r in results}
        assert by_pvalue[0.01] == 1
        assert by_pvalue[0.3] == 2
        assert by_pvalue[0.5] == 3

    def test_reduces_false_discoveries_vs_uncorrected_on_null_data(self):
        """
        The core empirical claim this module exists to satisfy: on data
        where NOTHING is actually drifted (all null hypotheses true),
        FDR correction should reject far fewer hypotheses than naive
        per-test alpha=0.05 thresholding, across many repeated trials.
        """
        rng = np.random.default_rng(7)
        n_tests = 20
        n_trials = 200
        uncorrected_false_positives = 0
        corrected_false_positives = 0

        for _ in range(n_trials):
            # Under the null, p-values are uniform(0,1) by definition.
            p_values = rng.uniform(0, 1, size=n_tests).tolist()

            uncorrected_false_positives += sum(1 for p in p_values if p <= 0.05)

            results = benjamini_hochberg(p_values, q=0.05)
            corrected_false_positives += sum(1 for r in results if r.rejected)

        # Uncorrected should average close to n_tests * 0.05 per trial.
        avg_uncorrected = uncorrected_false_positives / n_trials
        avg_corrected = corrected_false_positives / n_trials

        assert avg_uncorrected == pytest.approx(n_tests * 0.05, abs=0.5)
        # FDR-corrected should be meaningfully lower under the null.
        assert avg_corrected < avg_uncorrected


class TestExpectedFalsePositiveRate:
    def test_zero_tests(self):
        assert expected_false_positive_rate(0) == 0.0

    def test_single_test_equals_alpha(self):
        assert expected_false_positive_rate(1, alpha=0.05) == pytest.approx(0.05)

    def test_grows_with_test_count(self):
        rate_2 = expected_false_positive_rate(2, alpha=0.05)
        rate_8 = expected_false_positive_rate(8, alpha=0.05)
        rate_20 = expected_false_positive_rate(20, alpha=0.05)
        assert rate_2 < rate_8 < rate_20

    def test_eight_tests_matches_hand_calculation(self):
        # 1 - 0.95^8 = 0.33657...
        assert expected_false_positive_rate(8, alpha=0.05) == pytest.approx(0.3366, abs=0.001)

    def test_never_exceeds_one(self):
        assert expected_false_positive_rate(1000, alpha=0.05) <= 1.0
