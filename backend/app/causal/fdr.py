"""
Multiple Hypothesis Testing Correction — Benjamini-Hochberg FDR procedure.

THE GAP THIS CLOSES:

When the causal isolator tests N upstream ancestor nodes for drift, each
node's KS-test uses an independent significance threshold (alpha = 0.05).
That is correct for testing ONE hypothesis. It is wrong for testing N of
them at once.

The probability that at least one node looks "drifted" by pure chance when
NOTHING is actually wrong grows with N:

    P(at least one false positive) = 1 - (1 - alpha)^N

For a lineage graph with 8 upstream ancestor nodes, that is
1 - 0.95^8 ≈ 34%. One time in three, a completely healthy pipeline would
still throw at least one node into the causal isolator's candidate pool
purely from sampling noise — and a noise-drifted candidate can then win
the intervention-delta ranking and get reported as a "root cause" that
was never actually there.

Benjamini & Hochberg (1995), "Controlling the False Discovery Rate: A
Practical and Powerful Approach to Multiple Testing", JRSS-B 57(1):289-300
is the standard fix used across genomics, econometrics, and production
ML monitoring for exactly this scenario: many simultaneous hypothesis
tests, want to bound the *expected proportion of false discoveries*
among the ones we call significant (not just each test's own alpha).

Procedure:
  1. Sort the m p-values ascending: p(1) <= p(2) <= ... <= p(m)
  2. Find the largest k such that p(k) <= (k / m) * Q
  3. Reject the null (declare "real drift", not noise) for all tests
     1..k. Every other test, however small its raw p-value looked in
     isolation, is not treated as significant once the family-wise
     comparison is accounted for.

This module is deliberately generic (operates on a list of p-values) so
it can be reused anywhere multiple simultaneous statistical tests are
run — currently: causal isolator candidate screening.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FDRResult:
    """Per-hypothesis outcome of a Benjamini-Hochberg correction pass."""

    raw_p_value: float
    adjusted_p_value: float
    rejected: bool  # True = passes FDR-corrected significance (real signal)
    rank: int  # 1-indexed rank among the sorted p-values


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[FDRResult]:
    """
    Apply the Benjamini-Hochberg step-up procedure to a family of p-values.

    Args:
        p_values: raw p-values from m simultaneous hypothesis tests, in
            whatever order the caller wants results back in.
        q: desired false discovery rate (fraction of "significant" results
            that are expected to be false positives). 0.05 is standard.

    Returns:
        One FDRResult per input p-value, in the SAME order as the input,
        so callers can zip results back onto their original candidate list.

    Edge cases:
        - Empty input -> empty output.
        - m == 1 -> BH reduces to the raw p-value vs q (equivalent to
          uncorrected testing, as it should for a single hypothesis).
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort with original-index tracking so we can restore input order.
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])

    # Step 1: find the largest k such that p(k) <= (k/m) * q.
    largest_k = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        threshold = (rank / m) * q
        if p <= threshold:
            largest_k = rank  # keep updating; BH takes the LARGEST such k

    # Step 2: compute the monotone adjusted p-values (Benjamini-Yekutieli
    # style running minimum from the largest p-value down), so
    # adjusted_p_value is non-decreasing as raw p-value increases and can
    # be compared directly against q by the caller if desired.
    adjusted = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx, p = indexed[rank - 1]
        candidate = min(1.0, p * m / rank)
        running_min = min(running_min, candidate)
        adjusted[rank - 1] = running_min

    results_by_original_index: dict[int, FDRResult] = {}
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        results_by_original_index[orig_idx] = FDRResult(
            raw_p_value=p,
            adjusted_p_value=round(adjusted[rank - 1], 6),
            rejected=rank <= largest_k,
            rank=rank,
        )

    # Restore caller's original ordering.
    return [results_by_original_index[i] for i in range(m)]


def expected_false_positive_rate(n_tests: int, alpha: float = 0.05) -> float:
    """
    Family-wise false positive probability if n_tests were each run at
    the given per-test alpha WITHOUT correction. Used to surface, in the
    UI/report, exactly how much risk the FDR step is removing for the
    current graph size — not just applying the correction silently.
    """
    if n_tests <= 0:
        return 0.0
    return 1.0 - (1.0 - alpha) ** n_tests
