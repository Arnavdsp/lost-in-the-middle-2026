"""Tests for the statistics layer.

Every test here uses synthetic data with a KNOWN answer, so the test asserts
correctness rather than merely "it ran". Statistics code that silently returns
plausible-looking wrong numbers is the single most dangerous kind of bug in
this repository.
"""

import numpy as np
import pytest

from litm2026.stats import (
    accuracy,
    bootstrap_ci,
    holm_bonferroni,
    mcnemar_exact,
    mde_paired_binary,
    paired_bootstrap_diff,
    u_shape_severity,
)


class TestAccuracy:
    def test_all_correct(self):
        assert accuracy([1.0] * 10) == 1.0

    def test_all_wrong(self):
        assert accuracy([0.0] * 10) == 0.0

    def test_half(self):
        assert accuracy([1.0, 0.0, 1.0, 0.0]) == 0.5


class TestBootstrapCI:
    def test_point_estimate_is_the_sample_mean(self):
        scores = [1.0] * 60 + [0.0] * 40
        result = bootstrap_ci(scores, seed=0)
        assert result.point == pytest.approx(0.6)

    def test_interval_brackets_the_point_estimate(self):
        scores = [1.0] * 60 + [0.0] * 40
        result = bootstrap_ci(scores, seed=0)
        assert result.low <= result.point <= result.high

    def test_zero_variance_gives_degenerate_interval(self):
        result = bootstrap_ci([1.0] * 50, seed=0)
        assert result.low == pytest.approx(1.0)
        assert result.high == pytest.approx(1.0)

    def test_larger_n_gives_tighter_interval(self):
        # The whole reason n matters. If this fails, the CIs are meaningless.
        small = bootstrap_ci([1.0, 0.0] * 25, seed=0)
        large = bootstrap_ci([1.0, 0.0] * 500, seed=0)
        assert (large.high - large.low) < (small.high - small.low)

    def test_is_deterministic_under_a_fixed_seed(self):
        scores = list(np.random.default_rng(1).integers(0, 2, 200).astype(float))
        assert bootstrap_ci(scores, seed=7) == bootstrap_ci(scores, seed=7)


class TestPairedBootstrap:
    def test_identical_arms_have_zero_difference(self):
        scores = [1.0, 0.0] * 50
        result = paired_bootstrap_diff(scores, scores, seed=0)
        assert result.diff == pytest.approx(0.0)

    def test_detects_a_large_real_difference(self):
        # NOTE THE SIGN CONVENTION: the function returns mean(b) - mean(a),
        # with `a` as the reference condition. Getting this backwards would
        # flip the direction of every reported effect, so it is asserted here
        # rather than left to a reader of the docstring.
        a = [1.0] * 90 + [0.0] * 10   # reference, accuracy 0.90
        b = [1.0] * 40 + [0.0] * 60   # comparison, accuracy 0.40
        result = paired_bootstrap_diff(a, b, seed=0)
        assert result.diff == pytest.approx(-0.5, abs=1e-9)
        assert result.high < 0.0  # CI excludes zero, on the negative side

    def test_sign_flips_when_arguments_swap(self):
        a = [1.0] * 90 + [0.0] * 10
        b = [1.0] * 40 + [0.0] * 60
        forward = paired_bootstrap_diff(a, b, seed=0)
        reverse = paired_bootstrap_diff(b, a, seed=0)
        assert forward.diff == pytest.approx(-reverse.diff, abs=1e-9)

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            paired_bootstrap_diff([1.0, 0.0], [1.0], seed=0)


class TestMcNemar:
    def test_no_discordant_pairs_is_not_significant(self):
        scores = [1.0, 0.0] * 40
        result = mcnemar_exact(scores, scores)
        assert result.p_value == pytest.approx(1.0)

    def test_all_discordant_in_one_direction_is_significant(self):
        a = [1.0] * 30
        b = [0.0] * 30
        result = mcnemar_exact(a, b)
        assert result.p_value < 0.001

    def test_symmetric_discordance_is_not_significant(self):
        a = [1.0] * 10 + [0.0] * 10
        b = [0.0] * 10 + [1.0] * 10
        result = mcnemar_exact(a, b)
        assert result.p_value > 0.5

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            mcnemar_exact([1.0], [1.0, 0.0])


class TestHolmBonferroni:
    def test_corrected_thresholds_are_stricter_than_alpha(self):
        results = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.04}, alpha=0.05)
        # Holm inflates each raw p-value; adjusted >= raw, always.
        for result in results:
            assert result.p_adjusted >= result.p_value

    def test_a_single_test_is_uncorrected(self):
        results = holm_bonferroni({"only": 0.04}, alpha=0.05)
        assert results[0].p_adjusted == pytest.approx(0.04)
        assert results[0].reject is True

    def test_borderline_p_fails_once_corrected(self):
        # p=0.04 would pass alone; with five tests it should not.
        results = holm_bonferroni(
            {"a": 0.04, "b": 0.30, "c": 0.40, "d": 0.50, "e": 0.60}, alpha=0.05
        )
        assert next(r for r in results if r.label == "a").reject is False

    def test_all_tests_are_returned(self):
        assert len(holm_bonferroni({f"t{i}": 0.5 for i in range(7)})) == 7


class TestUShapeSeverity:
    """The headline statistic. Defined as:

        (mean(first, last) - min(middle)) / mean(first, last)
    """

    def test_flat_curve_has_zero_severity(self):
        assert u_shape_severity([0.7, 0.7, 0.7, 0.7, 0.7]) == pytest.approx(0.0)

    def test_pronounced_u_has_positive_severity(self):
        assert u_shape_severity([0.8, 0.6, 0.4, 0.6, 0.8]) > 0.4

    def test_matches_the_formula_by_hand(self):
        # edges (0.8 + 0.8)/2 = 0.8 ; min middle = 0.4 ; (0.8-0.4)/0.8 = 0.5
        assert u_shape_severity([0.8, 0.6, 0.4, 0.6, 0.8]) == pytest.approx(0.5)

    def test_inverted_u_gives_negative_severity(self):
        # A hump rather than a dip. Should not be reported as a U-shape.
        assert u_shape_severity([0.4, 0.7, 0.9, 0.7, 0.4]) < 0.0

    def test_deeper_dip_scores_higher(self):
        shallow = u_shape_severity([0.8, 0.75, 0.7, 0.75, 0.8])
        deep = u_shape_severity([0.8, 0.5, 0.2, 0.5, 0.8])
        assert deep > shallow

    def test_too_few_positions_rejected(self):
        with pytest.raises(ValueError):
            u_shape_severity([0.8, 0.8])


class TestMinimumDetectableEffect:
    """MDE is what makes an honest small-n claim possible.

    Report it next to the result: "n=150 can detect a difference of X or
    larger; the observed difference is Y." That sentence pre-empts the
    "your sample is small" objection instead of conceding it.
    """

    def test_larger_n_detects_smaller_effects(self):
        assert mde_paired_binary(1000) < mde_paired_binary(100)

    def test_is_a_sensible_proportion(self):
        assert 0.0 < mde_paired_binary(150) < 1.0

    def test_higher_power_demands_a_larger_effect(self):
        assert mde_paired_binary(150, power=0.95) > mde_paired_binary(150, power=0.80)
