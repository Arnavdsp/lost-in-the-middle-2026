"""Statistics for the replication: bootstrap CIs, paired tests, power, U-shape index.

Everything in this module is written to be *explained*, not just run. If you are
reading this in an interview, the four things worth knowing are:

1. **All confidence intervals are percentile bootstrap over questions**, 10,000
   resamples, seeded. Binomial/Wald intervals would be defensible for a single
   accuracy but not for the U-shape index, which is a ratio of minima and means --
   so one method is used throughout rather than two.

2. **Comparisons between positions are paired.** The same questions are scored at
   every gold position (see :mod:`litm2026.data`), so an unpaired test throws away
   the pairing and loses most of the power. Two paired procedures are provided:
   McNemar's *exact* binomial test (no chi-square approximation, which matters at
   n=150 where discordant counts are small) and a paired bootstrap on the
   difference of accuracies. They answer slightly different questions and are
   reported together.

3. **Multiple comparisons are corrected with Holm-Bonferroni.** Comparing 5
   positions against the first position is 4 tests per model; without correction
   the family-wise error rate is about 19%, not 5%.

4. **The U-shape index is defined once, in code**, as
   ``(mean(first, last) - min(middle)) / mean(first, last)``, so it means the same
   thing for every model and every context length.

Nothing here invents data. :func:`summarize_run` reads raw API responses from
``results/raw/`` and refuses to summarise records produced by the mock provider
unless explicitly asked, so a pipeline smoke test can never leak a fake number into
a results table.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

from .scoring import kv_accuracy, score_qa_prediction

__all__ = [
    "DEFAULT_N_RESAMPLES",
    "BootstrapResult",
    "PairedBootstrapResult",
    "McNemarResult",
    "HolmResult",
    "bootstrap_ci",
    "paired_bootstrap_diff",
    "mcnemar_exact",
    "holm_bonferroni",
    "power_paired_binary",
    "mde_paired_binary",
    "mde_two_proportion_unpaired",
    "u_shape_severity",
    "u_shape_severity_ci",
    "accuracy",
    "compare_positions_to_reference",
    "load_raw_records",
    "score_records",
    "summarize_run",
    "write_summary_csv",
]

#: The paper-replication default. Fixed here so every CI in the repo uses the same
#: number of resamples and nobody has to wonder whether a figure used 1,000.
DEFAULT_N_RESAMPLES = 10_000


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with a percentile bootstrap confidence interval.

    Attributes:
        point: The statistic computed on the observed data (not the bootstrap mean).
        low: Lower confidence bound.
        high: Upper confidence bound.
        n: Number of observations resampled.
        confidence: Nominal coverage, e.g. 0.95.
        n_resamples: Bootstrap replicates drawn.
        seed: RNG seed, so the interval is reproducible to the digit.
    """

    point: float
    low: float
    high: float
    n: int
    confidence: float
    n_resamples: int
    seed: int

    @property
    def half_width(self) -> float:
        """Half the CI width -- the number to quote when asked "how precise is this?"."""
        return (self.high - self.low) / 2.0

    def as_dict(self) -> Dict[str, Any]:
        """JSON/CSV-friendly dict."""
        return asdict(self)


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired bootstrap on the difference between two aligned score vectors.

    Attributes:
        diff: ``mean(b) - mean(a)`` on the observed data.
        low: Lower bound of the CI on the difference.
        high: Upper bound of the CI on the difference.
        p_value: Two-sided bootstrap p-value for ``H0: diff == 0``, computed by
            centring the bootstrap distribution on zero and measuring how often a
            replicate is at least as extreme as the observed difference. Includes
            the ``+1`` correction so it can never be exactly 0.
        n: Number of pairs.
        n_resamples: Bootstrap replicates.
        seed: RNG seed.
    """

    diff: float
    low: float
    high: float
    p_value: float
    n: int
    n_resamples: int
    seed: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class McNemarResult:
    """Exact McNemar test on two aligned binary score vectors.

    Attributes:
        n: Number of pairs.
        n01: Pairs where condition A was wrong and condition B was right.
        n10: Pairs where A was right and B was wrong.
        n_discordant: ``n01 + n10`` -- the only pairs the test uses. When this is
            small the test has very little power, no matter how large ``n`` is;
            that is the number to look at before believing a null result.
        p_value: Two-sided exact binomial p-value against ``p = 0.5``.
        proportion_diff: ``mean(B) - mean(A) == (n01 - n10) / n``.
    """

    n: int
    n01: int
    n10: int
    p_value: float
    proportion_diff: float

    @property
    def n_discordant(self) -> int:
        return self.n01 + self.n10

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["n_discordant"] = self.n_discordant
        return out


@dataclass(frozen=True)
class HolmResult:
    """One hypothesis after Holm-Bonferroni correction.

    Attributes:
        label: Caller-supplied name for the comparison.
        p_value: Raw p-value.
        p_adjusted: Holm-adjusted p-value (monotone-enforced, capped at 1.0).
        reject: Whether ``p_adjusted <= alpha``.
        rank: 0-based position in the ascending ordering of raw p-values.
    """

    label: str
    p_value: float
    p_adjusted: float
    reject: bool
    rank: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def _as_float_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap confidence interval for a statistic of one sample.

    Resamples ``values`` with replacement ``n_resamples`` times, recomputes the
    statistic on each replicate, and takes the empirical percentiles. For binary
    per-question scores and ``statistic=mean`` this is a CI on accuracy.

    Why percentile bootstrap rather than a Wilson interval: the same machinery has
    to produce intervals for the U-shape index, which is a ratio of a minimum to a
    mean and has no closed form. Using one method everywhere means the intervals in
    every figure are directly comparable.

    Args:
        values: Observations, typically per-question 0/1 scores.
        statistic: Function applied to each resample. Defaults to the mean.
        n_resamples: Number of bootstrap replicates.
        confidence: Nominal coverage, e.g. 0.95 for a 95% interval.
        seed: RNG seed.

    Returns:
        A :class:`BootstrapResult` whose ``point`` is the statistic on the observed
        data (not the bootstrap mean).

    Raises:
        ValueError: On an empty sample, or ``confidence`` outside (0, 1).
    """
    array = _as_float_array(values, "values")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    rng = np.random.default_rng(seed)
    n = array.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    replicates = np.apply_along_axis(statistic, 1, array[idx]) if statistic is not np.mean else array[idx].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapResult(
        point=float(statistic(array)),
        low=float(low),
        high=float(high),
        n=int(n),
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def paired_bootstrap_diff(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Paired bootstrap on ``mean(b) - mean(a)``, resampling *questions*, not scores.

    Both vectors must be aligned: element ``i`` of each is the same question. Each
    bootstrap replicate draws one set of question indices and applies it to *both*
    vectors, which is what preserves the pairing and gives the test its power.

    The p-value is the standard bootstrap hypothesis test: shift the replicate
    differences so their mean is zero (imposing H0), then ask how often a shifted
    replicate is at least as far from zero as the observed difference. The
    ``(count + 1) / (n_resamples + 1)`` form guarantees a non-zero p-value, so a
    result is never reported as ``p = 0``.

    Args:
        scores_a: Per-question scores in condition A (the reference).
        scores_b: Per-question scores in condition B.
        n_resamples: Bootstrap replicates.
        confidence: Nominal coverage of the interval on the difference.
        seed: RNG seed.

    Returns:
        A :class:`PairedBootstrapResult`.

    Raises:
        ValueError: If the vectors differ in length or are empty.
    """
    a = _as_float_array(scores_a, "scores_a")
    b = _as_float_array(scores_b, "scores_b")
    if a.size != b.size:
        raise ValueError(f"paired vectors must have equal length, got {a.size} and {b.size}")
    rng = np.random.default_rng(seed)
    n = a.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    replicate_diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    observed = float(b.mean() - a.mean())
    alpha = 1.0 - confidence
    low, high = np.quantile(replicate_diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    centred = replicate_diffs - replicate_diffs.mean()
    extreme = int(np.sum(np.abs(centred) >= abs(observed)))
    p_value = (extreme + 1) / (n_resamples + 1)
    return PairedBootstrapResult(
        diff=observed,
        low=float(low),
        high=float(high),
        p_value=float(p_value),
        n=int(n),
        n_resamples=n_resamples,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Paired significance testing
# --------------------------------------------------------------------------- #
def mcnemar_exact(scores_a: Sequence[float], scores_b: Sequence[float]) -> McNemarResult:
    """Exact McNemar test for two paired binary score vectors.

    McNemar's test conditions on the *discordant* pairs -- questions one condition
    got right and the other got wrong -- and asks whether they split evenly. The
    concordant pairs carry no information about the difference, which is precisely
    why the paired design is so much more efficient than an unpaired one.

    The **exact** binomial version is used rather than the chi-square approximation
    because at n=150 with a modest effect there may be only 10-25 discordant pairs,
    where the asymptotic approximation is unreliable. No continuity correction is
    needed or applied.

    Args:
        scores_a: Per-question 0/1 scores in condition A (the reference).
        scores_b: Per-question 0/1 scores in condition B, aligned with A.

    Returns:
        A :class:`McNemarResult`. With zero discordant pairs the p-value is 1.0.

    Raises:
        ValueError: If lengths differ, inputs are empty, or values are not 0/1.
    """
    a = _as_float_array(scores_a, "scores_a")
    b = _as_float_array(scores_b, "scores_b")
    if a.size != b.size:
        raise ValueError(f"paired vectors must have equal length, got {a.size} and {b.size}")
    if not (np.isin(a, (0.0, 1.0)).all() and np.isin(b, (0.0, 1.0)).all()):
        raise ValueError("McNemar's test requires binary 0/1 scores")

    n01 = int(np.sum((a == 0.0) & (b == 1.0)))  # A wrong, B right
    n10 = int(np.sum((a == 1.0) & (b == 0.0)))  # A right, B wrong
    n_discordant = n01 + n10
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = float(scipy_stats.binomtest(n01, n_discordant, 0.5, alternative="two-sided").pvalue)
    return McNemarResult(
        n=int(a.size),
        n01=n01,
        n10=n10,
        p_value=p_value,
        proportion_diff=float(b.mean() - a.mean()),
    )


def holm_bonferroni(
    p_values: Mapping[str, float] | Sequence[Tuple[str, float]],
    *,
    alpha: float = 0.05,
) -> List[HolmResult]:
    """Holm-Bonferroni step-down correction for a family of tests.

    Procedure: sort the ``m`` raw p-values ascending; the ``i``-th (0-based) is
    multiplied by ``m - i``; adjusted values are then made monotone non-decreasing
    by running maximum, and capped at 1.0. Reject while ``p_adjusted <= alpha``.

    Holm is used rather than plain Bonferroni because it is uniformly more powerful
    while controlling the same family-wise error rate, and rather than
    Benjamini-Hochberg because with only 4-6 comparisons per model the FWER is the
    quantity a reader actually cares about.

    Args:
        p_values: Mapping (or sequence of pairs) of comparison label to raw p-value.
        alpha: Family-wise error rate.

    Returns:
        Results in the caller's original order, each carrying its adjusted p-value,
        rejection decision, and rank in the sorted family.

    Raises:
        ValueError: On an empty family or a p-value outside [0, 1].
    """
    items = list(p_values.items()) if isinstance(p_values, Mapping) else list(p_values)
    if not items:
        raise ValueError("p_values must be non-empty")
    for label, p in items:
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"p-value for {label!r} must be in [0, 1], got {p}")

    m = len(items)
    order = sorted(range(m), key=lambda i: float(items[i][1]))
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, float(items[original_index][1]) * (m - rank))
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max

    rank_of = {original_index: rank for rank, original_index in enumerate(order)}
    return [
        HolmResult(
            label=label,
            p_value=float(p),
            p_adjusted=adjusted[i],
            reject=adjusted[i] <= alpha,
            rank=rank_of[i],
        )
        for i, (label, p) in enumerate(items)
    ]


# --------------------------------------------------------------------------- #
# Power / minimum detectable effect
# --------------------------------------------------------------------------- #
def _mcnemar_rejection_threshold(n_discordant: int, alpha: float) -> int:
    """Largest ``k`` such that the exact two-sided test rejects at ``n01 <= k``.

    Returns ``-1`` when no outcome can reject at this ``n_discordant``.
    """
    if n_discordant <= 0:
        return -1
    binom = scipy_stats.binom(n_discordant, 0.5)
    threshold = -1
    for k in range((n_discordant // 2) + 1):
        # For p = 0.5 the exact two-sided p-value is symmetric: 2 * P(X <= k),
        # capped at 1. Rejecting at k implies rejecting at every k' < k.
        if min(1.0, 2.0 * float(binom.cdf(k))) <= alpha:
            threshold = k
        else:
            break
    return threshold


def power_paired_binary(
    n_pairs: int,
    delta: float,
    discordance_rate: float,
    *,
    alpha: float = 0.05,
) -> float:
    """Exact power of the two-sided exact McNemar test.

    Computed by enumeration rather than approximation. Let ``pi_d`` be the
    probability a pair is discordant and ``delta = pi_10 - pi_01`` the true accuracy
    difference. Then the number of discordant pairs ``D ~ Binomial(n, pi_d)`` and,
    conditional on ``D``, the split ``n_10 ~ Binomial(D, pi_10 / pi_d)``. Summing
    the rejection probability over both gives exact power for the exact test.

    Args:
        n_pairs: Number of paired observations (questions per cell).
        delta: True difference in accuracy between the two conditions. Must satisfy
            ``|delta| <= discordance_rate``.
        discordance_rate: Probability that the two conditions disagree on a pair.
            This is the parameter people forget: with a small discordance rate even
            a large ``n`` has little power, because only discordant pairs count.
        alpha: Two-sided significance level.

    Returns:
        Probability of rejecting the null, in [0, 1].

    Raises:
        ValueError: On out-of-range arguments.
    """
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    if not 0.0 < discordance_rate <= 1.0:
        raise ValueError(f"discordance_rate must be in (0, 1], got {discordance_rate}")
    if abs(delta) > discordance_rate + 1e-12:
        raise ValueError(f"|delta| ({abs(delta)}) cannot exceed discordance_rate ({discordance_rate})")

    pi_10 = (discordance_rate + delta) / 2.0
    conditional_p = pi_10 / discordance_rate
    d_dist = scipy_stats.binom(n_pairs, discordance_rate)
    power = 0.0
    for d in range(n_pairs + 1):
        p_d = float(d_dist.pmf(d))
        if p_d < 1e-15:
            continue
        threshold = _mcnemar_rejection_threshold(d, alpha)
        if threshold < 0:
            continue
        split = scipy_stats.binom(d, conditional_p)
        # Reject when the smaller cell is <= threshold, i.e. n_10 <= t or n_10 >= d - t.
        p_reject = float(split.cdf(threshold)) + float(split.sf(d - threshold - 1))
        power += p_d * min(1.0, p_reject)
    return float(min(1.0, power))


def mde_paired_binary(
    n_pairs: int,
    *,
    discordance_rate: float = 0.20,
    alpha: float = 0.05,
    power: float = 0.80,
    tolerance: float = 1e-4,
) -> float:
    """Minimum detectable accuracy difference for the paired (McNemar) design.

    Answers the question every under-powered replication should answer before it
    runs: *given my n, how large does the effect have to be before I could have
    seen it?* Reported in the README next to the subsample size.

    The value is obtained by bisecting :func:`power_paired_binary` on ``delta``, so
    it is the exact-test MDE and not a normal approximation. (Connor 1987's closed
    form for paired-sample size is the usual approximation; inverting the exact test
    avoids its error at the small discordant counts this design produces.)

    Args:
        n_pairs: Questions per cell.
        discordance_rate: Assumed probability that the two conditions disagree on a
            question. **This must be assumed, not derived**, before data exists; the
            configs default to 0.20 and the README states the assumption. After a run
            you can recompute it from the observed discordance -- see
            :func:`mcnemar_exact` -- and report the honest, data-driven MDE.
        alpha: Two-sided significance level.
        power: Target power.
        tolerance: Bisection tolerance on ``delta``.

    Returns:
        The smallest accuracy difference detectable with at least ``power``, or
        ``nan`` if even the largest admissible difference cannot reach it.

    Raises:
        ValueError: On out-of-range arguments.
    """
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    high = discordance_rate
    if power_paired_binary(n_pairs, high, discordance_rate, alpha=alpha) < power:
        return float("nan")
    low = 0.0
    while high - low > tolerance:
        mid = (low + high) / 2.0
        if power_paired_binary(n_pairs, mid, discordance_rate, alpha=alpha) >= power:
            high = mid
        else:
            low = mid
    return float(high)


def mde_two_proportion_unpaired(
    n_per_group: int,
    *,
    baseline: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
    tolerance: float = 1e-5,
) -> float:
    """Minimum detectable effect for an *unpaired* two-proportion comparison.

    Included only as a contrast: run it next to :func:`mde_paired_binary` at the same
    ``n`` and the gap is the concrete value of keeping the question subset fixed
    across cells. Uses the standard normal approximation

        ``n = (z_(1-a/2) * sqrt(2 p_bar q_bar) + z_power * sqrt(p0 q0 + p1 q1))^2 / delta^2``

    inverted by bisection on ``delta``.

    Args:
        n_per_group: Observations per group.
        baseline: Assumed accuracy of the reference group.
        alpha: Two-sided significance level.
        power: Target power.
        tolerance: Bisection tolerance.

    Returns:
        The smallest detectable absolute difference, or ``nan`` if unreachable.
    """
    if n_per_group < 2:
        raise ValueError(f"n_per_group must be >= 2, got {n_per_group}")
    if not 0.0 < baseline < 1.0:
        raise ValueError(f"baseline must be in (0, 1), got {baseline}")
    z_alpha = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(scipy_stats.norm.ppf(power))

    def required_n(delta: float) -> float:
        p1 = min(1.0 - 1e-9, baseline + delta)
        p_bar = (baseline + p1) / 2.0
        term_a = z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        term_b = z_power * math.sqrt(baseline * (1.0 - baseline) + p1 * (1.0 - p1))
        return ((term_a + term_b) ** 2) / (delta**2)

    low, high = tolerance, 1.0 - baseline
    if required_n(high) > n_per_group:
        return float("nan")
    while high - low > tolerance:
        mid = (low + high) / 2.0
        if required_n(mid) <= n_per_group:
            high = mid
        else:
            low = mid
    return float(high)


# --------------------------------------------------------------------------- #
# The U-shape index
# --------------------------------------------------------------------------- #
def u_shape_severity(accuracy_by_position: Sequence[float]) -> float:
    """U-Shape Severity Index (USI): how much worse the middle is than the edges.

    Definition, fixed for the whole project::

        USI = (mean(first, last) - min(middle)) / mean(first, last)

    where ``first``/``last`` are the accuracies at the two extreme gold positions
    and ``middle`` is every position strictly between them.

    Reading it: **0 means flat** (no middle penalty); **0.25 means the worst middle
    position loses a quarter of the edge accuracy**; **negative means the middle is
    actually better than the edges**, which is a real possible outcome and is not
    clipped away. It is a *relative* measure on purpose, so a model at 80% edge
    accuracy and one at 40% can be compared on the same axis -- that is what makes
    the headline "USI vs context length" chart legible across models.

    Known limitations, stated because a reviewer will ask: it uses the minimum, so
    it is sensitive to noise at a single middle position (hence the bootstrap CI in
    :func:`u_shape_severity_ci`, which is mandatory before plotting), and it is
    undefined when edge accuracy is zero.

    Args:
        accuracy_by_position: Accuracies **ordered by gold position**, first to
            last. At least 3 positions are required.

    Returns:
        The index, or ``nan`` when mean edge accuracy is 0.

    Raises:
        ValueError: If fewer than 3 positions are supplied.
    """
    values = [float(v) for v in accuracy_by_position]
    if len(values) < 3:
        raise ValueError(f"USI needs at least 3 positions (first, >=1 middle, last), got {len(values)}")
    edges = (values[0] + values[-1]) / 2.0
    if edges == 0.0:
        return float("nan")
    middle_min = min(values[1:-1])
    return (edges - middle_min) / edges


def u_shape_severity_ci(
    scores_by_position: Sequence[Sequence[float]],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap CI for the USI, resampling questions jointly across positions.

    The joint resample is the whole point: because the same questions are scored at
    every position, a replicate must draw one set of question indices and apply it
    to every position. Resampling positions independently would ignore the pairing
    and produce intervals that are too wide.

    Args:
        scores_by_position: One aligned per-question score vector per gold position,
            ordered first-to-last. All vectors must have the same length.
        n_resamples: Bootstrap replicates.
        confidence: Nominal coverage.
        seed: RNG seed.

    Returns:
        A :class:`BootstrapResult` whose ``point`` is the USI on the observed data.
        Replicates where edge accuracy is 0 yield ``nan`` and are dropped from the
        percentile computation; ``n`` still reports the number of questions.

    Raises:
        ValueError: On fewer than 3 positions or ragged inputs.
    """
    matrix = np.asarray([np.asarray(v, dtype=float) for v in scores_by_position], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        raise ValueError("scores_by_position must be a (positions >= 3, questions) matrix")
    n_positions, n_questions = matrix.shape
    if n_questions == 0:
        raise ValueError("scores_by_position must contain at least one question")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_questions, size=(n_resamples, n_questions))
    # (resamples, positions): mean score per position within each replicate.
    means = matrix[:, idx].mean(axis=2).T
    edges = (means[:, 0] + means[:, -1]) / 2.0
    middle_min = means[:, 1:-1].min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        replicates = np.where(edges == 0.0, np.nan, (edges - middle_min) / edges)

    finite = replicates[np.isfinite(replicates)]
    alpha = 1.0 - confidence
    if finite.size == 0:  # pragma: no cover - degenerate all-zero data
        low = high = float("nan")
    else:
        low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapResult(
        point=u_shape_severity(matrix.mean(axis=1).tolist()),
        low=float(low),
        high=float(high),
        n=int(n_questions),
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Convenience aggregation
# --------------------------------------------------------------------------- #
def accuracy(scores: Sequence[float]) -> float:
    """Mean of a score vector. Trivial, but named so call sites read as English."""
    return float(np.mean(_as_float_array(scores, "scores")))


def compare_positions_to_reference(
    scores_by_position: Mapping[int, Sequence[float]],
    *,
    reference_position: Optional[int] = None,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 0,
) -> Dict[str, Any]:
    """Full paired comparison of every gold position against a reference position.

    Runs both paired procedures at every position, then applies Holm-Bonferroni
    across the family of comparisons (one family per model per context length).

    Args:
        scores_by_position: Gold position -> aligned per-question 0/1 scores. Every
            vector must contain the same questions in the same order.
        reference_position: Position to compare against; defaults to the smallest
            (i.e. gold-at-the-front, the paper's usual reference).
        alpha: Family-wise error rate for the Holm correction.
        n_resamples: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        A dict with ``reference_position``, per-position ``accuracy`` bootstrap
        results, the ``mcnemar`` and ``paired_bootstrap`` results per comparison,
        the ``holm`` correction applied to the McNemar p-values, and the ``usi``.

    Raises:
        ValueError: On ragged inputs or fewer than 2 positions.
    """
    positions = sorted(scores_by_position)
    if len(positions) < 2:
        raise ValueError("need at least two positions to compare")
    lengths = {len(scores_by_position[p]) for p in positions}
    if len(lengths) != 1:
        raise ValueError(f"positions have different numbers of questions: {lengths} -- pairing is broken")

    reference = positions[0] if reference_position is None else reference_position
    if reference not in scores_by_position:
        raise ValueError(f"reference_position {reference} not present in {positions}")

    accuracies = {
        p: bootstrap_ci(scores_by_position[p], n_resamples=n_resamples, seed=seed) for p in positions
    }
    mcnemar: Dict[int, McNemarResult] = {}
    paired: Dict[int, PairedBootstrapResult] = {}
    for p in positions:
        if p == reference:
            continue
        mcnemar[p] = mcnemar_exact(scores_by_position[reference], scores_by_position[p])
        paired[p] = paired_bootstrap_diff(
            scores_by_position[reference], scores_by_position[p], n_resamples=n_resamples, seed=seed
        )

    holm = holm_bonferroni(
        [(f"pos{reference}_vs_pos{p}", mcnemar[p].p_value) for p in mcnemar],
        alpha=alpha,
    )
    usi = (
        u_shape_severity_ci(
            [scores_by_position[p] for p in positions], n_resamples=n_resamples, seed=seed
        )
        if len(positions) >= 3
        else None
    )
    return {
        "reference_position": reference,
        "positions": positions,
        "n_questions": lengths.pop(),
        "accuracy": {p: accuracies[p].as_dict() for p in positions},
        "mcnemar": {p: mcnemar[p].as_dict() for p in mcnemar},
        "paired_bootstrap": {p: paired[p].as_dict() for p in paired},
        "holm": [h.as_dict() for h in holm],
        "usi": usi.as_dict() if usi is not None else None,
    }


# --------------------------------------------------------------------------- #
# Reading raw results
# --------------------------------------------------------------------------- #
class MockRecordsRefused(RuntimeError):
    """Raised when a summary would include responses from the mock provider."""


def load_raw_records(run_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Load every raw API record for a run, keeping the last attempt per example.

    ``results/raw/<run_id>/*.jsonl`` is append-only: a retried example appears more
    than once. The last record for a ``(cell_id, example_id)`` wins, matching the
    runner's own resume logic, so the summary always reflects the final state.

    Args:
        run_dir: ``results/raw/<run_id>``.

    Returns:
        Records with ``error`` unset, deduplicated. Failed calls are dropped here
        and counted separately by :func:`summarize_run`.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    run_dir = pathlib.Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No such run directory: {run_dir}")
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                latest[(record["cell_id"], str(record["example_id"]))] = record
    return [r for r in latest.values() if not r.get("error")]


def score_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Attach a ``score`` to each raw record using the ported metrics.

    Scoring happens here, at analysis time, rather than at request time: the raw
    files hold model text only, so the metric can be changed and everything
    rescored without re-spending a cent.

    Args:
        records: Raw records from :func:`load_raw_records`.

    Returns:
        New dicts with a ``score`` key (1.0/0.0). QA records use
        :func:`litm2026.scoring.score_qa_prediction`; KV records use
        :func:`litm2026.scoring.kv_accuracy`.

    Raises:
        KeyError: If a record lacks the fields its task needs.
    """
    scored: List[Dict[str, Any]] = []
    for record in records:
        row = dict(record)
        if row["experiment"] == "kv_retrieval":
            row["score"] = kv_accuracy(row["response_text"] or "", row["gold_value"])
        else:
            row["score"] = score_qa_prediction(row["response_text"] or "", row["gold_answers"])
        scored.append(row)
    return scored


def summarize_run(
    run_dir: pathlib.Path,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 0,
    allow_mock: bool = False,
) -> List[Dict[str, Any]]:
    """Turn one run's raw responses into per-cell accuracy rows with CIs.

    Args:
        run_dir: ``results/raw/<run_id>``.
        n_resamples: Bootstrap replicates per cell.
        seed: RNG seed for the bootstrap.
        allow_mock: Must be set explicitly to summarise mock-provider records. The
            default refusal is deliberate: it is the guard that stops an offline
            smoke test from ever producing a plausible-looking accuracy table.

    Returns:
        One row per cell, sorted, each with accuracy, CI bounds and counts.

    Raises:
        MockRecordsRefused: If mock records are present and ``allow_mock`` is False.
    """
    records = load_raw_records(run_dir)
    if not allow_mock and any(r.get("is_mock") for r in records):
        raise MockRecordsRefused(
            f"{run_dir} contains responses from the mock provider. These are not model "
            "outputs and must never appear in a results table. Re-run with a real "
            "provider, or pass allow_mock=True if you are deliberately inspecting the "
            "plumbing."
        )
    scored = score_records(records)

    cells: Dict[str, List[Dict[str, Any]]] = {}
    for row in scored:
        cells.setdefault(row["cell_id"], []).append(row)

    summary: List[Dict[str, Any]] = []
    for cell_id, rows in sorted(cells.items()):
        rows.sort(key=lambda r: str(r["example_id"]))
        scores = [float(r["score"]) for r in rows]
        ci = bootstrap_ci(scores, n_resamples=n_resamples, seed=seed)
        head = rows[0]
        summary.append(
            {
                "run_id": head.get("run_id"),
                "cell_id": cell_id,
                "experiment": head["experiment"],
                "model_key": head["model_key"],
                "model": head.get("model"),
                "provider": head.get("provider"),
                "num_documents": head.get("num_documents"),
                "num_keys": head.get("num_keys"),
                "gold_index": head.get("gold_index"),
                "n": len(scores),
                "accuracy": ci.point,
                "ci_low": ci.low,
                "ci_high": ci.high,
                "ci_n_resamples": ci.n_resamples,
                "ci_seed": ci.seed,
                "is_mock": bool(head.get("is_mock", False)),
                "subset_signature": head.get("subset_signature"),
            }
        )
    return summary


SUMMARY_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "cell_id",
    "experiment",
    "model_key",
    "model",
    "provider",
    "num_documents",
    "num_keys",
    "gold_index",
    "n",
    "accuracy",
    "ci_low",
    "ci_high",
    "ci_n_resamples",
    "ci_seed",
    "is_mock",
    "subset_signature",
)


def write_summary_csv(rows: Sequence[Mapping[str, Any]], path: pathlib.Path) -> pathlib.Path:
    """Write summary rows to CSV with a stable column order.

    Args:
        rows: Rows from :func:`summarize_run`.
        path: Destination CSV path (parents are created).

    Returns:
        The path written.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in SUMMARY_COLUMNS})
    return path


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``python -m litm2026.stats --run-dir results/raw/<run_id>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Summarise a run's raw responses into results/summary.csv")
    parser.add_argument("--run-dir", required=True, type=pathlib.Path, help="results/raw/<run_id>")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/summary.csv"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Summarise mock-provider records (plumbing inspection only; never for reporting).",
    )
    args = parser.parse_args(argv)

    rows = summarize_run(
        args.run_dir, n_resamples=args.n_resamples, seed=args.seed, allow_mock=args.allow_mock
    )
    if not rows:
        print(f"No completed records found in {args.run_dir}. Nothing to summarise.")
        return 1
    write_summary_csv(rows, args.out)
    print(f"Wrote {len(rows)} cell(s) to {args.out}")
    for row in rows:
        print(
            f"  {row['cell_id']:<60} n={row['n']:<5} "
            f"acc={row['accuracy']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
