"""Scoring metrics, ported verbatim from the original *Lost in the Middle* release.

PROVENANCE
----------
``normalize_answer`` and ``best_subspan_em`` below are a **verbatim port** of:

    file:    src/lost_in_the_middle/metrics.py
    repo:    https://github.com/nelson-liu/lost-in-the-middle
    commit:  29b8a6d042ce29abccee3db1a73171a107d7e6af  ("Fix typo in README", 2024-01-03)
    license: MIT (Copyright (c) 2023 Nelson Liu) -- see THIRD_PARTY_LICENSES.md

That file in turn documents its own provenance: the normalization originates from
the **SQuAD evaluation script**, specifically

    https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/

The two functions are reproduced character-for-character. **Do not "improve" them.**
The whole point of a replication is that the metric is identical to the original;
a "better" normalizer produces numbers that cannot be compared to the paper.

Deliberately preserved quirks, so nobody is tempted to "fix" them later:

* The composed order is ``white_space_fix(remove_articles(remove_punc(lower(s))))``
  -- punctuation is stripped *before* articles are removed, so e.g. ``"a-the"``
  becomes ``"athe"`` and then no longer matches the article pattern.
* ``best_subspan_em`` lower-cases again inside the loop even though
  ``normalize_answer`` already lower-cased. Harmless, and kept.
* The metric is a **substring** test, not equality: a prediction of
  "The first Nobel Prize in Physics went to Wilhelm Conrad Rontgen in 1901"
  scores 1.0 against the gold answer "Wilhelm Conrad Rontgen". This is generous
  to verbose chat models and is a real threat to validity (see REPLICATION.md).
* The third-party ``regex`` module is imported rather than stdlib ``re``, matching
  the original import. For this pattern the two behave identically, but we keep the
  original dependency rather than assume that.

The two helpers *below* the port are not in ``metrics.py``. They reproduce
pre-processing that the original applies in its evaluation scripts, at the same
commit:

* :func:`first_line` / :func:`score_qa_prediction` reproduce
  ``scripts/evaluate_qa_responses.py``, which truncates the model answer at the
  first newline before scoring.
* :func:`kv_accuracy` reproduces ``scripts/evaluate_kv_responses.py``, whose metric
  is a plain case-insensitive substring check on the *raw* model answer -- it does
  **not** use ``normalize_answer`` (the values are UUIDs, whose hyphens
  ``remove_punc`` would delete) and does **not** truncate at the first newline.
  That asymmetry with the QA metric exists in the original and is preserved.
"""

from __future__ import annotations

import string
from typing import List, Sequence

import regex

__all__ = [
    "normalize_answer",
    "best_subspan_em",
    "first_line",
    "score_qa_prediction",
    "kv_accuracy",
]


# --------------------------------------------------------------------------- #
# VERBATIM PORT BEGINS -- src/lost_in_the_middle/metrics.py @ 29b8a6d
# --------------------------------------------------------------------------- #
def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


# --------------------------------------------------------------------------- #
# VERBATIM PORT ENDS
# --------------------------------------------------------------------------- #


def first_line(model_answer: str) -> str:
    """Return everything up to the first newline of ``model_answer``, stripped.

    Reproduces the pre-processing in the original repository's
    ``scripts/evaluate_qa_responses.py``. The original's comment, quoted verbatim
    (typo included)::

        # NOTE: we take everything up to the first newline, since otherwise models
        # could hack the metric by simply copying te input context (as the gold
        # answer is guaranteed to occur in the input context).

    This matters more in 2026 than it did in 2023: chat models are far more verbose
    than the completion models the paper evaluated, and a model that restates the
    provided documents before answering would score 1.0 on nearly every open-book
    example without this truncation. Keeping it is a fidelity requirement.

    Args:
        model_answer: Raw model output text.

    Returns:
        The first line of the output, with surrounding whitespace stripped.
    """
    return model_answer.split("\n")[0].strip()


def score_qa_prediction(
    prediction: str,
    answers: Sequence[str],
    *,
    truncate_at_first_newline: bool = True,
) -> float:
    """Score one multi-document QA prediction with best-subspan exact match.

    This is exactly ``evaluate_qa_responses.get_metrics_for_example`` composed with
    :func:`best_subspan_em`: truncate at the first newline, strip, then subspan-match.

    Args:
        prediction: Raw model output text (not pre-truncated).
        answers: Gold answer aliases for the question (NQ-open supplies several).
        truncate_at_first_newline: Keep ``True`` to match the original protocol.
            ``False`` exists only so the sensitivity of the metric to verbosity can
            be measured and reported; it is **not** the replication setting.

    Returns:
        1.0 if any gold alias is a normalized subspan of the (truncated) prediction,
        else 0.0.
    """
    text = first_line(prediction) if truncate_at_first_newline else prediction
    return best_subspan_em(prediction=text, ground_truths=list(answers))


def kv_accuracy(prediction: str, value: str) -> float:
    """Key-value retrieval accuracy, ported from ``scripts/evaluate_kv_responses.py``.

    A prediction is correct iff the gold value appears as a case-insensitive
    substring of the raw model output. No SQuAD normalization and no first-line
    truncation are applied, matching the original::

        accuracy = 1.0 if example["value"].lower() in model_answer.lower() else 0.0

    Args:
        prediction: Raw model output text.
        value: The gold value string (a UUID) for the queried key.

    Returns:
        1.0 if correct, else 0.0.
    """
    return 1.0 if value.lower() in prediction.lower() else 0.0
