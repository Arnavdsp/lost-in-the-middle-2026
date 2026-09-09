"""Tests for the ported scoring metric.

These matter more than ordinary unit tests. `scoring.py` is a verbatim port of
the original paper's `metrics.py`, and the entire comparability of this
replication rests on it behaving identically. A silent change here would not
crash anything -- it would just quietly produce numbers that cannot be compared
to the paper, which is the worst kind of bug in a replication.
"""

from litm2026.scoring import best_subspan_em, normalize_answer


class TestNormalizeAnswer:
    """The SQuAD normalization: lowercase, strip punctuation, drop articles."""

    def test_lowercases(self):
        assert normalize_answer("Wilhelm Conrad Röntgen") == "wilhelm conrad röntgen"

    def test_removes_articles(self):
        assert normalize_answer("the Nobel Prize") == "nobel prize"
        assert normalize_answer("a house") == "house"
        assert normalize_answer("an apple") == "apple"

    def test_article_removal_is_word_bounded(self):
        # "a" inside a word must survive; only standalone articles go.
        assert "theatre" in normalize_answer("theatre")
        assert normalize_answer("banana") == "banana"

    def test_removes_punctuation(self):
        assert normalize_answer("U.S.A.") == "usa"
        assert normalize_answer("hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_answer("  too    many   spaces ") == "too many spaces"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_only_articles_and_punctuation(self):
        assert normalize_answer("the, a. an!") == ""


class TestBestSubspanEM:
    """Correct if ANY gold answer appears as a substring after normalization."""

    def test_exact_match(self):
        assert best_subspan_em("Paris", ["Paris"]) == 1.0

    def test_substring_of_longer_answer(self):
        # This is the defining property of the metric -- and its main weakness.
        pred = "Based on the search results, the answer is Paris, France."
        assert best_subspan_em(pred, ["Paris"]) == 1.0

    def test_any_of_several_gold_answers(self):
        assert best_subspan_em("Röntgen won it", ["Wilhelm Röntgen", "Röntgen"]) == 1.0

    def test_no_match(self):
        assert best_subspan_em("London", ["Paris"]) == 0.0

    def test_case_and_punctuation_insensitive(self):
        assert best_subspan_em("the answer is PARIS!", ["paris"]) == 1.0

    def test_article_insensitive(self):
        assert best_subspan_em("It is the United States", ["United States"]) == 1.0

    def test_empty_prediction_scores_zero(self):
        assert best_subspan_em("", ["Paris"]) == 0.0

    def test_returns_float_not_bool(self):
        # Downstream code averages these; bools would work but floats are the
        # original's contract and np operations depend on it.
        assert isinstance(best_subspan_em("Paris", ["Paris"]), float)


class TestKnownWeaknesses:
    """Documenting the metric's flaws as tests.

    These are not bugs. They are properties of the original metric that this
    replication keeps ON PURPOSE, because changing the metric would break
    comparability with the paper. Be ready to discuss these in an interview:
    knowing your metric's failure modes is the point.
    """

    def test_false_positive_on_hedged_waffle(self):
        # A model that lists candidates without committing still scores 1.0.
        pred = "It could be London, Paris, or Berlin -- I am not certain."
        assert best_subspan_em(pred, ["Paris"]) == 1.0

    def test_false_negative_on_correct_paraphrase(self):
        # Semantically right, lexically different -> scored wrong.
        assert best_subspan_em("The City of Light", ["Paris"]) == 0.0

    def test_false_positive_on_negation(self):
        # The metric has no notion of negation.
        assert best_subspan_em("The answer is not Paris", ["Paris"]) == 1.0
