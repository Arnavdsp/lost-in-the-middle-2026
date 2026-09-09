"""Tests for data loading and the paired question subset.

The subset logic is load-bearing. Every positional cell must run on the SAME
questions, or the paired statistics in `stats.py` are invalid and the whole
analysis quietly becomes weaker and wrong. These tests exist to make that
guarantee explicit.
"""

import pytest

from litm2026.data import (
    evenly_spaced_positions,
    qa_oracle_path,
    qa_positional_path,
    question_id,
    select_index_subset,
    select_question_subset,
)


class TestQuestionId:
    def test_is_stable_across_calls(self):
        assert question_id("who won the first nobel prize") == question_id(
            "who won the first nobel prize"
        )

    def test_differs_for_different_questions(self):
        assert question_id("question a") != question_id("question b")

    def test_is_not_the_raw_question(self):
        # An id, not the text -- filenames and cell ids depend on it being short.
        qid = question_id("a fairly long natural language question about physics")
        assert len(qid) < 65


class TestSelectQuestionSubset:
    """The paired design lives or dies here."""

    def test_same_seed_gives_the_same_subset(self):
        universe = [f"q{i}" for i in range(1000)]
        a = select_question_subset(universe, n=150, seed=0)
        b = select_question_subset(universe, n=150, seed=0)
        assert a.question_ids == b.question_ids

    def test_different_seeds_give_different_subsets(self):
        universe = [f"q{i}" for i in range(1000)]
        a = select_question_subset(universe, n=150, seed=0)
        b = select_question_subset(universe, n=150, seed=1)
        assert a.question_ids != b.question_ids

    def test_returns_exactly_n(self):
        universe = [f"q{i}" for i in range(1000)]
        assert len(select_question_subset(universe, n=150, seed=0).question_ids) == 150

    def test_has_no_duplicates(self):
        universe = [f"q{i}" for i in range(1000)]
        ids = select_question_subset(universe, n=150, seed=0).question_ids
        assert len(set(ids)) == len(ids)

    def test_every_id_comes_from_the_universe(self):
        universe = [f"q{i}" for i in range(1000)]
        subset = select_question_subset(universe, n=50, seed=3)
        assert set(subset.question_ids) <= set(universe)

    def test_n_larger_than_universe_is_rejected(self):
        with pytest.raises(ValueError):
            select_question_subset([f"q{i}" for i in range(10)], n=50, seed=0)

    def test_subset_is_order_independent(self):
        # Files may enumerate questions in a different order across cells; the
        # chosen subset must not depend on that.
        universe = [f"q{i}" for i in range(500)]
        forward = select_question_subset(universe, n=100, seed=0)
        backward = select_question_subset(list(reversed(universe)), n=100, seed=0)
        assert set(forward.question_ids) == set(backward.question_ids)


class TestSelectIndexSubset:
    def test_is_deterministic(self):
        assert select_index_subset(1000, 100, 0) == select_index_subset(1000, 100, 0)

    def test_indices_are_in_range(self):
        indices = select_index_subset(1000, 100, 0)
        assert all(0 <= i < 1000 for i in indices)

    def test_no_duplicates(self):
        indices = select_index_subset(1000, 100, 0)
        assert len(set(indices)) == len(indices)


class TestEvenlySpacedPositions:
    def test_includes_both_ends(self):
        positions = evenly_spaced_positions(20, 5)
        assert positions[0] == 0
        assert positions[-1] == 19

    def test_matches_the_papers_20_document_positions(self):
        # The paper's released 20-document files are gold_at 0, 4, 9, 14, 19.
        assert evenly_spaced_positions(20, 5) == (0, 4, 9, 14, 19)

    def test_is_sorted_and_unique(self):
        positions = evenly_spaced_positions(30, 7)
        assert list(positions) == sorted(set(positions))


class TestPathHelpers:
    def test_positional_path_encodes_count_and_gold_index(self, tmp_path):
        path = qa_positional_path(tmp_path, num_documents=20, gold_index=9)
        assert "20_total_documents" in str(path)
        assert "gold_at_9" in str(path)
        assert str(path).endswith(".jsonl.gz")

    def test_oracle_path_is_the_single_document_file(self, tmp_path):
        assert "oracle" in str(qa_oracle_path(tmp_path))

    def test_invalid_document_count_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            qa_positional_path(tmp_path, num_documents=17, gold_index=0)

    def test_gold_index_out_of_range_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            qa_positional_path(tmp_path, num_documents=20, gold_index=25)
