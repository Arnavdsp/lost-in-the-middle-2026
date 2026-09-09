"""Tests for prompt construction.

The prompt is the other half of what makes this a replication rather than a
different experiment. If the document format string or the template drifts, the
numbers stop being comparable to the paper -- and nothing would visibly break.
"""

import pytest

from litm2026.prompting import (
    ChatAdaptation,
    Document,
    documents_from_example,
    get_closedbook_qa_prompt,
    get_kv_retrieval_prompt,
    get_qa_prompt,
    to_chat_messages,
)


def _docs(n: int) -> list[Document]:
    return [Document(title=f"Title {i}", text=f"Text body {i}.") for i in range(n)]


class TestQAPrompt:
    def test_document_format_matches_original_exactly(self):
        # The original: f"Document [{i+1}](Title: {title}) {text}"
        # 1-INDEXED. Getting this wrong is silent and changes every number.
        prompt = get_qa_prompt("q?", _docs(2), False, False)
        assert "Document [1](Title: Title 0) Text body 0." in prompt
        assert "Document [2](Title: Title 1) Text body 1." in prompt
        assert "Document [0]" not in prompt

    def test_question_is_included(self):
        prompt = get_qa_prompt("who won the first nobel prize?", _docs(3), False, False)
        assert "who won the first nobel prize?" in prompt

    def test_documents_appear_in_given_order(self):
        prompt = get_qa_prompt("q?", _docs(5), False, False)
        positions = [prompt.index(f"Text body {i}.") for i in range(5)]
        assert positions == sorted(positions)

    def test_ends_with_answer_cue(self):
        # The paper's template ends "Answer:" -- it is what makes a completion
        # model produce the answer rather than more documents.
        assert get_qa_prompt("q?", _docs(2), False, False).rstrip().endswith("Answer:")

    def test_empty_question_rejected(self):
        with pytest.raises(ValueError):
            get_qa_prompt("", _docs(2), False, False)

    def test_empty_documents_rejected(self):
        with pytest.raises(ValueError):
            get_qa_prompt("q?", [], False, False)

    def test_mutually_exclusive_variants_rejected(self):
        with pytest.raises(ValueError):
            get_qa_prompt("q?", _docs(2), True, True)


class TestClosedbookPrompt:
    def test_contains_question_but_no_documents(self):
        prompt = get_closedbook_qa_prompt("who won?")
        assert "who won?" in prompt
        assert "Document [" not in prompt

    def test_empty_question_rejected(self):
        with pytest.raises(ValueError):
            get_closedbook_qa_prompt("")


class TestKVPrompt:
    def test_renders_json_like_block_and_key(self):
        pairs = [("k1", "v1"), ("k2", "v2"), ("k3", "v3")]
        prompt = get_kv_retrieval_prompt(pairs, "k2")
        assert '"k1": "v1"' in prompt
        assert '"k2": "v2"' in prompt
        assert "k2" in prompt

    def test_rejects_key_not_present(self):
        with pytest.raises(ValueError):
            get_kv_retrieval_prompt([("a", "1"), ("b", "2")], "missing")

    def test_rejects_duplicate_keys(self):
        with pytest.raises(ValueError):
            get_kv_retrieval_prompt([("a", "1"), ("a", "2")], "a")

    def test_rejects_fewer_than_two_pairs(self):
        with pytest.raises(ValueError):
            get_kv_retrieval_prompt([("a", "1")], "a")


class TestDocumentsFromExample:
    def test_reads_the_papers_record_shape(self):
        example = {
            "question": "q?",
            "answers": ["a"],
            "ctxs": [
                {"title": "T0", "text": "x", "hasanswer": False, "isgold": False},
                {"title": "T1", "text": "y", "hasanswer": True, "isgold": True},
            ],
        }
        docs = documents_from_example(example)
        assert len(docs) == 2
        assert docs[1].isgold is True
        assert docs[0].title == "T0"


class TestChatAdaptation:
    """The chat adaptation is a DOCUMENTED DEVIATION from the original.

    The paper's models were completion models; the free-tier models used here
    are chat models. That difference is unavoidable, so it is made explicit and
    switchable rather than hidden.
    """

    def test_user_verbatim_passes_prompt_through_unchanged(self):
        prompt = get_qa_prompt("q?", _docs(2), False, False)
        messages = to_chat_messages(prompt, ChatAdaptation.USER_VERBATIM)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == prompt

    def test_every_adaptation_produces_valid_messages(self):
        prompt = get_qa_prompt("q?", _docs(2), False, False)
        for adaptation in ChatAdaptation:
            messages = to_chat_messages(prompt, adaptation)
            assert messages, f"{adaptation} produced no messages"
            for message in messages:
                assert message["role"] in {"system", "user", "assistant"}
                assert isinstance(message["content"], str)

    def test_adaptations_are_actually_different(self):
        # If they all rendered identically the A/B flag would be meaningless.
        prompt = get_qa_prompt("q?", _docs(2), False, False)
        rendered = {
            adaptation: repr(to_chat_messages(prompt, adaptation)) for adaptation in ChatAdaptation
        }
        assert len(set(rendered.values())) > 1
