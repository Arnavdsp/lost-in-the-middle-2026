"""Prompt construction, ported from the original *Lost in the Middle* release.

PROVENANCE
----------
``Document``, ``get_qa_prompt``, ``get_closedbook_qa_prompt`` and
``get_kv_retrieval_prompt`` are ported from:

    file:    src/lost_in_the_middle/prompting.py
    repo:    https://github.com/nelson-liu/lost-in-the-middle
    commit:  29b8a6d042ce29abccee3db1a73171a107d7e6af  ("Fix typo in README", 2024-01-03)
    license: MIT (Copyright (c) 2023 Nelson Liu) -- see THIRD_PARTY_LICENSES.md

The ``.prompt`` templates in ``litm2026/prompts/`` are **byte-identical copies** of
``src/lost_in_the_middle/prompts/*.prompt`` from that commit. Their SHA-256 digests
are recorded in :data:`VENDORED_PROMPT_SHA256` and verified by the test suite, so a
silent edit to a template fails CI rather than quietly changing the experiment.

The document format string is the single most load-bearing line in this file::

    f"Document [{document_index+1}](Title: {document.title}) {document.text}"

It is reproduced exactly. Changing spacing, bracket style or 1-indexing changes the
tokenization of every prompt and therefore the numbers.

TWO DELIBERATE DEVIATIONS FROM THE ORIGINAL
-------------------------------------------
1. ``Document`` is a stdlib ``dataclasses.dataclass`` here; the original uses
   ``pydantic.dataclasses.dataclass``. Field names, order, defaults and
   ``from_dict`` semantics (including ``float(score)`` coercion) are preserved, so
   the emitted prompt text is identical. We drop the pydantic dependency because
   nothing in this replication relies on pydantic's runtime type coercion.

2. **Chat adaptation.** The original prompts are *completion-style*: they end in
   the literal token sequence ``"Answer:"`` and the model continues the text. Every
   model in this 2026 replication is served through a chat-completions API, so the
   prompt has to be wrapped in a message list. That wrapping is a real deviation
   from the paper's protocol, not a formatting detail, and it is one of the leading
   candidate explanations for any divergence in results. :func:`to_chat_messages`
   implements it, :class:`ChatAdaptation` enumerates the variants, and the variants
   exist specifically so the effect of the adaptation can be A/B'd and reported
   (see ``REPLICATION.md``, section "What I changed and why").
"""

from __future__ import annotations

import enum
import hashlib
import pathlib
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

PROMPTS_ROOT = (pathlib.Path(__file__).parent / "prompts").resolve()

#: SHA-256 of each vendored template as copied from upstream commit 29b8a6d.
#: Verified by ``tests/test_prompting.py::test_vendored_prompt_files_are_unmodified``.
VENDORED_PROMPT_SHA256: Dict[str, str] = {
    "closedbook_qa.prompt": "5b75e2b95b1562cdd9a1e8f69a32073eb8966cfa82634bd70b1259daf717193b",
    "kv_retrieval.prompt": "62a513f9d0e039c3a3783d1f946e18ffdb8c549d7ccd6f4f4c0b01bddee3e450",
    "kv_retrieval_with_query_aware_contextualization.prompt": (
        "0f6e4c5db0ecf66f2f5bbe0e82335889bf5e72d8371929c985e15f50da5c76ad"
    ),
    "qa.prompt": "368a84fe24373cc0a3a789ce5e62e6854994ece138747c93c76f25aafea25411",
    "qa_ordered_randomly.prompt": "e3fb49897502903ff46db87f770d253f93e0d11c5bd4e1b48451c5cf4d4c4f67",
    "qa_with_query_aware_contextualization.prompt": (
        "ff1168b14640bbb00ade41cf1d4d8d3a43835cb0e5ba95dd2c4d00b6684fb22e"
    ),
}

T = TypeVar("T")

__all__ = [
    "PROMPTS_ROOT",
    "VENDORED_PROMPT_SHA256",
    "Document",
    "get_qa_prompt",
    "get_closedbook_qa_prompt",
    "get_kv_retrieval_prompt",
    "ChatAdaptation",
    "to_chat_messages",
    "documents_from_example",
    "randomize_distractor_ordering",
    "place_kv_gold_at",
    "prompt_file_digests",
]


# --------------------------------------------------------------------------- #
# PORT BEGINS -- src/lost_in_the_middle/prompting.py @ 29b8a6d
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Document:
    """One retrieved passage, as stored in the paper's ``ctxs`` lists.

    Field order and defaults match the original ``pydantic`` dataclass exactly.
    """

    title: str
    text: str
    id: Optional[str] = None
    score: Optional[float] = None
    hasanswer: Optional[bool] = None
    isgold: Optional[bool] = None
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        return cls(**dict(data, id=id, score=score))


def get_qa_prompt(
    question: str,
    documents: List[Document],
    mention_random_ordering: bool,
    query_aware_contextualization: bool,
) -> str:
    """Build the open-book multi-document QA prompt (ported verbatim).

    Args:
        question: The NQ-open question string.
        documents: Retrieved passages **in the exact order they should appear**.
            The gold document's position in this list *is* the independent variable
            of the whole experiment; do not sort or shuffle here.
        mention_random_ordering: Use the ``qa_ordered_randomly`` template.
        query_aware_contextualization: Use the template that repeats the question
            before *and* after the documents.

    Returns:
        The completion-style prompt string, ending in ``"Answer:"``.

    Raises:
        ValueError: On empty question/documents, or if both boolean flags are set.
    """
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not documents:
        raise ValueError(f"Provided `documents` must be truthy, got: {documents}")

    if mention_random_ordering and query_aware_contextualization:
        raise ValueError("Mentioning random ordering cannot be currently used with query aware contextualization")

    if mention_random_ordering:
        prompt_filename = "qa_ordered_randomly.prompt"
    elif query_aware_contextualization:
        prompt_filename = "qa_with_query_aware_contextualization.prompt"
    else:
        prompt_filename = "qa.prompt"

    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")

    # Format the documents into strings
    formatted_documents = []
    for document_index, document in enumerate(documents):
        formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")
    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))


def get_closedbook_qa_prompt(question: str) -> str:
    """Build the closed-book prompt (no documents at all). Ported verbatim.

    This is the **floor** baseline: what the model knows from parameters alone.

    Args:
        question: The NQ-open question string.

    Returns:
        ``"Question: {question}\\nAnswer:"``.
    """
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    with open(PROMPTS_ROOT / "closedbook_qa.prompt") as f:
        prompt_template = f.read().rstrip("\n")

    return prompt_template.format(question=question)


def get_kv_retrieval_prompt(
    data: List[Tuple[str, str]],
    key: str,
    query_aware_contextualization: bool = False,
) -> str:
    """Build the synthetic key-value retrieval prompt (ported verbatim).

    Args:
        data: Ordered ``(key, value)`` pairs. Order is the independent variable.
        key: The key whose value the model must return. Must be present in ``data``.
        query_aware_contextualization: Repeat the key before the JSON blob as well.

    Returns:
        The completion-style prompt string, ending in ``"Corresponding value:"``.

    Raises:
        ValueError: On empty inputs, a missing key, duplicate keys, or fewer than
            two records.
    """
    if not data:
        raise ValueError(f"Provided `data` must be truthy, got: {data}")
    if not key:
        raise ValueError(f"Provided `key` must be truthy, got: {key}")
    if key not in [x[0] for x in data]:
        raise ValueError(f"Did not find provided `key` {key} in data {data}")
    if len(data) != len(set([x[0] for x in data])):
        raise ValueError(f"`data` has duplicate keys: {data}")
    if len(data) < 2:
        raise ValueError(f"Must have at least 2 items in data: {data}")

    if query_aware_contextualization:
        with open(PROMPTS_ROOT / "kv_retrieval_with_query_aware_contextualization.prompt") as f:
            prompt_template = f.read().rstrip("\n")
    else:
        with open(PROMPTS_ROOT / "kv_retrieval.prompt") as f:
            prompt_template = f.read().rstrip("\n")

    # Format the KV data into a string
    formatted_kv_records = ""
    for index, record in enumerate(data):
        start_character = "{" if index == 0 else " "
        data_string = f'"{record[0]}": "{record[1]}"'
        end_character = ",\n" if index != len(data) - 1 else "}"
        formatted_kv_records += start_character + data_string + end_character

    return prompt_template.format(formatted_kv_records=formatted_kv_records, key=key)


# --------------------------------------------------------------------------- #
# PORT ENDS
# --------------------------------------------------------------------------- #


def documents_from_example(example: Dict[str, Any]) -> List[Document]:
    """Convert an example's ``ctxs`` list into :class:`Document` objects, in order.

    Args:
        example: A record loaded from the paper's gzipped JSONL QA data.

    Returns:
        Documents in the file's original order (i.e. with the gold document already
        at the position encoded in the filename).

    Raises:
        ValueError: If the example has no contexts.
    """
    documents = [Document.from_dict(ctx) for ctx in deepcopy(example["ctxs"])]
    if not documents:
        raise ValueError(f"Did not find any documents for example: {example.get('question')!r}")
    return documents


def randomize_distractor_ordering(documents: Sequence[Document], rng: random.Random) -> List[Document]:
    """Shuffle only the distractors, keeping the gold document at its index.

    Ported from ``scripts/get_qa_responses_from_llama_2.py`` (``--use-random-ordering``).
    Not used by the default configs; kept because it is the natural control for
    "is the effect about position, or about retrieval-score ordering?".

    Args:
        documents: Documents in file order; exactly one must have ``isgold=True``.
        rng: Seeded RNG, so the shuffle is reproducible.

    Returns:
        A new list with the gold document at its original index and the distractors
        permuted around it.
    """
    (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
    original_gold_document = documents[original_gold_index]
    distractors = [doc for doc in documents if doc.isgold is False]
    rng.shuffle(distractors)
    distractors.insert(original_gold_index, original_gold_document)
    return list(distractors)


def place_kv_gold_at(
    ordered_kv_records: Sequence[Sequence[str]],
    key: str,
    value: str,
    gold_index: int,
) -> List[Tuple[str, str]]:
    """Move the queried key-value pair to ``gold_index``, preserving all other order.

    Ported from ``scripts/get_kv_responses_from_longchat.py``: pop the gold record
    from wherever it is and re-insert it at the requested index. This is how the
    paper sweeps position in the KV task -- the *same* 500 examples are reused at
    every position, which is what makes the KV comparison naturally paired.

    Args:
        ordered_kv_records: The example's records, each a 2-sequence ``[key, value]``.
        key: Gold key.
        value: Gold value.
        gold_index: Target index for the gold pair, ``0 <= gold_index < len(records)``.

    Returns:
        A new list of ``(key, value)`` tuples with the gold pair at ``gold_index``.

    Raises:
        ValueError: If ``gold_index`` is out of range or the pair is absent.
    """
    records = [(str(k), str(v)) for k, v in ordered_kv_records]
    if not 0 <= gold_index < len(records):
        raise ValueError(f"gold_index {gold_index} out of range for {len(records)} records")
    try:
        original_kv_index = records.index((key, value))
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Gold pair ({key!r}, {value!r}) not present in records") from exc
    original_kv = records.pop(original_kv_index)
    records.insert(gold_index, original_kv)
    return records


class ChatAdaptation(str, enum.Enum):
    """How a completion-style prompt is wrapped for a chat-completions API.

    Every variant is a *deviation* from the paper, which used raw completion. The
    variants exist so the size of that deviation can be measured rather than assumed.

    Members:
        USER_VERBATIM: One user message whose content is the original prompt,
            unmodified, trailing ``"Answer:"`` included. This is the default and the
            closest thing to the original protocol that a chat API allows. It is
            what all shipped configs use.
        USER_NO_ANSWER_CUE: Same, but the trailing ``"Answer:"`` / ``"Corresponding
            value:"`` cue line is removed. Tests whether the dangling cue confuses
            instruction-tuned models.
        SYSTEM_INSTRUCTION: The template's leading instruction line is moved into a
            system message and the rest stays in the user message. Tests whether
            role placement of the instruction matters.
        USER_VERBATIM_TERSE: ``USER_VERBATIM`` plus a short system message asking for
            the answer only, with no preamble. Tests whether chat verbosity (which
            interacts with the first-newline truncation in the metric) is depressing
            open-book scores.
    """

    USER_VERBATIM = "user_verbatim"
    USER_NO_ANSWER_CUE = "user_no_answer_cue"
    SYSTEM_INSTRUCTION = "system_instruction"
    USER_VERBATIM_TERSE = "user_verbatim_terse"


#: System message used by :attr:`ChatAdaptation.USER_VERBATIM_TERSE`. Kept as a
#: module constant so it is logged with every run and cannot drift silently.
TERSE_SYSTEM_MESSAGE = "Answer with the answer only. Do not explain. Do not restate the question or the documents."

#: Cue lines that :attr:`ChatAdaptation.USER_NO_ANSWER_CUE` will strip if trailing.
_ANSWER_CUES = ("Answer:", "Corresponding value:")


def to_chat_messages(
    prompt: str,
    adaptation: ChatAdaptation = ChatAdaptation.USER_VERBATIM,
) -> List[Dict[str, str]]:
    """Adapt a completion-style prompt into chat-completions messages.

    THIS IS A DOCUMENTED DEVIATION FROM THE ORIGINAL PAPER. The paper fed raw text
    to base/completion models and read the continuation. Every 2026 API model in
    this replication is chat-only, so the prompt must be placed in a message list,
    which inserts provider-specific role tokens and (for hosted models) an unknown
    server-side system preamble that we cannot inspect or control.

    Args:
        prompt: A prompt produced by :func:`get_qa_prompt`,
            :func:`get_closedbook_qa_prompt` or :func:`get_kv_retrieval_prompt`.
        adaptation: Which wrapping variant to use. See :class:`ChatAdaptation`.

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts, ordered
        system-then-user. Suitable for both the OpenAI-compatible and Anthropic
        request builders in :mod:`litm2026.providers`.

    Raises:
        ValueError: If ``prompt`` is empty or ``adaptation`` is unknown.
    """
    if not prompt:
        raise ValueError("`prompt` must be a non-empty string")
    adaptation = ChatAdaptation(adaptation)

    if adaptation is ChatAdaptation.USER_VERBATIM:
        return [{"role": "user", "content": prompt}]

    if adaptation is ChatAdaptation.USER_VERBATIM_TERSE:
        return [
            {"role": "system", "content": TERSE_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]

    if adaptation is ChatAdaptation.USER_NO_ANSWER_CUE:
        stripped = prompt
        for cue in _ANSWER_CUES:
            if stripped.endswith("\n" + cue):
                stripped = stripped[: -(len(cue) + 1)]
                break
            if stripped == cue:  # pragma: no cover - degenerate
                stripped = ""
                break
        return [{"role": "user", "content": stripped.rstrip("\n")}]

    if adaptation is ChatAdaptation.SYSTEM_INSTRUCTION:
        lines = prompt.split("\n")
        instruction = lines[0]
        remainder = "\n".join(lines[1:]).lstrip("\n")
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": remainder},
        ]

    raise ValueError(f"Unhandled adaptation: {adaptation!r}")  # pragma: no cover


def prompt_file_digests() -> Dict[str, str]:
    """Return the SHA-256 digest of every vendored ``.prompt`` file on disk.

    Used by the test suite to prove the vendored templates still match upstream, and
    logged into each run's manifest so a result can always be traced to the exact
    template bytes that produced it.

    Returns:
        Mapping of filename to lowercase hex digest.
    """
    digests: Dict[str, str] = {}
    for path in sorted(PROMPTS_ROOT.glob("*.prompt")):
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests
