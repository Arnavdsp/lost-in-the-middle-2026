"""Loaders for the original *Lost in the Middle* data, plus the paired subsampling.

The data itself is **not** vendored into this repository (it is ~257 MB gzipped).
Fetch it with ``python scripts/download_data.py``; ``data/original/`` is gitignored.

WHY THE SUBSAMPLING CODE IN HERE IS THE MOST IMPORTANT PART OF THIS MODULE
--------------------------------------------------------------------------
The paper's design is *within-question*: every gold position is evaluated on the
same 2,655 NQ-open questions, so position effects are not confounded with question
difficulty. A budget-constrained replication that draws a fresh random n=150 per
cell throws that away and needs a much larger n to see the same effect.

So: :func:`select_question_subset` derives one subset from the full question
universe with a fixed seed, and every cell in the run uses **that same subset**.
The subset carries a :meth:`QuestionSubset.signature` that is written into every
run manifest, so "were these cells actually paired?" is an auditable question and
not a matter of trust. ``tests/test_data.py`` asserts the property directly.

DATA INTEGRITY
--------------
:data:`DATA_MANIFEST` records the SHA-256 and byte size of every upstream file, as
measured from a clean clone of commit 29b8a6d042ce29abccee3db1a73171a107d7e6af.
``scripts/download_data.py`` verifies downloads against it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_DATA_ROOT",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPO",
    "N_QA_EXAMPLES",
    "N_KV_EXAMPLES",
    "QA_GOLD_INDICES",
    "KV_NUM_KEYS",
    "DataFileSpec",
    "DATA_MANIFEST",
    "qa_positional_path",
    "qa_oracle_path",
    "kv_path",
    "iter_jsonl_gz",
    "load_jsonl_gz",
    "question_id",
    "canonical_question_ids",
    "QuestionSubset",
    "select_question_subset",
    "apply_question_subset",
    "select_index_subset",
    "evenly_spaced_positions",
    "missing_data_files",
    "require_data",
    "sha256_file",
]

DEFAULT_DATA_ROOT = pathlib.Path("data/original")

UPSTREAM_REPO = "https://github.com/nelson-liu/lost-in-the-middle"
UPSTREAM_COMMIT = "29b8a6d042ce29abccee3db1a73171a107d7e6af"

#: Number of NQ-open questions in every multi-document QA file (verified).
N_QA_EXAMPLES = 2655
#: Number of examples in every key-value retrieval file (verified).
N_KV_EXAMPLES = 500

#: Gold positions the upstream repo ships for each total-document count.
QA_GOLD_INDICES: Mapping[int, Tuple[int, ...]] = {
    10: (0, 4, 9),
    20: (0, 4, 9, 14, 19),
    30: (0, 4, 9, 14, 19, 24, 29),
}

#: Key counts the upstream repo ships for the KV retrieval task.
KV_NUM_KEYS: Tuple[int, ...] = (75, 140, 300)


@dataclass(frozen=True)
class DataFileSpec:
    """One upstream data file: its repo-relative path, size and digest."""

    relpath: str
    size_bytes: int
    sha256: str

    @property
    def raw_url(self) -> str:
        """URL for the file on ``raw.githubusercontent.com`` at the pinned commit."""
        return (
            "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/"
            f"{UPSTREAM_COMMIT}/{self.relpath}"
        )


def _spec(relpath: str, size_bytes: int, sha256: str) -> Tuple[str, DataFileSpec]:
    return relpath, DataFileSpec(relpath=relpath, size_bytes=size_bytes, sha256=sha256)


#: Every upstream data file this replication can use, keyed by repo-relative path.
#: Sizes and digests measured from a clean clone at :data:`UPSTREAM_COMMIT`.
DATA_MANIFEST: Dict[str, DataFileSpec] = dict(
    [
        _spec(
            "qa_data/nq-open-oracle.jsonl.gz",
            900334,
            "6a7f72b846f663a133ef952244e5a4e43f06bcf190afe34e8fbe687ce0d1653e",
        ),
        _spec(
            "qa_data/10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl.gz",
            7801797,
            "192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9",
        ),
        _spec(
            "qa_data/10_total_documents/nq-open-10_total_documents_gold_at_4.jsonl.gz",
            7806238,
            "f8425253dafc19cd9de31c06ab6c8b57d4da2bf48b8ea2d0790b805fb3aa1968",
        ),
        _spec(
            "qa_data/10_total_documents/nq-open-10_total_documents_gold_at_9.jsonl.gz",
            7774527,
            "fab97badc6d69bae53c805bb198e8742a78ddf3c40a076989ef62891b76c997f",
        ),
        _spec(
            "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz",
            15151873,
            "69a8dd87a45f07fb3ed9b804917a89db975cd6e488682fc838edca029cc3881f",
        ),
        _spec(
            "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_4.jsonl.gz",
            15163003,
            "c5e44700c4db24536f139f7383e3dd002413515ee889aeadc72767dd7f289682",
        ),
        _spec(
            "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_9.jsonl.gz",
            15155756,
            "cbada411f70235afc47d9b5407e9358ad72eb021742402ff032c8b65076ffb2f",
        ),
        _spec(
            "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_14.jsonl.gz",
            15152418,
            "86c1a25093615baefdd1ee4db7a993548b3bef24ca02eb9bd8db43fbedf517f2",
        ),
        _spec(
            "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_19.jsonl.gz",
            15114700,
            "d79b715e1b8c8301335e93cb01659f68c24f0253a808c3190b06d21896506c6c",
        ),
        _spec(
            "qa_data/30_total_documents/nq-open-30_total_documents_gold_at_0.jsonl.gz",
            22465103,
            "0500933e0957ee264a8bf57ca4cd58042124f222c814302dbe2e3af788ff4e95",
        ),
        _spec(
            "qa_data/30_total_documents/nq-open-30_total_documents_gold_at_4.jsonl.gz",
            22477142,
            "3cfc86ec1056d15d32c3e0ae115bf55c1c7e14fa6bea53e1a3d57af18329cb56",
        ),
        _spec(
            "qa_data/30_total_documents/nq-open-30_total_documents_gold_at_9.jsonl.gz",
            22475002,
            "6da79584eac95b3d388fe6702f7c4ac32c50f8431865ec591c8af3de485128e2",
        ),
        _spec(
            "qa_data/30_total_documents/nq-open-30_total_documents_gold_at_14.jsonl.gz",
            22472929,
            "14ec97b971d111e381b8ad6a3c14eb87f6bff83401678afcdc32ec4afed9aeb7",
        ),
        _spec(
            "qa_data/30_total_documents/nq-open-30_total_documents_gold_at_19.jsonl.gz",
            22464022,
            "aef115bb3855b6d8ece2f6f1222cd3470bbeedd3f7c52bd106979810d09940e0",
        ),
        _spec(
            "qa_data/30_total_documents/nq-open-30_total_documents_gold_at_24.jsonl.gz",
            22459904,
            "bf9307930a0ed7fe151012d5619c4978c1dc13bf51f70481242839704d9ac477",
        ),
        _spec(
            "qa_data/30_total_documents/nq-open-30_total_documents_gold_at_29.jsonl.gz",
            22419963,
            "e768fcd157c321e0423432decb8623f537e2a97e2dd570873261f1acb6e2dfc6",
        ),
        _spec(
            "kv_retrieval_data/kv-retrieval-75_keys.jsonl.gz",
            1667554,
            "eb6d61fc3473349f2b86df76757c6ab659cc929d907d24caf1c69928e03f6203",
        ),
        _spec(
            "kv_retrieval_data/kv-retrieval-140_keys.jsonl.gz",
            3104449,
            "66d54aae7e38779da482895ccd538faaf5309f144e54f30dc530a95091581fa1",
        ),
        _spec(
            "kv_retrieval_data/kv-retrieval-300_keys.jsonl.gz",
            6641428,
            "47c795feb8ee5e47e7a151f866babfca968c786818fed3d7caa5ee6e9fedc70a",
        ),
    ]
)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def qa_positional_path(root: pathlib.Path, num_documents: int, gold_index: int) -> pathlib.Path:
    """Path to the multi-document QA file for a (num_documents, gold_index) cell.

    Args:
        root: Data root, i.e. the directory containing ``qa_data/``.
        num_documents: 10, 20 or 30.
        gold_index: A position available for that document count.

    Returns:
        Path to the gzipped JSONL file (which may not exist yet).

    Raises:
        ValueError: If the combination is not one the upstream repo ships.
    """
    if num_documents not in QA_GOLD_INDICES:
        raise ValueError(f"num_documents must be one of {sorted(QA_GOLD_INDICES)}, got {num_documents}")
    if gold_index not in QA_GOLD_INDICES[num_documents]:
        raise ValueError(
            f"gold_index {gold_index} is not available for {num_documents} documents; "
            f"available: {QA_GOLD_INDICES[num_documents]}"
        )
    return (
        pathlib.Path(root)
        / "qa_data"
        / f"{num_documents}_total_documents"
        / f"nq-open-{num_documents}_total_documents_gold_at_{gold_index}.jsonl.gz"
    )


def qa_oracle_path(root: pathlib.Path) -> pathlib.Path:
    """Path to the oracle file (one document per question: the gold passage).

    This file is also the source of the closed-book arm -- the paper runs closed-book
    over the same questions, simply discarding the documents.
    """
    return pathlib.Path(root) / "qa_data" / "nq-open-oracle.jsonl.gz"


def kv_path(root: pathlib.Path, num_keys: int) -> pathlib.Path:
    """Path to the key-value retrieval file with ``num_keys`` pairs per example.

    Upstream ships :data:`KV_NUM_KEYS` = ``(75, 140, 300)``. Larger key counts are
    permitted here because the 2026 extension needs context lengths the original
    could not test, and the paper itself provides the generator for them
    (``scripts/make_kv_retrieval_data.py`` upstream, ported to this repo). Files
    beyond the shipped three are **generated locally** and therefore are not in
    :data:`DATA_MANIFEST` -- they carry no upstream checksum, and any result that
    uses them must say so.

    Args:
        root: Data root, i.e. the directory containing ``kv_retrieval_data/``.
        num_keys: Key-value pairs per example. Must be >= 2.

    Returns:
        Path to the gzipped JSONL file (which may not exist yet).

    Raises:
        ValueError: If ``num_keys`` is below 2.
    """
    if num_keys < 2:
        raise ValueError(f"num_keys must be >= 2, got {num_keys}")
    return pathlib.Path(root) / "kv_retrieval_data" / f"kv-retrieval-{num_keys}_keys.jsonl.gz"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def iter_jsonl_gz(path: pathlib.Path) -> Iterator[Dict[str, Any]]:
    """Stream records from a gzipped JSONL file.

    Streaming matters here: the 30-document files are ~250 MB decompressed and a
    full run touches several of them.

    Args:
        path: Path to a ``.jsonl.gz`` file.

    Yields:
        One decoded JSON object per line.

    Raises:
        FileNotFoundError: With a message pointing at ``scripts/download_data.py``.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            "The paper's data is not vendored in this repo (it is ~257 MB).\n"
            "Fetch it with:  python scripts/download_data.py"
        )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl_gz(path: pathlib.Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a gzipped JSONL file into memory.

    Args:
        path: Path to a ``.jsonl.gz`` file.
        limit: If given, stop after this many records.

    Returns:
        A list of decoded records, in file order.
    """
    out: List[Dict[str, Any]] = []
    for i, record in enumerate(iter_jsonl_gz(path)):
        if limit is not None and i >= limit:
            break
        out.append(record)
    return out


def sha256_file(path: pathlib.Path, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def missing_data_files(root: pathlib.Path, relpaths: Optional[Iterable[str]] = None) -> List[str]:
    """Return the repo-relative paths of upstream data files not present under ``root``.

    Args:
        root: Data root to check.
        relpaths: Restrict the check to these manifest keys; default is all of them.

    Returns:
        Sorted list of missing repo-relative paths (empty if everything is present).
    """
    keys = list(relpaths) if relpaths is not None else list(DATA_MANIFEST)
    return sorted(rel for rel in keys if not (pathlib.Path(root) / rel).exists())


def require_data(root: pathlib.Path, relpaths: Iterable[str]) -> None:
    """Raise a helpful error if any of ``relpaths`` is absent under ``root``.

    Called by the runner before it spends a cent, so a missing download fails in
    the first second of a run rather than halfway through.
    """
    missing = missing_data_files(root, relpaths)
    if missing:
        listed = "\n  ".join(missing)
        raise FileNotFoundError(
            f"{len(missing)} required data file(s) missing under {root}:\n  {listed}\n"
            "Fetch them with:  python scripts/download_data.py"
        )


# --------------------------------------------------------------------------- #
# The paired design: one question subset, shared by every cell
# --------------------------------------------------------------------------- #
def question_id(question: str) -> str:
    """Stable 16-hex-character id for a question string.

    Content-addressed rather than positional, so a subset stays meaningful even if
    upstream ever reorders a file. (As of commit 29b8a6d all QA files share one
    order and contain no duplicate questions -- ``tests/test_data.py`` checks this
    when the data is present.)
    """
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def canonical_question_ids(root: pathlib.Path) -> List[str]:
    """Read the question universe (in file order) from the oracle file.

    The oracle file is used because it is by far the smallest (0.9 MB) and contains
    exactly the same 2,655 questions as every positional file.
    """
    return [question_id(rec["question"]) for rec in iter_jsonl_gz(qa_oracle_path(root))]


@dataclass(frozen=True)
class QuestionSubset:
    """An immutable, seeded selection of question ids shared across all cells.

    Attributes:
        seed: The seed used to draw it.
        n: Number of questions requested.
        question_ids: The selected ids, sorted (order is not experimentally
            meaningful; sorting makes the signature order-independent).
        universe_size: Size of the pool the subset was drawn from.
    """

    seed: int
    n: int
    question_ids: Tuple[str, ...]
    universe_size: int
    _id_set: frozenset = field(default_factory=frozenset, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_id_set", frozenset(self.question_ids))

    def __contains__(self, qid: object) -> bool:
        return qid in self._id_set

    def __len__(self) -> int:
        return len(self.question_ids)

    def signature(self) -> str:
        """A digest of the exact question set, for the run manifest.

        Two cells with the same signature provably scored the same questions, which
        is the precondition for the paired tests in :mod:`litm2026.stats`.
        """
        joined = "\n".join(self.question_ids).encode("utf-8")
        return hashlib.sha256(joined).hexdigest()[:32]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable summary for manifests (ids included: n is small)."""
        return {
            "seed": self.seed,
            "n": self.n,
            "universe_size": self.universe_size,
            "signature": self.signature(),
            "question_ids": list(self.question_ids),
        }


def select_question_subset(universe: Sequence[str], n: int, seed: int) -> QuestionSubset:
    """Draw the fixed question subset used by *every* cell of a run.

    The draw is taken from ``sorted(set(universe))`` rather than from the sequence as
    given, so the result depends only on the *set* of questions and the seed -- never
    on which file the universe happened to be read from or in what order.

    Args:
        universe: Question ids (duplicates tolerated and ignored).
        n: Subset size. Must be ``1 <= n <= len(set(universe))``.
        seed: RNG seed. Change this and you get a different (equally valid) subset;
            keeping it fixed is what makes cells comparable and reruns reproducible.

    Returns:
        A :class:`QuestionSubset`.

    Raises:
        ValueError: If ``n`` is out of range.
    """
    pool = sorted(set(universe))
    if not 1 <= n <= len(pool):
        raise ValueError(f"n must be in [1, {len(pool)}], got {n}")
    rng = random.Random(seed)
    chosen = rng.sample(pool, n)
    return QuestionSubset(seed=seed, n=n, question_ids=tuple(sorted(chosen)), universe_size=len(pool))


def apply_question_subset(
    records: Iterable[Mapping[str, Any]],
    subset: QuestionSubset,
    *,
    strict: bool = True,
) -> List[Dict[str, Any]]:
    """Filter records down to the subset, in the file's own order.

    Args:
        records: Records with a ``question`` key (typically a streamed QA file).
        subset: The subset produced by :func:`select_question_subset`.
        strict: If ``True`` (default), raise when the file does not contain every
            selected question. Silently returning a short list here would break the
            pairing without anyone noticing, which is exactly the failure mode this
            module exists to prevent.

    Returns:
        The matching records, each with an added ``question_id`` key.

    Raises:
        ValueError: If ``strict`` and some selected questions are absent.
    """
    kept: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        qid = question_id(record["question"])
        if qid in subset and qid not in seen:
            enriched = dict(record)
            enriched["question_id"] = qid
            kept.append(enriched)
            seen.add(qid)
    if strict and len(seen) != len(subset):
        missing = sorted(set(subset.question_ids) - seen)
        raise ValueError(
            f"Subset pairing broken: {len(missing)} of {len(subset)} selected questions "
            f"are absent from this file (first few: {missing[:3]})."
        )
    return kept


def select_index_subset(universe_size: int, n: int, seed: int) -> Tuple[int, ...]:
    """Draw a fixed subset of example *indices*, sorted ascending.

    Used for the key-value retrieval task, whose examples have no natural content id
    (the keys and values are random UUIDs, and the same 500 examples are reused at
    every gold position -- so index pairing is exact).

    Args:
        universe_size: Total examples in the file (500 for every shipped KV file).
        n: Subset size.
        seed: RNG seed.

    Returns:
        Sorted tuple of indices.

    Raises:
        ValueError: If ``n`` is out of range.
    """
    if not 1 <= n <= universe_size:
        raise ValueError(f"n must be in [1, {universe_size}], got {n}")
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(range(universe_size), n)))


def evenly_spaced_positions(num_items: int, num_positions: int = 5) -> Tuple[int, ...]:
    """Evenly spaced gold positions across ``num_items`` slots, inclusive of both ends.

    Uses ``floor(i * (num_items - 1) / (num_positions - 1))``, which reproduces the
    paper's key-value positions exactly: for 140 keys and 5 positions it returns
    ``(0, 34, 69, 104, 139)``, the sweep reported in the original ``EXPERIMENTS.md``.
    ``tests/test_data.py`` asserts that.

    Args:
        num_items: Number of slots (documents or key-value pairs).
        num_positions: How many positions to sample.

    Returns:
        A sorted tuple of distinct indices.

    Raises:
        ValueError: If arguments are too small to produce distinct positions.
    """
    if num_items < 2:
        raise ValueError(f"num_items must be >= 2, got {num_items}")
    if not 2 <= num_positions <= num_items:
        raise ValueError(f"num_positions must be in [2, {num_items}], got {num_positions}")
    positions = [(i * (num_items - 1)) // (num_positions - 1) for i in range(num_positions)]
    return tuple(sorted(set(positions)))
