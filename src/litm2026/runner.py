"""The experiment runner: async, resumable, rate-limited, cost-capped.

Run one config::

    python -m litm2026.runner --config experiments/configs/pilot.yaml --dry-run
    python -m litm2026.runner --config experiments/configs/pilot.yaml

FOUR PROPERTIES THAT ARE NOT NEGOTIABLE
---------------------------------------
**Resumable.** Every completed call is appended to
``results/raw/<run_id>/<cell_id>.jsonl`` and flushed *and* ``fsync``-ed before the
next call starts. On restart the runner reads those files, skips the examples that
already succeeded, and continues. A crash at hour three costs you the call that was
in flight, not the run. Resume is the default; there is no flag to remember.

**Cost-capped.** :class:`litm2026.costs.CostTracker` is checked before every call
and again after every response. When the cap is hit the run stops immediately with
a message that says how to resume. If a model has no configured price and a dollar
cap is set, the run is refused *before the first request* rather than proceeding
uncapped.

**Rate-limit aware.** Each model gets its own RPM limiter and shared 429 cooldown
(see :mod:`litm2026.providers`). Concurrency is a config value, and the shipped
free-tier configs keep it at 2.

**Every raw response is saved.** Response text, provider usage counts, finish
reason, latency and the provider's raw JSON body all land in ``results/raw/``. If a
number appears in the README, the response that produced it is on disk. Prompts are
*not* stored by default (they are large and exactly reconstructable from the pinned
data files, the pinned prompt templates and the example id); set
``store_prompts: true`` in the config if you want them inline anyway -- a SHA-256 of
the prompt is always stored either way.

ARMS
----
``qa_closedbook``   No documents. The floor.
``qa_oracle``       Only the gold document. The ceiling.
``qa_positional``   The sweep: N total documents, gold at each available position.
``kv_retrieval``    Synthetic key-value retrieval, gold key swept across positions.
                    This is the arm that scales past the paper's context lengths.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from . import __version__
from .costs import BudgetExceeded, CostTracker, TokenPrice, UnknownPricingError, estimate_tokens_from_chars
from .data import (
    DEFAULT_DATA_ROOT,
    QA_GOLD_INDICES,
    UPSTREAM_COMMIT,
    QuestionSubset,
    apply_question_subset,
    canonical_question_ids,
    evenly_spaced_positions,
    iter_jsonl_gz,
    kv_path,
    qa_oracle_path,
    qa_positional_path,
    require_data,
    select_index_subset,
    select_question_subset,
)
from .prompting import (
    ChatAdaptation,
    documents_from_example,
    get_closedbook_qa_prompt,
    get_kv_retrieval_prompt,
    get_qa_prompt,
    place_kv_gold_at,
    prompt_file_digests,
    to_chat_messages,
)
from .providers import PROVIDER_SPECS, BaseProvider, RetryPolicy, build_provider

__all__ = [
    "RAW_SCHEMA_VERSION",
    "EXPERIMENT_KINDS",
    "RunConfig",
    "ModelConfig",
    "ExperimentConfig",
    "BudgetConfig",
    "Task",
    "CheckpointWriter",
    "load_config",
    "build_tasks",
    "completed_example_ids",
    "run",
    "main",
]

RAW_SCHEMA_VERSION = 2

EXPERIMENT_KINDS = ("qa_closedbook", "qa_oracle", "qa_positional", "kv_retrieval")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _reject_unknown(section: str, data: Mapping[str, Any], allowed: Iterable[str]) -> None:
    """Fail loudly on unknown config keys.

    A typo like ``gold_index:`` instead of ``gold_indices:`` would otherwise silently
    run the wrong experiment, which is worse than a crash.
    """
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown key(s) in {section}: {unknown}. Allowed: {sorted(allowed)}")


@dataclass(frozen=True)
class ModelConfig:
    """One model under test.

    Attributes:
        key: Short identifier used in cell ids, filenames and figures.
        provider: A key of :data:`litm2026.providers.PROVIDER_SPECS`.
        model: Provider-side model id, verbatim.
        label: Human-readable name for charts and tables.
        rpm: Requests per minute. ``None`` uses the provider's conservative default.
        max_context_tokens: Optional guard; prompts whose *estimated* size exceeds it
            are handled per ``on_oversize``.
        pricing: Per-token prices with a source string (see :mod:`litm2026.costs`).
        extra_headers: Extra HTTP headers (OpenRouter attribution, for instance).
    """

    key: str
    provider: str
    model: str
    label: str
    rpm: Optional[float] = None
    max_context_tokens: Optional[int] = None
    pricing: TokenPrice = field(default_factory=TokenPrice)
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        _reject_unknown(
            "models[]",
            data,
            ("key", "provider", "model", "label", "rpm", "max_context_tokens", "pricing", "extra_headers"),
        )
        for required in ("key", "provider", "model"):
            if not data.get(required):
                raise ValueError(f"models[] entry missing required key {required!r}")
        provider = str(data["provider"])
        if provider not in PROVIDER_SPECS:
            raise ValueError(f"Unknown provider {provider!r}; known: {sorted(PROVIDER_SPECS)}")
        return cls(
            key=str(data["key"]),
            provider=provider,
            model=str(data["model"]),
            label=str(data.get("label") or data["model"]),
            rpm=None if data.get("rpm") is None else float(data["rpm"]),
            max_context_tokens=(None if data.get("max_context_tokens") is None else int(data["max_context_tokens"])),
            pricing=TokenPrice.from_mapping(data.get("pricing")),
            extra_headers=dict(data.get("extra_headers") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form for the run manifest."""
        return {
            "key": self.key,
            "provider": self.provider,
            "model": self.model,
            "label": self.label,
            "rpm": self.rpm,
            "max_context_tokens": self.max_context_tokens,
            "pricing": self.pricing.to_dict(),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """One arm of the experiment matrix.

    Attributes:
        kind: One of :data:`EXPERIMENT_KINDS`.
        n: Examples for this arm; defaults to the run-level ``n``.
        num_documents: Total documents, for ``qa_positional`` (10, 20 or 30).
        gold_indices: Gold positions to sweep. ``None`` means "every position the
            upstream data provides" (QA) or five evenly spaced positions (KV).
        num_keys: Key-value pairs per example, for ``kv_retrieval`` (75, 140, 300).
    """

    kind: str
    n: Optional[int] = None
    num_documents: Optional[int] = None
    gold_indices: Optional[Tuple[int, ...]] = None
    num_keys: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        _reject_unknown("experiments[]", data, ("kind", "n", "num_documents", "gold_indices", "num_keys"))
        kind = str(data.get("kind", ""))
        if kind not in EXPERIMENT_KINDS:
            raise ValueError(f"Unknown experiment kind {kind!r}; allowed: {EXPERIMENT_KINDS}")
        if kind == "qa_positional" and data.get("num_documents") is None:
            raise ValueError("qa_positional requires `num_documents`")
        if kind == "kv_retrieval" and data.get("num_keys") is None:
            raise ValueError("kv_retrieval requires `num_keys`")
        gold = data.get("gold_indices")
        return cls(
            kind=kind,
            n=None if data.get("n") is None else int(data["n"]),
            num_documents=None if data.get("num_documents") is None else int(data["num_documents"]),
            gold_indices=None if gold is None else tuple(int(g) for g in gold),
            num_keys=None if data.get("num_keys") is None else int(data["num_keys"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form for the run manifest."""
        return {
            "kind": self.kind,
            "n": self.n,
            "num_documents": self.num_documents,
            "gold_indices": list(self.gold_indices) if self.gold_indices else None,
            "num_keys": self.num_keys,
        }


@dataclass(frozen=True)
class BudgetConfig:
    """Spend limits. See :mod:`litm2026.costs` for why pricing lives in config.

    Attributes:
        max_usd: Hard dollar cap for the whole run. ``None`` disables it.
        max_total_tokens: Hard token cap. Useful when every model is free-tier and a
            dollar cap is meaningless.
        max_calls: Hard cap on the number of API calls, as a last-resort circuit
            breaker against a runaway loop.
        stop_on_unknown_pricing: Refuse to start a dollar-capped run when a model has
            no configured price. Leave this ``true``.
    """

    max_usd: Optional[float] = None
    max_total_tokens: Optional[int] = None
    max_calls: Optional[int] = None
    stop_on_unknown_pricing: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "BudgetConfig":
        data = data or {}
        _reject_unknown("budget", data, ("max_usd", "max_total_tokens", "max_calls", "stop_on_unknown_pricing"))
        return cls(
            max_usd=None if data.get("max_usd") is None else float(data["max_usd"]),
            max_total_tokens=None if data.get("max_total_tokens") is None else int(data["max_total_tokens"]),
            max_calls=None if data.get("max_calls") is None else int(data["max_calls"]),
            stop_on_unknown_pricing=bool(data.get("stop_on_unknown_pricing", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form for the run manifest."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RunConfig:
    """A complete, self-describing run. Every parameter lives here, not in code.

    Attributes:
        run_id: Names the output directory ``results/raw/<run_id>``. Reusing a
            run_id is how you resume.
        seed: Seeds the example subsets and every bootstrap downstream.
        n: Default examples per cell.
        max_new_tokens: 100, matching the paper's ``--max-new-tokens``.
        temperature: 0.0, matching the paper's greedy decoding.
        top_p: 1.0, matching the paper.
        chat_adaptation: How completion-style prompts become chat messages. A
            documented deviation from the original -- see
            :class:`litm2026.prompting.ChatAdaptation`.
        data_root: Where ``scripts/download_data.py`` put the paper's data.
        results_root: Where raw responses and figures go.
        concurrency: Simultaneous in-flight requests across all models.
        timeout_s: Per-request timeout.
        max_attempts: Attempts per call, including the first.
        store_prompts: Persist full prompt text in the raw records (large).
        on_oversize: ``"error"`` (default), ``"skip"`` or ``"ignore"`` when a prompt
            exceeds a model's ``max_context_tokens``. ``"skip"`` breaks the paired
            design, so it must be chosen deliberately.
        models: Models under test.
        experiments: Arms to run.
        budget: Spend limits.
    """

    run_id: str
    seed: int = 0
    n: int = 150
    max_new_tokens: int = 100
    temperature: float = 0.0
    top_p: float = 1.0
    chat_adaptation: ChatAdaptation = ChatAdaptation.USER_VERBATIM
    data_root: pathlib.Path = DEFAULT_DATA_ROOT
    results_root: pathlib.Path = pathlib.Path("results")
    concurrency: int = 2
    timeout_s: float = 180.0
    max_attempts: int = 6
    store_prompts: bool = False
    on_oversize: str = "error"
    models: Tuple[ModelConfig, ...] = ()
    experiments: Tuple[ExperimentConfig, ...] = ()
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    @property
    def raw_dir(self) -> pathlib.Path:
        """Directory holding this run's append-only checkpoint files."""
        return pathlib.Path(self.results_root) / "raw" / self.run_id

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable config, written verbatim into the run manifest."""
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "n": self.n,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "chat_adaptation": self.chat_adaptation.value,
            "data_root": str(self.data_root),
            "results_root": str(self.results_root),
            "concurrency": self.concurrency,
            "timeout_s": self.timeout_s,
            "max_attempts": self.max_attempts,
            "store_prompts": self.store_prompts,
            "on_oversize": self.on_oversize,
            "models": [m.to_dict() for m in self.models],
            "experiments": [e.to_dict() for e in self.experiments],
            "budget": self.budget.to_dict(),
        }


_RUN_KEYS = (
    "run_id",
    "seed",
    "n",
    "max_new_tokens",
    "temperature",
    "top_p",
    "chat_adaptation",
    "data_root",
    "results_root",
    "concurrency",
    "timeout_s",
    "max_attempts",
    "store_prompts",
    "on_oversize",
    "models",
    "experiments",
    "budget",
)


def load_config(path: pathlib.Path) -> RunConfig:
    """Parse and validate a YAML run config.

    Unknown keys are rejected rather than ignored, and every enum-like field is
    checked here so an invalid run fails in the first second instead of after the
    first API call.

    Args:
        path: Path to an ``experiments/configs/*.yaml`` file.

    Returns:
        A validated :class:`RunConfig`.

    Raises:
        ValueError: On any invalid or unknown field.
        FileNotFoundError: If the config does not exist.
    """
    path = pathlib.Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: top level of a config must be a mapping")
    _reject_unknown(str(path), data, _RUN_KEYS)

    if not data.get("run_id"):
        raise ValueError(f"{path}: `run_id` is required")
    models = tuple(ModelConfig.from_dict(m) for m in (data.get("models") or []))
    if not models:
        raise ValueError(f"{path}: at least one model is required")
    keys = [m.key for m in models]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate model keys: {keys}")
    experiments = tuple(ExperimentConfig.from_dict(e) for e in (data.get("experiments") or []))
    if not experiments:
        raise ValueError(f"{path}: at least one experiment is required")

    on_oversize = str(data.get("on_oversize", "error"))
    if on_oversize not in ("error", "skip", "ignore"):
        raise ValueError(f"{path}: on_oversize must be one of error/skip/ignore, got {on_oversize!r}")

    concurrency = int(data.get("concurrency", 2))
    if concurrency < 1:
        raise ValueError(f"{path}: concurrency must be >= 1")

    return RunConfig(
        run_id=str(data["run_id"]),
        seed=int(data.get("seed", 0)),
        n=int(data.get("n", 150)),
        max_new_tokens=int(data.get("max_new_tokens", 100)),
        temperature=float(data.get("temperature", 0.0)),
        top_p=float(data.get("top_p", 1.0)),
        chat_adaptation=ChatAdaptation(str(data.get("chat_adaptation", ChatAdaptation.USER_VERBATIM.value))),
        data_root=pathlib.Path(data.get("data_root", DEFAULT_DATA_ROOT)),
        results_root=pathlib.Path(data.get("results_root", "results")),
        concurrency=concurrency,
        timeout_s=float(data.get("timeout_s", 180.0)),
        max_attempts=int(data.get("max_attempts", 6)),
        store_prompts=bool(data.get("store_prompts", False)),
        on_oversize=on_oversize,
        models=models,
        experiments=experiments,
        budget=BudgetConfig.from_dict(data.get("budget")),
    )


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    """One (cell, example) unit of work: exactly one API call.

    Attributes:
        cell_id: Stable identifier for the cell; also the checkpoint filename.
        experiment: Arm name.
        model_key: Which model.
        example_id: Stable id of the example (question id for QA, ``kv<index>`` for
            key-value retrieval).
        messages: Chat messages to send.
        gold_answers: Accepted answer strings (QA arms).
        gold_value: Gold value string (KV arm).
        num_documents: Total documents in the prompt, for the results table.
        num_keys: Total key-value pairs in the prompt.
        gold_index: Position of the gold item.
        question: The question (QA) or the queried key (KV), for traceability.
        prompt_chars: Character length of the assembled prompt.
        prompt_sha256: Digest of the assembled prompt, so a response can be tied to
            the exact bytes that produced it without storing them.
        subset_signature: Digest of the example set this cell used. Cells with equal
            signatures provably scored the same examples, which is the precondition
            for the paired tests in :mod:`litm2026.stats`.
    """

    cell_id: str
    experiment: str
    model_key: str
    example_id: str
    messages: Tuple[Dict[str, str], ...]
    gold_answers: Tuple[str, ...] = ()
    gold_value: Optional[str] = None
    num_documents: Optional[int] = None
    num_keys: Optional[int] = None
    gold_index: Optional[int] = None
    question: Optional[str] = None
    prompt_chars: int = 0
    prompt_sha256: str = ""
    subset_signature: str = ""

    @property
    def estimated_prompt_tokens(self) -> int:
        """Crude pre-run token estimate (see :mod:`litm2026.costs`). Projection only."""
        return estimate_tokens_from_chars("\n".join(m["content"] for m in self.messages))


def _cell_id(
    model_key: str,
    experiment: str,
    *,
    num_documents: Optional[int] = None,
    num_keys: Optional[int] = None,
    gold_index: Optional[int] = None,
) -> str:
    """Build the stable cell identifier used for checkpoint filenames."""
    parts = [model_key, experiment]
    if num_documents is not None:
        parts.append(f"docs{num_documents}")
    if num_keys is not None:
        parts.append(f"keys{num_keys}")
    if gold_index is not None:
        parts.append(f"gold{gold_index}")
    slug = "__".join(parts)
    return "".join(ch if (ch.isalnum() or ch in "_-.") else "-" for ch in slug)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _required_relpaths(config: RunConfig) -> List[str]:
    """Repo-relative data files this config needs, for the pre-flight existence check."""
    needed: List[str] = []
    for exp in config.experiments:
        if exp.kind in ("qa_closedbook", "qa_oracle"):
            needed.append("qa_data/nq-open-oracle.jsonl.gz")
        elif exp.kind == "qa_positional":
            assert exp.num_documents is not None
            for gold in exp.gold_indices or QA_GOLD_INDICES.get(exp.num_documents, ()):
                needed.append(qa_positional_path(pathlib.Path(""), exp.num_documents, gold).as_posix())
        elif exp.kind == "kv_retrieval":
            assert exp.num_keys is not None
            needed.append(kv_path(pathlib.Path(""), exp.num_keys).as_posix())
    return sorted(set(needed))


def _narrow_subset(subset: QuestionSubset, n: int) -> QuestionSubset:
    """Take the first ``n`` question ids of a subset, deterministically.

    Subsets are stored sorted by question id, so narrowing is stable and *nested*:
    an arm with a smaller ``n`` uses a strict subset of the questions used by the
    larger arms. That keeps every arm comparable to every other arm.
    """
    if n >= len(subset.question_ids):
        return subset
    return QuestionSubset(
        seed=subset.seed,
        n=n,
        question_ids=subset.question_ids[:n],
        universe_size=subset.universe_size,
    )


def build_tasks(config: RunConfig) -> Tuple[List[Task], Dict[str, Any]]:
    """Materialise every API call the config implies, without making any of them.

    This is where the paired design is enforced: one question subset is drawn from
    the canonical universe with ``config.seed`` and reused by *every* QA cell, and
    one index subset is reused by every KV cell at every gold position.

    Args:
        config: A validated run config.

    Returns:
        ``(tasks, plan)`` where ``plan`` is a JSON-serializable description of the
        matrix (cells, call counts, estimated tokens, subset signature), used for
        ``--dry-run`` output and written into the run manifest.

    Raises:
        FileNotFoundError: If required data files are missing.
        ValueError: On an unsupported gold position, or when a prompt exceeds a
            model's ``max_context_tokens`` and ``on_oversize`` is ``"error"``.
    """
    require_data(config.data_root, _required_relpaths(config))

    qa_arms = [e for e in config.experiments if e.kind.startswith("qa_")]
    subset: Optional[QuestionSubset] = None
    if qa_arms:
        universe = canonical_question_ids(config.data_root)
        n_qa = max((e.n or config.n) for e in qa_arms)
        subset = select_question_subset(universe, n=n_qa, seed=config.seed)

    tasks: List[Task] = []
    cells: List[Dict[str, Any]] = []

    for exp in config.experiments:
        n = exp.n or config.n

        if exp.kind in ("qa_closedbook", "qa_oracle"):
            assert subset is not None
            arm_subset = _narrow_subset(subset, n)
            examples = apply_question_subset(iter_jsonl_gz(qa_oracle_path(config.data_root)), arm_subset)
            for model in config.models:
                cell = _cell_id(model.key, exp.kind)
                built: List[Task] = []
                for example in examples:
                    if exp.kind == "qa_closedbook":
                        prompt = get_closedbook_qa_prompt(example["question"])
                    else:
                        prompt = get_qa_prompt(
                            question=example["question"],
                            documents=documents_from_example(example),
                            mention_random_ordering=False,
                            query_aware_contextualization=False,
                        )
                    task = _build_qa_task(
                        config, model, cell, exp.kind, example, prompt, arm_subset.signature()
                    )
                    if task is not None:
                        built.append(task)
                tasks.extend(built)
                cells.append(_cell_summary(cell, exp.kind, model, built))

        elif exp.kind == "qa_positional":
            assert subset is not None and exp.num_documents is not None
            arm_subset = _narrow_subset(subset, n)
            for gold_index in exp.gold_indices or QA_GOLD_INDICES[exp.num_documents]:
                path = qa_positional_path(config.data_root, exp.num_documents, gold_index)
                examples = apply_question_subset(iter_jsonl_gz(path), arm_subset)
                for model in config.models:
                    cell = _cell_id(model.key, exp.kind, num_documents=exp.num_documents, gold_index=gold_index)
                    built = []
                    for example in examples:
                        prompt = get_qa_prompt(
                            question=example["question"],
                            documents=documents_from_example(example),
                            mention_random_ordering=False,
                            query_aware_contextualization=False,
                        )
                        task = _build_qa_task(
                            config,
                            model,
                            cell,
                            exp.kind,
                            example,
                            prompt,
                            arm_subset.signature(),
                            num_documents=exp.num_documents,
                            gold_index=gold_index,
                        )
                        if task is not None:
                            built.append(task)
                    tasks.extend(built)
                    cells.append(
                        _cell_summary(
                            cell, exp.kind, model, built, num_documents=exp.num_documents, gold_index=gold_index
                        )
                    )

        elif exp.kind == "kv_retrieval":
            assert exp.num_keys is not None
            all_examples = list(iter_jsonl_gz(kv_path(config.data_root, exp.num_keys)))
            keep = select_index_subset(len(all_examples), n=min(n, len(all_examples)), seed=config.seed)
            signature = _sha256_text(",".join(str(i) for i in keep))[:32]
            for gold_index in exp.gold_indices or evenly_spaced_positions(exp.num_keys, 5):
                if not 0 <= gold_index < exp.num_keys:
                    raise ValueError(f"kv gold_index {gold_index} out of range for {exp.num_keys} keys")
                for model in config.models:
                    cell = _cell_id(model.key, exp.kind, num_keys=exp.num_keys, gold_index=gold_index)
                    built = []
                    for example_index in keep:
                        example = all_examples[example_index]
                        records = place_kv_gold_at(
                            example["ordered_kv_records"], example["key"], example["value"], gold_index
                        )
                        prompt = get_kv_retrieval_prompt(data=records, key=example["key"])
                        task = _apply_oversize_policy(
                            config,
                            model,
                            Task(
                                cell_id=cell,
                                experiment=exp.kind,
                                model_key=model.key,
                                example_id=f"kv{example_index}",
                                messages=tuple(to_chat_messages(prompt, config.chat_adaptation)),
                                gold_value=example["value"],
                                num_keys=exp.num_keys,
                                gold_index=gold_index,
                                question=example["key"],
                                prompt_chars=len(prompt),
                                prompt_sha256=_sha256_text(prompt),
                                subset_signature=signature,
                            ),
                        )
                        if task is not None:
                            built.append(task)
                    tasks.extend(built)
                    cells.append(
                        _cell_summary(cell, exp.kind, model, built, num_keys=exp.num_keys, gold_index=gold_index)
                    )

    plan: Dict[str, Any] = {
        "run_id": config.run_id,
        "n_cells": len(cells),
        "n_calls": len(tasks),
        "estimated_prompt_tokens_TOTAL_APPROX": sum(t.estimated_prompt_tokens for t in tasks),
        "estimate_method": "prompt_chars / 4.0 -- APPROXIMATE, projection only, never reported as a result",
        "question_subset": subset.to_dict() if subset is not None else None,
        "cells": cells,
    }
    return tasks, plan


def _build_qa_task(
    config: RunConfig,
    model: ModelConfig,
    cell_id: str,
    experiment: str,
    example: Mapping[str, Any],
    prompt: str,
    subset_signature: str,
    *,
    num_documents: Optional[int] = None,
    gold_index: Optional[int] = None,
) -> Optional[Task]:
    """Wrap one QA prompt into a :class:`Task`, applying the oversize policy."""
    return _apply_oversize_policy(
        config,
        model,
        Task(
            cell_id=cell_id,
            experiment=experiment,
            model_key=model.key,
            example_id=str(example["question_id"]),
            messages=tuple(to_chat_messages(prompt, config.chat_adaptation)),
            gold_answers=tuple(example["answers"]),
            num_documents=num_documents,
            gold_index=gold_index,
            question=example["question"],
            prompt_chars=len(prompt),
            prompt_sha256=_sha256_text(prompt),
            subset_signature=subset_signature,
        ),
    )


def _apply_oversize_policy(config: RunConfig, model: ModelConfig, task: Task) -> Optional[Task]:
    """Enforce ``max_context_tokens``. Returns ``None`` when the task is skipped."""
    limit = model.max_context_tokens
    if limit is None or config.on_oversize == "ignore":
        return task
    estimated = task.estimated_prompt_tokens
    if estimated <= limit:
        return task
    if config.on_oversize == "skip":
        return None
    raise ValueError(
        f"Prompt for cell {task.cell_id} example {task.example_id} is ~{estimated} tokens "
        f"(estimated), above max_context_tokens={limit} for model {model.key!r}.\n"
        "Either raise max_context_tokens, drop this cell from the config, or set "
        "`on_oversize: skip` -- but note that skipping breaks the paired design, because "
        "the affected cells no longer contain the same examples."
    )


def _cell_summary(
    cell_id: str,
    experiment: str,
    model: ModelConfig,
    tasks: Sequence[Task],
    *,
    num_documents: Optional[int] = None,
    num_keys: Optional[int] = None,
    gold_index: Optional[int] = None,
) -> Dict[str, Any]:
    """One row of the planned experiment matrix."""
    return {
        "cell_id": cell_id,
        "experiment": experiment,
        "model_key": model.key,
        "model": model.model,
        "provider": model.provider,
        "num_documents": num_documents,
        "num_keys": num_keys,
        "gold_index": gold_index,
        "n_calls": len(tasks),
        "max_prompt_chars": max((t.prompt_chars for t in tasks), default=0),
        "estimated_prompt_tokens_max_APPROX": max((t.estimated_prompt_tokens for t in tasks), default=0),
        "subset_signature": tasks[0].subset_signature if tasks else None,
    }


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def completed_example_ids(raw_dir: pathlib.Path, cell_id: str) -> set:
    """Read a cell's checkpoint file and return the example ids already done.

    Only records with no ``error`` count as done, so a transient failure is retried
    on the next run rather than silently leaving a hole in the cell. A truncated
    final line (hard kill mid-write) is ignored, and that example is simply redone.

    Args:
        raw_dir: ``results/raw/<run_id>``.
        cell_id: Cell whose checkpoint to inspect.

    Returns:
        Set of completed example ids (empty if the file does not exist).
    """
    path = pathlib.Path(raw_dir) / f"{cell_id}.jsonl"
    if not path.exists():
        return set()
    done: set = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            example_id = str(record.get("example_id"))
            if record.get("error"):
                done.discard(example_id)
            else:
                done.add(example_id)
    return done


class CheckpointWriter:
    """Append-only, fsync-ed JSONL writer -- one file per cell.

    ``fsync`` after every record is the difference between "resumable" and "probably
    resumable". It costs a few milliseconds per call, against API latencies measured
    in seconds.
    """

    def __init__(self, raw_dir: pathlib.Path) -> None:
        self.raw_dir = pathlib.Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._handles: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def write(self, cell_id: str, record: Mapping[str, Any]) -> None:
        """Append one record and flush it to stable storage before returning."""
        async with self._lock:
            handle = self._handles.get(cell_id)
            if handle is None:
                handle = open(self.raw_dir / f"{cell_id}.jsonl", "a", encoding="utf-8")
                self._handles[cell_id] = handle
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def close(self) -> None:
        """Close every open checkpoint file."""
        for handle in self._handles.values():
            with contextlib.suppress(Exception):
                handle.close()
        self._handles.clear()


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _base_record(config: RunConfig, task: Task, provider: BaseProvider) -> Dict[str, Any]:
    """The fields written for every attempt, successful or not."""
    record: Dict[str, Any] = {
        "schema_version": RAW_SCHEMA_VERSION,
        "run_id": config.run_id,
        "cell_id": task.cell_id,
        "experiment": task.experiment,
        "model_key": task.model_key,
        "provider": provider.provider_name,
        "model": provider.model,
        "num_documents": task.num_documents,
        "num_keys": task.num_keys,
        "gold_index": task.gold_index,
        "example_id": task.example_id,
        "question": task.question,
        "gold_answers": list(task.gold_answers),
        "gold_value": task.gold_value,
        "subset_signature": task.subset_signature,
        "chat_adaptation": config.chat_adaptation.value,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_new_tokens": config.max_new_tokens,
        "prompt_chars": task.prompt_chars,
        "prompt_sha256": task.prompt_sha256,
        "timestamp_utc": _utcnow(),
        "litm2026_version": __version__,
        "upstream_data_commit": UPSTREAM_COMMIT,
    }
    if config.store_prompts:
        record["messages"] = [dict(m) for m in task.messages]
    return record


async def _worker(
    name: str,
    queue: "asyncio.Queue",
    providers: Mapping[str, BaseProvider],
    config: RunConfig,
    tracker: CostTracker,
    writer: CheckpointWriter,
    stop: asyncio.Event,
    counters: Dict[str, int],
) -> None:
    """Pull tasks off the queue until it is drained or the run is stopped."""
    while True:
        task = await queue.get()
        try:
            if task is None or stop.is_set():
                return
            if config.budget.max_calls is not None and counters["calls"] >= config.budget.max_calls:
                print(f"[{name}] max_calls={config.budget.max_calls} reached; stopping.", flush=True)
                stop.set()
                return

            provider = providers[task.model_key]
            record = _base_record(config, task, provider)

            try:
                tracker.check_before_call()
                response = await provider.complete(
                    task.messages,
                    max_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )
            except BudgetExceeded as exc:
                print(f"\n*** BUDGET CAP REACHED ***\n{exc}\n", flush=True)
                stop.set()
                return
            except Exception as exc:  # noqa: BLE001 - recorded to disk, never swallowed
                counters["errors"] += 1
                record.update({"response_text": None, "error": f"{type(exc).__name__}: {exc}", "is_mock": False})
                await writer.write(task.cell_id, record)
                continue

            budget_stop = False
            cost_usd: Optional[float] = None
            try:
                cost_usd = tracker.record(
                    task.model_key,
                    input_tokens=int(response.input_tokens or 0),
                    output_tokens=int(response.output_tokens or 0),
                ).cost_usd
            except BudgetExceeded as exc:
                budget_stop = True
                print(f"\n*** BUDGET CAP REACHED ***\n{exc}\n", flush=True)
            except UnknownPricingError as exc:  # pragma: no cover - blocked at startup
                budget_stop = True
                print(f"\n*** PRICING ERROR ***\n{exc}\n", flush=True)

            counters["calls"] += 1
            record.update(
                {
                    "response_text": response.text,
                    "response_raw": response.raw,
                    "prompt_tokens": response.input_tokens,
                    "completion_tokens": response.output_tokens,
                    "finish_reason": response.finish_reason,
                    "latency_s": round(response.latency_s, 4),
                    "attempts": response.attempts,
                    "cost_usd": cost_usd,
                    "is_mock": response.is_mock,
                    "error": None,
                }
            )
            await writer.write(task.cell_id, record)

            if counters["calls"] % 25 == 0:
                spent = tracker.summary()
                print(
                    f"[progress] {counters['calls']} calls done, {counters['errors']} errors, "
                    f"{spent['total_tokens']:,} tokens, ${spent['total_usd']:.4f}",
                    flush=True,
                )
            if budget_stop:
                stop.set()
                return
        finally:
            queue.task_done()


async def run(config: RunConfig, *, dry_run: bool = False) -> int:
    """Execute (or plan) a run.

    Args:
        config: A validated run config.
        dry_run: Build every prompt, print the matrix and write
            ``dry_run_plan.json``, but make **zero** API calls. Always do this first.

    Returns:
        Process exit code: 0 success, 1 setup failure, 2 stopped early (cap hit).
    """
    raw_dir = config.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    tasks, plan = build_tasks(config)
    plan["prompt_template_sha256"] = prompt_file_digests()
    plan["built_at_utc"] = _utcnow()

    if dry_run:
        (raw_dir / "dry_run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        _print_plan(config, plan)
        print(f"\nDRY RUN -- no API calls were made. Plan written to {raw_dir / 'dry_run_plan.json'}")
        return 0

    tracker = CostTracker(
        max_usd=config.budget.max_usd,
        max_total_tokens=config.budget.max_total_tokens,
        prices={m.key: m.pricing for m in config.models},
        stop_on_unknown_pricing=config.budget.stop_on_unknown_pricing,
    )
    tracker.validate_models([m.key for m in config.models])

    done_by_cell: Dict[str, set] = {}
    pending: List[Task] = []
    for task in tasks:
        done = done_by_cell.get(task.cell_id)
        if done is None:
            done = completed_example_ids(raw_dir, task.cell_id)
            done_by_cell[task.cell_id] = done
        if task.example_id not in done:
            pending.append(task)

    _print_plan(config, plan)
    print(f"\nResume: {len(tasks) - len(pending)} call(s) already checkpointed, {len(pending)} remaining.")
    if not pending:
        print("Nothing to do -- this run is complete.")
        return 0

    providers: Dict[str, BaseProvider] = {}
    try:
        for model in config.models:
            providers[model.key] = build_provider(
                model.provider,
                model.model,
                rpm=model.rpm,
                timeout_s=config.timeout_s,
                retry=RetryPolicy(max_attempts=config.max_attempts),
                extra_headers=model.extra_headers,
            )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        for provider in providers.values():
            await provider.aclose()
        print(f"Failed to construct providers: {exc}", file=sys.stderr)
        return 1

    manifest = {
        "run_id": config.run_id,
        "started_utc": _utcnow(),
        "litm2026_version": __version__,
        "config": config.to_dict(),
        "plan": {k: v for k, v in plan.items() if k != "cells"},
        "cells": plan["cells"],
        "note": (
            "Accuracy numbers are NOT stored here. Run "
            "`python -m litm2026.stats --run-dir <this dir>` to score the raw responses."
        ),
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    writer = CheckpointWriter(raw_dir)
    stop = asyncio.Event()
    counters = {"calls": 0, "errors": 0}
    queue: "asyncio.Queue" = asyncio.Queue()
    for task in pending:
        queue.put_nowait(task)
    for _ in range(config.concurrency):
        queue.put_nowait(None)

    workers = [
        asyncio.create_task(
            _worker(f"w{i}", queue, providers, config, tracker, writer, stop, counters),
            name=f"litm-worker-{i}",
        )
        for i in range(config.concurrency)
    ]
    try:
        await asyncio.gather(*workers)
    finally:
        writer.close()
        for provider in providers.values():
            await provider.aclose()

    summary = tracker.summary()
    summary.update(
        {
            "run_id": config.run_id,
            "finished_utc": _utcnow(),
            "calls_completed_this_session": counters["calls"],
            "errors_this_session": counters["errors"],
            "stopped_early": stop.is_set(),
        }
    )
    (raw_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- run summary ---")
    print(json.dumps(summary, indent=2))
    print(f"\nRaw responses: {raw_dir}")
    print(f"Next: python -m litm2026.stats --run-dir {raw_dir} --out results/summary.csv")
    if stop.is_set():
        print(
            "\nThe run stopped early (budget cap or call cap).\n"
            "Completed work is checkpointed. Raise the cap and rerun the SAME config to resume.",
            file=sys.stderr,
        )
        return 2
    if counters["errors"]:
        print(f"\n{counters['errors']} call(s) failed and were recorded with an `error` field.", file=sys.stderr)
    return 0


def _print_plan(config: RunConfig, plan: Mapping[str, Any]) -> None:
    """Print the experiment matrix as a table, with call counts and token estimates."""
    print(
        f"\nRun: {config.run_id}   seed={config.seed}   n={config.n}   "
        f"chat_adaptation={config.chat_adaptation.value}"
    )
    print(f"Models: {', '.join(m.key + ' (' + m.provider + ')' for m in config.models)}")
    subset = plan.get("question_subset")
    if subset:
        print(f"Question subset: n={subset['n']} of {subset['universe_size']}, signature={subset['signature']}")
        print("  (the SAME questions are used in every QA cell -- this is what makes the tests paired)")
    header = f"{'cell_id':<58} {'calls':>6} {'~max prompt tok':>16}"
    print("\n" + header)
    print("-" * len(header))
    for cell in plan["cells"]:
        print(f"{cell['cell_id']:<58} {cell['n_calls']:>6} {cell['estimated_prompt_tokens_max_APPROX']:>16,}")
    print("-" * len(header))
    print(f"{'TOTAL':<58} {plan['n_calls']:>6} {plan['estimated_prompt_tokens_TOTAL_APPROX']:>16,}")
    print(f"\nToken figures are ESTIMATES ({plan['estimate_method']}).")
    budget = config.budget
    print(
        f"Budget: max_usd={budget.max_usd} max_total_tokens={budget.max_total_tokens} "
        f"max_calls={budget.max_calls} stop_on_unknown_pricing={budget.stop_on_unknown_pricing}"
    )
    unpriced = [m.key for m in config.models if not m.pricing.is_known]
    if unpriced:
        print(f"WARNING: no pricing configured for {unpriced} -- dollar spend cannot be computed for these.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. See the module docstring for usage."""
    parser = argparse.ArgumentParser(
        description="Run one Lost-in-the-Middle replication config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=pathlib.Path, help="experiments/configs/*.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build every prompt and print the matrix. Makes zero API calls. Do this first, every time.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    try:
        return asyncio.run(run(config, dry_run=args.dry_run))
    except (UnknownPricingError, BudgetExceeded) as exc:
        print(f"\n*** REFUSING TO RUN ***\n{exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted. Completed calls are checkpointed; rerun the same config to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
