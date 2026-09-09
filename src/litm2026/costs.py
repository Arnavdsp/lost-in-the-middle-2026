"""Token accounting and the hard spend cap.

DESIGN NOTE — WHY THERE IS NO BUILT-IN PRICE TABLE
--------------------------------------------------
It would be easy to ship a dict of per-token prices for a dozen providers. It would
also be **wrong**: prices change, they differ by region and tier, and a stale number
baked into a repo silently produces a fake "total spend" figure in a README. This
project's rule is that no number appears anywhere unless it can be sourced.

So pricing is *configuration*, not code. Each model block in an
``experiments/configs/*.yaml`` carries::

    pricing:
      input_per_mtok: 0.0
      output_per_mtok: 0.0
      source: "Groq free tier, no per-token billing -- console.groq.com, checked 2026-09-05"

If ``pricing`` is omitted, the model's cost is **unknown**, not zero. A run with
unknown pricing and a dollar cap is refused up front (``stop_on_unknown_pricing:
true``, the default), because a dollar cap you cannot evaluate is not a cap. Set
``stop_on_unknown_pricing: false`` only together with ``max_total_tokens``, which
gives you a cap in a currency the tracker can actually count.

For genuinely free tiers, writing ``0.0`` with a ``source`` line is accurate and is
the intended setup for the Groq/Mistral free-tier paths.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "TokenPrice",
    "UnknownPricingError",
    "BudgetExceeded",
    "CallCost",
    "CostTracker",
    "estimate_tokens_from_chars",
]

#: Crude characters-per-token ratio used ONLY for pre-run projections in
#: ``--dry-run``. English prose sits near 4 chars/token for most BPE tokenizers;
#: the key-value task (UUID soup) is far denser, so projections there run low.
#: Never used to produce a reported result -- reported token counts always come
#: from the provider's own usage fields.
CHARS_PER_TOKEN_ESTIMATE = 4.0


class UnknownPricingError(RuntimeError):
    """Raised when a dollar cap is requested for a model with no configured price."""


class BudgetExceeded(RuntimeError):
    """Raised when a run would exceed its configured cap. The runner stops hard."""


@dataclass(frozen=True)
class TokenPrice:
    """Per-million-token prices for one model, with a mandatory provenance string.

    Attributes:
        input_per_mtok: USD per 1,000,000 input tokens. ``None`` means unknown.
        output_per_mtok: USD per 1,000,000 output tokens. ``None`` means unknown.
        source: Where the number came from and when you checked. Required for
            non-``None`` prices so a reader can audit the spend figure.
    """

    input_per_mtok: Optional[float] = None
    output_per_mtok: Optional[float] = None
    source: str = ""

    def __post_init__(self) -> None:
        if self.is_known and not self.source:
            raise ValueError(
                "TokenPrice with concrete numbers must carry a `source` "
                "(where you read the price, and the date you read it)."
            )

    @property
    def is_known(self) -> bool:
        """True when both input and output prices are set."""
        return self.input_per_mtok is not None and self.output_per_mtok is not None

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Cost of one call in USD.

        Raises:
            UnknownPricingError: If either price is unset.
        """
        if not self.is_known:
            raise UnknownPricingError(
                "Cannot price this call: model pricing is not configured. "
                "Add a `pricing:` block to the model in your config, or set "
                "`budget.stop_on_unknown_pricing: false` with a `max_total_tokens` cap."
            )
        assert self.input_per_mtok is not None and self.output_per_mtok is not None
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1_000_000.0

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "TokenPrice":
        """Build from a config ``pricing`` block (or ``None`` for unknown)."""
        if not data:
            return cls()
        return cls(
            input_per_mtok=_opt_float(data.get("input_per_mtok")),
            output_per_mtok=_opt_float(data.get("output_per_mtok")),
            source=str(data.get("source", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form for the run manifest."""
        return {
            "input_per_mtok": self.input_per_mtok,
            "output_per_mtok": self.output_per_mtok,
            "source": self.source,
            "is_known": self.is_known,
        }


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


@dataclass(frozen=True)
class CallCost:
    """The accounting record for a single completed API call."""

    model_key: str
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostTracker:
    """Thread-safe running total with a hard cap, checked before and after each call.

    The cap is enforced in two places, deliberately:

    * :meth:`check_before_call` refuses to start a call once the cap is reached, so
      the overshoot is bounded by whatever is already in flight.
    * :meth:`record` raises after the total crosses the cap, so an unexpectedly
      expensive response still halts the run.

    Attributes:
        max_usd: Hard dollar cap. ``None`` disables the dollar cap (only sensible
            when every model is genuinely free and priced at 0.0, or when
            ``max_total_tokens`` is set instead).
        max_total_tokens: Optional hard cap on input+output tokens across the run.
        prices: Per-model-key prices.
        stop_on_unknown_pricing: If ``True`` and a dollar cap is set, a call for a
            model with unknown pricing raises :class:`UnknownPricingError` rather
            than being counted as free.
    """

    max_usd: Optional[float] = None
    max_total_tokens: Optional[int] = None
    prices: Dict[str, TokenPrice] = field(default_factory=dict)
    stop_on_unknown_pricing: bool = True

    total_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    n_calls: int = 0
    n_unpriced_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- validation ------------------------------------------------------- #
    def validate_models(self, model_keys: Any) -> None:
        """Fail fast if a dollar cap cannot be enforced for some model.

        Called once at run start, before any request. This is the check that turns
        "I forgot to configure pricing" from a silently-uncapped run into a
        one-second error.

        Raises:
            UnknownPricingError: If a dollar cap is set, ``stop_on_unknown_pricing``
                is on, and some model has no usable price.
        """
        if self.max_usd is None or not self.stop_on_unknown_pricing:
            return
        unknown = [k for k in model_keys if not self.prices.get(k, TokenPrice()).is_known]
        if unknown:
            raise UnknownPricingError(
                "A dollar cap (budget.max_usd) is set but these models have no pricing "
                f"configured: {unknown}. Add a `pricing:` block with a `source:` to each "
                "model (use 0.0 for a genuinely free tier), or set "
                "`budget.stop_on_unknown_pricing: false` together with `budget.max_total_tokens`."
            )

    # -- enforcement ------------------------------------------------------ #
    def check_before_call(self) -> None:
        """Raise :class:`BudgetExceeded` if the cap has already been reached."""
        with self._lock:
            self._raise_if_over(prefix="Refusing to start another call")

    def record(self, model_key: str, input_tokens: int, output_tokens: int) -> CallCost:
        """Account one completed call and enforce the cap.

        Args:
            model_key: Config key of the model that produced the call.
            input_tokens: Prompt tokens **as reported by the provider**.
            output_tokens: Completion tokens as reported by the provider.

        Returns:
            The :class:`CallCost` record (``cost_usd`` is ``None`` when unpriced).

        Raises:
            UnknownPricingError: Unknown price while a dollar cap is being enforced.
            BudgetExceeded: If this call pushes the run over a cap.
        """
        price = self.prices.get(model_key, TokenPrice())
        cost: Optional[float]
        if price.is_known:
            cost = price.cost_usd(input_tokens, output_tokens)
        else:
            if self.max_usd is not None and self.stop_on_unknown_pricing:
                raise UnknownPricingError(
                    f"No pricing configured for model {model_key!r} while a dollar cap is active."
                )
            cost = None

        with self._lock:
            self.n_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            if cost is None:
                self.n_unpriced_calls += 1
            else:
                self.total_usd += cost
            self._raise_if_over(prefix="Budget cap reached")

        return CallCost(model_key=model_key, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)

    def _raise_if_over(self, prefix: str) -> None:
        """Caller must hold ``_lock``."""
        if self.max_usd is not None and self.total_usd >= self.max_usd:
            raise BudgetExceeded(
                f"{prefix}: spent ${self.total_usd:.4f} of ${self.max_usd:.2f} cap "
                f"across {self.n_calls} calls. Stopping. Completed work is already "
                f"checkpointed in results/raw/ -- raise budget.max_usd and rerun the "
                f"same config to resume exactly where this stopped."
            )
        if self.max_total_tokens is not None and self.total_tokens >= self.max_total_tokens:
            raise BudgetExceeded(
                f"{prefix}: used {self.total_tokens:,} of {self.max_total_tokens:,} token cap "
                f"across {self.n_calls} calls. Stopping. Completed work is already "
                f"checkpointed in results/raw/ -- raise budget.max_total_tokens and rerun to resume."
            )

    # -- reporting -------------------------------------------------------- #
    @property
    def total_tokens(self) -> int:
        """Input + output tokens billed so far."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def remaining_usd(self) -> Optional[float]:
        """Dollars left under the cap, or ``None`` when no dollar cap is set."""
        return None if self.max_usd is None else max(0.0, self.max_usd - self.total_usd)

    def summary(self) -> Dict[str, Any]:
        """JSON-serializable spend summary for the run manifest and the README."""
        return {
            "n_calls": self.n_calls,
            "n_unpriced_calls": self.n_unpriced_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_usd": round(self.total_usd, 6),
            "max_usd": self.max_usd,
            "max_total_tokens": self.max_total_tokens,
            "pricing_complete": self.n_unpriced_calls == 0,
        }


def estimate_tokens_from_chars(text: str) -> int:
    """Rough token count for pre-run projection only.

    Divides character length by :data:`CHARS_PER_TOKEN_ESTIMATE`. This is an
    estimate and is labelled as such everywhere it surfaces (``--dry-run`` output,
    ``dry_run_plan.json``). It never enters a results table: reported token counts
    come from the provider's ``usage`` fields.

    Args:
        text: Prompt text.

    Returns:
        Estimated token count (at least 1 for non-empty text).
    """
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))
