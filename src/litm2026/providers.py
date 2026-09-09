"""Thin async provider abstraction: Anthropic + any OpenAI-compatible endpoint.

Everything here exists to serve one requirement: run the same prompt against
several 2026 models, at temperature 0, on a free tier, without getting rate-limited
into a corrupted run.

SUPPORTED ENDPOINTS
-------------------
``AnthropicProvider`` wraps the official ``anthropic`` SDK (Messages API).
``OpenAICompatProvider`` is a ~100-line httpx client for the
``POST {base_url}/chat/completions`` shape, which Groq, Mistral, OpenRouter,
Together and Cerebras all speak. Adding a provider is one entry in
:data:`PROVIDER_SPECS`; no new code.

RATE LIMITS — WHAT THE NUMBERS IN :data:`PROVIDER_SPECS` MEAN
-------------------------------------------------------------
``default_rpm`` below is **this repository's conservative default**, not a claim
about any provider's published limit. Free tiers change often and differ per
account, per model and per region, so quoting them here would create exactly the
kind of stale, unsourceable number this project refuses to print. Each spec instead
carries ``limits_doc_url``: check it, then set ``rpm:`` explicitly in your config.

    provider    | env var              | default_rpm used here | authoritative limits
    ------------|----------------------|-----------------------|----------------------------------
    anthropic   | ANTHROPIC_API_KEY    | 20                    | docs.anthropic.com/en/api/rate-limits
    groq        | GROQ_API_KEY         | 12                    | console.groq.com/docs/rate-limits
    mistral     | MISTRAL_API_KEY      | 12                    | docs.mistral.ai/deployment/laplateforme/tier
    openrouter  | OPENROUTER_API_KEY   | 10                    | openrouter.ai/docs/api-reference/limits
    together    | TOGETHER_API_KEY     | 10                    | docs.together.ai/docs/rate-limits
    cerebras    | CEREBRAS_API_KEY     | 10                    | inference-docs.cerebras.ai/support/rate-limits

Every default is set low enough to be safe on a free tier without being checked.
Raise it once you have read your dashboard; the run gets faster and nothing else
changes. On a 429 the client honours ``Retry-After`` exactly and puts *all* workers
for that model into the same cooldown, so a burst of concurrent 429s does not turn
into a burst of retries.

DETERMINISM
-----------
All requests are sent with ``temperature=0`` and ``top_p=1`` by default, matching
the paper's greedy decoding. Hosted models are still not bit-deterministic (batching
and kernel non-determinism), which is recorded as a threat to validity in
REPLICATION.md rather than papered over.
"""

from __future__ import annotations

import abc
import asyncio
import email.utils
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import httpx

__all__ = [
    "ProviderSpec",
    "PROVIDER_SPECS",
    "LLMResponse",
    "ProviderError",
    "NonRetryableProviderError",
    "RateLimiter",
    "RetryPolicy",
    "parse_retry_after",
    "BaseProvider",
    "OpenAICompatProvider",
    "AnthropicProvider",
    "MockProvider",
    "build_provider",
]


@dataclass(frozen=True)
class ProviderSpec:
    """Static description of a provider endpoint.

    Attributes:
        name: Config key, e.g. ``"groq"``.
        kind: ``"openai_compatible"`` or ``"anthropic"``.
        base_url: Chat-completions base URL (unused for ``anthropic``).
        api_key_env: Environment variable holding the key.
        default_rpm: This repo's conservative requests-per-minute default. Override
            per model in the config once you have checked ``limits_doc_url``.
        limits_doc_url: Where the authoritative rate limits are published.
    """

    name: str
    kind: str
    base_url: Optional[str]
    api_key_env: str
    default_rpm: int
    limits_doc_url: str


PROVIDER_SPECS: Dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        name="anthropic",
        kind="anthropic",
        base_url=None,
        api_key_env="ANTHROPIC_API_KEY",
        default_rpm=20,
        limits_doc_url="https://docs.anthropic.com/en/api/rate-limits",
    ),
    "groq": ProviderSpec(
        name="groq",
        kind="openai_compatible",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_rpm=12,
        limits_doc_url="https://console.groq.com/docs/rate-limits",
    ),
    "mistral": ProviderSpec(
        name="mistral",
        kind="openai_compatible",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        default_rpm=12,
        limits_doc_url="https://docs.mistral.ai/deployment/laplateforme/tier/",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        kind="openai_compatible",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_rpm=10,
        limits_doc_url="https://openrouter.ai/docs/api-reference/limits",
    ),
    "together": ProviderSpec(
        name="together",
        kind="openai_compatible",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        default_rpm=10,
        limits_doc_url="https://docs.together.ai/docs/rate-limits",
    ),
    "cerebras": ProviderSpec(
        name="cerebras",
        kind="openai_compatible",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        default_rpm=10,
        limits_doc_url="https://inference-docs.cerebras.ai/support/rate-limits",
    ),
    "mock": ProviderSpec(
        name="mock",
        kind="mock",
        base_url=None,
        api_key_env="",
        default_rpm=100000,
        limits_doc_url="",
    ),
}


class ProviderError(RuntimeError):
    """A call failed after exhausting retries."""


class NonRetryableProviderError(ProviderError):
    """A call failed in a way retrying cannot fix (bad key, bad model, 400)."""


@dataclass
class LLMResponse:
    """One model response plus the accounting the runner needs.

    Attributes:
        text: Generated text (empty string if the model returned nothing).
        input_tokens: Prompt tokens as reported by the provider. ``None`` if the
            provider did not report usage -- never estimated.
        output_tokens: Completion tokens as reported by the provider.
        model: Model id echoed by the provider, when available.
        finish_reason: Provider's stop reason, useful for spotting truncation.
        latency_s: Wall-clock seconds for the successful attempt.
        attempts: How many HTTP attempts it took (1 = first try succeeded).
        raw: The provider's decoded JSON response, stored verbatim in results/raw/.
        is_mock: True only for :class:`MockProvider`. Propagated into every raw
            record so analysis can refuse to summarise fake data.
    """

    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    latency_s: float = 0.0
    attempts: int = 1
    raw: Optional[Dict[str, Any]] = None
    is_mock: bool = False


class RateLimiter:
    """Async requests-per-minute limiter with a shared cooldown for 429s.

    Two mechanisms, both needed:

    * A minimum interval between request starts (``60 / rpm``), which keeps a
      steady-state run inside the quota.
    * A shared ``cooldown_until`` timestamp. When any worker sees a 429 it calls
      :meth:`penalise`, and every other worker on that model waits too. Without
      this, N concurrent workers each retry independently and multiply the 429s.

    Args:
        rpm: Requests per minute. Must be > 0.
    """

    def __init__(self, rpm: float) -> None:
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        self.rpm = float(rpm)
        self._min_interval = 60.0 / float(rpm)
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0
        self._cooldown_until = 0.0

    async def acquire(self) -> None:
        """Block until this worker may start a request."""
        while True:
            async with self._lock:
                now = time.monotonic()
                start_at = max(now, self._next_allowed, self._cooldown_until)
                wait = start_at - now
                if wait <= 0:
                    self._next_allowed = now + self._min_interval
                    return
                # Reserve our slot before sleeping so concurrent callers queue up.
                self._next_allowed = start_at + self._min_interval
            await asyncio.sleep(wait)
            async with self._lock:
                if time.monotonic() >= self._cooldown_until:
                    return
            # A cooldown started while we slept; loop and wait it out.

    async def penalise(self, seconds: float) -> None:
        """Put every worker on this model into a cooldown of ``seconds``."""
        async with self._lock:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + max(0.0, seconds))


def parse_retry_after(value: Optional[str], *, now: Optional[float] = None) -> Optional[float]:
    """Parse a ``Retry-After`` header into seconds.

    Handles both forms in RFC 9110: delta-seconds (``"12"``, ``"1.5"``) and an
    HTTP-date (``"Wed, 21 Oct 2026 07:28:00 GMT"``).

    Args:
        value: Raw header value, or ``None``.
        now: Unix timestamp to measure an HTTP-date against (defaults to
            ``time.time()``); injectable so the behaviour is testable.

    Returns:
        Non-negative seconds to wait, or ``None`` if the header is absent/unparsable.
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    reference = time.time() if now is None else now
    return max(0.0, when.timestamp() - reference)


@dataclass
class RetryPolicy:
    """Backoff configuration for transient failures.

    Attributes:
        max_attempts: Total attempts per call, including the first.
        base_delay_s: First backoff delay; doubles each attempt.
        max_delay_s: Ceiling on a single backoff delay.
        jitter: Multiplicative jitter fraction, to avoid synchronised retries.
    """

    max_attempts: int = 6
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0
    jitter: float = 0.25

    def delay_for(self, attempt: int, rng: Optional[random.Random] = None) -> float:
        """Backoff delay before attempt number ``attempt`` (1-based)."""
        rng = rng or random
        raw = min(self.max_delay_s, self.base_delay_s * (2 ** max(0, attempt - 1)))
        return raw * (1.0 + rng.uniform(-self.jitter, self.jitter))


class BaseProvider(abc.ABC):
    """Common interface: rate-limited, retrying, temperature-0 chat completion."""

    def __init__(
        self,
        *,
        model: str,
        rate_limiter: RateLimiter,
        retry: Optional[RetryPolicy] = None,
        timeout_s: float = 180.0,
    ) -> None:
        self.model = model
        self.rate_limiter = rate_limiter
        self.retry = retry or RetryPolicy()
        self.timeout_s = timeout_s

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Config key of the provider ("groq", "anthropic", ...)."""

    @abc.abstractmethod
    async def _attempt(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> LLMResponse:
        """Make one HTTP attempt. Raise :class:`_Retryable` for transient failures."""

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> LLMResponse:
        """Generate one completion, honouring the rate limit and retry policy.

        Args:
            messages: Chat messages from :func:`litm2026.prompting.to_chat_messages`.
            max_tokens: Max new tokens. The paper used 100; keep it there.
            temperature: 0.0 for greedy decoding, matching the paper.
            top_p: 1.0, matching the paper.

        Returns:
            An :class:`LLMResponse`.

        Raises:
            NonRetryableProviderError: Auth/model/validation failures.
            ProviderError: Transient failures that survived every retry.
        """
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.retry.max_attempts + 1):
            await self.rate_limiter.acquire()
            started = time.monotonic()
            try:
                response = await self._attempt(
                    messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p
                )
                response.latency_s = time.monotonic() - started
                response.attempts = attempt
                return response
            except _Retryable as exc:
                last_error = exc.__cause__ or exc
                if exc.retry_after is not None:
                    await self.rate_limiter.penalise(exc.retry_after)
                    delay = exc.retry_after
                else:
                    delay = self.retry.delay_for(attempt)
                if attempt >= self.retry.max_attempts:
                    break
                await asyncio.sleep(max(0.0, delay))
            except NonRetryableProviderError:
                raise
        raise ProviderError(
            f"{self.provider_name}/{self.model}: failed after {self.retry.max_attempts} attempts: {last_error}"
        ) from last_error

    async def aclose(self) -> None:
        """Release any network resources. Safe to call more than once."""


class _Retryable(Exception):
    """Internal marker for a transient failure, carrying any ``Retry-After``."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class OpenAICompatProvider(BaseProvider):
    """Client for any ``POST {base_url}/chat/completions`` endpoint.

    Verified against the request/response shape used by Groq, Mistral, OpenRouter,
    Together and Cerebras. Providers differ only in ``base_url`` and model naming,
    both of which come from config.

    Args:
        spec: The provider spec (base URL, key env var).
        model: Provider-side model id, e.g. ``"llama-3.3-70b-versatile"``.
        api_key: Key; read from ``spec.api_key_env`` when omitted.
        rate_limiter: Shared limiter for this model.
        extra_headers: Additional headers (OpenRouter likes ``HTTP-Referer``).
    """

    def __init__(
        self,
        *,
        spec: ProviderSpec,
        model: str,
        rate_limiter: RateLimiter,
        api_key: Optional[str] = None,
        retry: Optional[RetryPolicy] = None,
        timeout_s: float = 180.0,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(model=model, rate_limiter=rate_limiter, retry=retry, timeout_s=timeout_s)
        if spec.base_url is None:
            raise ValueError(f"Provider {spec.name} has no base_url")
        self.spec = spec
        key = api_key if api_key is not None else os.environ.get(spec.api_key_env, "")
        if not key:
            raise NonRetryableProviderError(
                f"No API key for provider {spec.name!r}: set {spec.api_key_env} "
                "(see .env.example)."
            )
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        headers.update(dict(extra_headers or {}))
        self._client = httpx.AsyncClient(
            base_url=spec.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s),
        )

    @property
    def provider_name(self) -> str:
        return self.spec.name

    async def _attempt(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise _Retryable(f"transport error: {exc}") from exc

        if response.status_code == 429:
            raise _Retryable(
                "429 rate limited",
                retry_after=parse_retry_after(response.headers.get("Retry-After")),
            )
        if response.status_code in (408, 409, 500, 502, 503, 504, 529):
            raise _Retryable(f"{response.status_code}: {response.text[:400]}")
        if response.status_code >= 400:
            raise NonRetryableProviderError(
                f"{self.spec.name}/{self.model}: HTTP {response.status_code}: {response.text[:800]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise _Retryable(f"non-JSON response: {response.text[:200]}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise _Retryable(f"no choices in response: {str(body)[:400]}")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        usage = body.get("usage") or {}
        return LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            model=body.get("model"),
            finish_reason=choices[0].get("finish_reason"),
            raw=body,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicProvider(BaseProvider):
    """Client for the Anthropic Messages API via the official SDK.

    A leading ``system`` message (produced by the ``system_instruction`` or
    ``user_verbatim_terse`` chat adaptations) is hoisted into the API's dedicated
    ``system`` parameter, because the Messages API does not accept a system role
    inside ``messages``.

    Args:
        model: Anthropic model id.
        rate_limiter: Shared limiter for this model.
        api_key: Key; read from ``ANTHROPIC_API_KEY`` when omitted.
    """

    def __init__(
        self,
        *,
        model: str,
        rate_limiter: RateLimiter,
        api_key: Optional[str] = None,
        retry: Optional[RetryPolicy] = None,
        timeout_s: float = 180.0,
    ) -> None:
        super().__init__(model=model, rate_limiter=rate_limiter, retry=retry, timeout_s=timeout_s)
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise NonRetryableProviderError("The `anthropic` package is required for provider 'anthropic'") from exc
        import anthropic as _anthropic

        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise NonRetryableProviderError("No API key: set ANTHROPIC_API_KEY (see .env.example).")
        self._sdk = _anthropic
        # max_retries=0: this class owns retry/backoff so that 429s feed the shared
        # RateLimiter cooldown instead of being swallowed inside the SDK.
        self._client = _anthropic.AsyncAnthropic(api_key=key, timeout=timeout_s, max_retries=0)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def _attempt(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> LLMResponse:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat = [dict(m) for m in messages if m.get("role") != "system"]
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        # top_p is omitted at temperature 0: the Anthropic API rejects setting both.
        if temperature > 0:
            kwargs["top_p"] = top_p

        try:
            message = await self._client.messages.create(**kwargs)
        except self._sdk.RateLimitError as exc:
            retry_after = None
            response = getattr(exc, "response", None)
            if response is not None:
                retry_after = parse_retry_after(response.headers.get("retry-after"))
            raise _Retryable("429 rate limited", retry_after=retry_after) from exc
        except (self._sdk.APITimeoutError, self._sdk.APIConnectionError) as exc:
            raise _Retryable(f"transport error: {exc}") from exc
        except self._sdk.APIStatusError as exc:
            status = getattr(exc, "status_code", 500)
            if status in (408, 409, 500, 502, 503, 504, 529):
                raise _Retryable(f"{status}: {exc}") from exc
            raise NonRetryableProviderError(f"anthropic/{self.model}: {exc}") from exc

        text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
        raw = message.model_dump() if hasattr(message, "model_dump") else None
        return LLMResponse(
            text=text,
            input_tokens=getattr(message.usage, "input_tokens", None),
            output_tokens=getattr(message.usage, "output_tokens", None),
            model=getattr(message, "model", self.model),
            finish_reason=getattr(message, "stop_reason", None),
            raw=raw,
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()


class MockProvider(BaseProvider):
    """Offline stub for smoke-testing the plumbing. **Never** a source of results.

    Returns a canned string and marks every response ``is_mock=True``. The runner
    stamps that flag onto every raw record, and :mod:`litm2026.stats` refuses to
    aggregate mock records unless explicitly asked. That is the mechanism that makes
    it impossible for a pipeline test to leak a fake accuracy number into the README.
    """

    def __init__(
        self,
        *,
        model: str = "mock-model",
        rate_limiter: Optional[RateLimiter] = None,
        response_text: str = "MOCK RESPONSE - NOT A REAL MODEL OUTPUT",
    ) -> None:
        super().__init__(model=model, rate_limiter=rate_limiter or RateLimiter(rpm=100000))
        self.response_text = response_text

    @property
    def provider_name(self) -> str:
        return "mock"

    async def _attempt(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> LLMResponse:
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        return LLMResponse(
            text=self.response_text,
            input_tokens=max(1, prompt_chars // 4),
            output_tokens=len(self.response_text) // 4,
            model=self.model,
            finish_reason="stop",
            raw={"mock": True},
            is_mock=True,
        )


def build_provider(
    provider: str,
    model: str,
    *,
    rpm: Optional[float] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 180.0,
    retry: Optional[RetryPolicy] = None,
    extra_headers: Optional[Mapping[str, str]] = None,
) -> BaseProvider:
    """Construct a provider client from config values.

    Args:
        provider: A key of :data:`PROVIDER_SPECS`.
        model: Provider-side model id.
        rpm: Requests per minute; falls back to the spec's conservative default.
        api_key: Overrides the environment variable (used by tests).
        timeout_s: Per-request timeout.
        retry: Backoff policy.
        extra_headers: Extra HTTP headers for OpenAI-compatible endpoints.

    Returns:
        A ready :class:`BaseProvider`.

    Raises:
        ValueError: Unknown provider name.
        NonRetryableProviderError: Missing API key.
    """
    if provider not in PROVIDER_SPECS:
        raise ValueError(f"Unknown provider {provider!r}; known: {sorted(PROVIDER_SPECS)}")
    spec = PROVIDER_SPECS[provider]
    limiter = RateLimiter(rpm=rpm if rpm is not None else spec.default_rpm)
    if spec.kind == "mock":
        return MockProvider(model=model, rate_limiter=limiter)
    if spec.kind == "anthropic":
        return AnthropicProvider(model=model, rate_limiter=limiter, api_key=api_key, retry=retry, timeout_s=timeout_s)
    return OpenAICompatProvider(
        spec=spec,
        model=model,
        rate_limiter=limiter,
        api_key=api_key,
        retry=retry,
        timeout_s=timeout_s,
        extra_headers=extra_headers,
    )
