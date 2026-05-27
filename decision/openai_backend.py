from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum number of prompt→response pairs to keep in the in-process cache.
# Each entry is O(prompt_tokens + response_tokens) in string size, so 1 024
# entries costs a few MB at most, while preventing unbounded growth in long
# multi-seed runs.
_CACHE_MAX_SIZE = 1_024


class OpenAIBackend:
    """OpenAI chat-completions backend with LRU caching and retry/backoff."""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 8,
        min_delay: float = 0.2,
        max_batch_size: int = 4,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.base_url = base_url  # None → OpenAI default; set for Ollama/Groq/etc.
        # Prefer explicit argument; fall back to environment variable.
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_delay = min_delay
        # Consumed by simulation.kernel via getattr(backend, "_max_batch_size", 16).
        # Default 4 stays well under OpenAI Tier-1 200K TPM for N≤200.
        self._max_batch_size = int(os.environ.get("BGF_OPENAI_BATCH_SIZE", max_batch_size))

        self._client = None
        # Bounded LRU cache: OrderedDict with move-to-end on hit, pop-from-front on overflow.
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        # Cumulative token usage counters (for cost tracking across a run).
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cached_hits: int = 0

    def load(self) -> None:
        from openai import OpenAI

        if not self.api_key:
            raise ValueError(
                "API key not set. Pass api_key= or set OPENAI_API_KEY / GROQ_API_KEY. "
                "For Ollama (local, no key needed) use provider='ollama'."
            )

        kwargs: dict = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    # ── LRU cache helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(messages: list[dict]) -> str:
        """Stable SHA-256 key for a messages list (no collision risk unlike MD5)."""
        return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[tuple[str, float]]:
        if key in self._cache:
            self._cache.move_to_end(key)  # mark as recently used
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: tuple[str, float]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > _CACHE_MAX_SIZE:
            evicted_key, _ = self._cache.popitem(last=False)  # evict oldest
            logger.debug("OpenAIBackend cache evicted oldest entry (%s)", evicted_key[:12])

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[str, float]:

        if self._client is None:
            self.load()

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        key = self._cache_key(messages)
        cached = self._cache_get(key)
        if cached is not None:
            self._total_cached_hits += 1
            return cached

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            retry_temp = max(0.1, temp - (attempt * 0.1))

            try:
                time.sleep(self.min_delay)

                start = time.time()
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=retry_temp,
                    max_tokens=max_tok,
                )
                text = resp.choices[0].message.content.strip()
                latency = time.time() - start

                # Track cumulative token usage for cost monitoring.
                if resp.usage is not None:
                    self._total_prompt_tokens += resp.usage.prompt_tokens
                    self._total_completion_tokens += resp.usage.completion_tokens

                self._cache_put(key, (text, latency))
                return text, latency

            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break

                # 429 rate-limit: honor the server's retry-after hint.
                exc_str = str(exc)
                wait_s: Optional[float] = None
                if "429" in exc_str or "rate_limit" in exc_str.lower():
                    import re

                    m = re.search(r"try again in ([\d.]+)\s*(ms|s|m)", exc_str)
                    if m:
                        val, unit = float(m.group(1)), m.group(2)
                        wait_s = {"ms": val / 1000.0, "s": val, "m": val * 60.0}[unit]
                        wait_s += 0.5  # margin
                    else:
                        wait_s = 5.0 + random.random() * 5.0

                if wait_s is None:
                    wait_s = (2**attempt) + random.random()

                logger.warning(
                    "OpenAI generate() attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    exc_str[:120],
                    wait_s,
                )
                time.sleep(wait_s)

        raise last_error  # type: ignore[misc]

    def generate_batch(
        self,
        messages_list: list[list[dict]],
        max_batch_size: int = 32,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """Concurrent batched generation via ThreadPoolExecutor.

        Issues up to ``max_batch_size`` OpenAI chat-completion calls in
        parallel.  Required by ``simulation.kernel`` to take the batched
        execution path (otherwise it falls back to serial per-agent calls,
        which at N=300 agents × 30 rounds is ~10h per cell).

        Returns:
            List of (text, latency) tuples in the same order as ``messages_list``.
        """
        if not messages_list:
            return []
        if self._client is None:
            self.load()

        results: list[Optional[tuple[str, float]]] = [None] * len(messages_list)

        def _one(idx: int) -> None:
            results[idx] = self.generate(
                messages_list[idx],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        with ThreadPoolExecutor(max_workers=max_batch_size) as ex:
            list(ex.map(_one, range(len(messages_list))))

        # mypy: all slots now filled
        return [r for r in results if r is not None]

    def usage_report(self, model_id: Optional[str] = None) -> dict:
        """Return cumulative token usage and estimated cost.

        Cost rates (as of 2025): gpt-4o-mini $0.15/$0.60 per 1M tokens
        (prompt/completion).  Override with ``model_id`` for other models.
        """
        _COST_PER_1M = {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
            "gpt-3.5-turbo": (0.50, 1.50),
        }
        mid = (model_id or self.model_id).lower()
        prompt_rate, completion_rate = _COST_PER_1M.get(mid, (0.0, 0.0))
        est_cost = self._total_prompt_tokens / 1e6 * prompt_rate + self._total_completion_tokens / 1e6 * completion_rate
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "cached_hits": self._total_cached_hits,
            "estimated_cost_usd": round(est_cost, 4),
        }
