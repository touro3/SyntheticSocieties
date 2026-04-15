from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from collections import OrderedDict
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
        timeout: float = 30.0,
        max_retries: int = 2,
        min_delay: float = 0.2,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # Prefer explicit argument; fall back to environment variable.
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_delay = min_delay

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
            raise ValueError("OpenAI API key not set. Pass api_key= or set the OPENAI_API_KEY environment variable.")

        self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)

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
                sleep_s = (2**attempt) + random.random()
                logger.warning(
                    "OpenAI generate() attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)

        raise last_error  # type: ignore[misc]

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
