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

_CACHE_MAX_SIZE = 1_024


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split OpenAI-style messages into (system_prompt, conversation).

    Anthropic's Messages API takes the system prompt as a separate ``system=``
    argument; only user/assistant turns go in ``messages=``.
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    convo = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") in ("user", "assistant")]
    if not convo:  # Anthropic requires at least one user turn
        convo = [{"role": "user", "content": " ".join(system_parts) or ""}]
        system_parts = []
    return "\n\n".join(system_parts), convo


class AnthropicBackend:
    """Anthropic Messages API backend with LRU caching and retry/backoff.

    Mirrors :class:`decision.openai_backend.OpenAIBackend` so it is a drop-in
    under ``BatchLLMBackendProtocol``. The ``anthropic`` SDK is imported lazily
    in :meth:`load` so this module imports without the dependency installed
    (dry runs / tests do not need it).
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-6",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 8,
        min_delay: float = 0.2,
        max_batch_size: int = 4,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_delay = min_delay
        self._max_batch_size = int(os.environ.get("BGF_ANTHROPIC_BATCH_SIZE", max_batch_size))

        self._client = None
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cached_hits = 0

    def load(self) -> None:
        import anthropic

        if not self.api_key:
            raise ValueError("API key not set. Pass api_key= or set ANTHROPIC_API_KEY.")
        self._client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)

    # ── LRU cache ─────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(messages: list[dict]) -> str:
        return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[tuple[str, float]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: tuple[str, float]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > _CACHE_MAX_SIZE:
            self._cache.popitem(last=False)

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

        system, convo = _split_system(messages)
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            retry_temp = max(0.1, temp - (attempt * 0.1))
            try:
                time.sleep(self.min_delay)
                start = time.time()
                kwargs: dict = {
                    "model": self.model_id,
                    "max_tokens": max_tok,
                    "temperature": retry_temp,
                    "messages": convo,
                }
                if system:
                    kwargs["system"] = system
                resp = self._client.messages.create(**kwargs)
                text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text").strip()
                latency = time.time() - start

                if getattr(resp, "usage", None) is not None:
                    self._total_prompt_tokens += resp.usage.input_tokens
                    self._total_completion_tokens += resp.usage.output_tokens

                self._cache_put(key, (text, latency))
                return text, latency

            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                exc_str = str(exc)
                if "429" in exc_str or "rate_limit" in exc_str.lower() or "overloaded" in exc_str.lower():
                    wait_s = 5.0 + random.random() * 5.0
                else:
                    wait_s = (2**attempt) + random.random()
                logger.warning(
                    "Anthropic generate() attempt %d/%d failed (%s); retrying in %.1fs",
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
        if not messages_list:
            return []
        if self._client is None:
            self.load()

        results: list[Optional[tuple[str, float]]] = [None] * len(messages_list)

        def _one(idx: int) -> None:
            results[idx] = self.generate(messages_list[idx], temperature=temperature, max_new_tokens=max_new_tokens)

        with ThreadPoolExecutor(max_workers=max_batch_size) as ex:
            list(ex.map(_one, range(len(messages_list))))
        return [r for r in results if r is not None]

    def usage_report(self, model_id: Optional[str] = None) -> dict:
        # Claude 3.5 Sonnet: $3.00 / $15.00 per 1M tokens (prompt/completion).
        _COST_PER_1M = {
            "claude-sonnet-4-6": (3.00, 15.00),
            "claude-3-5-sonnet-20241022": (3.00, 15.00),
            "claude-haiku-4-5-20251001": (0.80, 4.00),
        }
        mid = (model_id or self.model_id).lower()
        prompt_rate, completion_rate = _COST_PER_1M.get(mid, (3.00, 15.00))
        est = self._total_prompt_tokens / 1e6 * prompt_rate + self._total_completion_tokens / 1e6 * completion_rate
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "cached_hits": self._total_cached_hits,
            "estimated_cost_usd": round(est, 4),
        }
