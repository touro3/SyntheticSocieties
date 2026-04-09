from __future__ import annotations

import hashlib
import json
import os
import random
import time
from typing import Optional


class OpenAIBackend:
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
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_delay = min_delay

        self._client = None
        self._cache = {}

    def load(self) -> None:
        from openai import OpenAI

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)

    def _hash_messages(self, messages):
        return hashlib.md5(json.dumps(messages, sort_keys=True).encode()).hexdigest()

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

        key = self._hash_messages(messages)

        # ✅ CACHE — return cached (text, latency) directly
        if key in self._cache:
            return self._cache[key]

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                time.sleep(self.min_delay)  # throttle

                start = time.time()

                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                )

                text = resp.choices[0].message.content.strip()
                latency = time.time() - start

                self._cache[key] = (text, latency)
                return text, latency

            except Exception as e:
                last_error = e

                if attempt == self.max_retries:
                    raise e

                sleep = (2 ** attempt) + random.random()
                time.sleep(sleep)

        raise last_error