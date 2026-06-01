"""Protocol definitions for LLM backend interfaces.

Provides static-analysis-time contracts for backend implementations,
catching signature mismatches before runtime.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackendProtocol(Protocol):
    """Minimal contract for single-item LLM generation."""

    def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> tuple[str, float]:
        """Generate text from chat messages.

        Returns (generated_text, latency_seconds).
        """
        ...


@runtime_checkable
class BatchLLMBackendProtocol(LLMBackendProtocol, Protocol):
    """Extended contract for backends that support batched generation."""

    def generate_batch(
        self,
        messages_list: list[list[dict]],
        temperature: float | None = None,
    ) -> list[tuple[str, float]]:
        """Batch-generate from multiple chat message sets.

        Returns list of (generated_text, latency_seconds) tuples.
        """
        ...
