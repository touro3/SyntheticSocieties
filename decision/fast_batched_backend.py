"""Deprecated — use decision.llm_backend.LLMBackend instead.

This module is retained only for backward compatibility with legacy scripts
(run_bad_apple.py, run_macro_shock.py, run_topology.py, run_phase_d_scaling.py).
It wraps LLMBackend and converts raw-string prompts to chat messages.
"""

import warnings

_DEPRECATION_MSG = (
    "FastBatchedBackend is deprecated. Use decision.llm_backend.LLMBackend "
    "with generate_batch() instead."
)

warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)

from decision.llm_backend import LLMBackend


class _FastBatchedBackend:
    """Deprecated wrapper around LLMBackend for legacy scripts.

    Legacy scripts pass raw text prompts; this wrapper converts them to
    chat-format messages before delegating to LLMBackend.generate_batch().
    """

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        temperature: float = 0.5,
        cache_dir: str | None = None,
    ):
        self._backend = LLMBackend(
            model_id=model_id,
            temperature=temperature,
            cache_dir=cache_dir,
        )

    def load(self) -> None:
        self._backend.load()

    def generate_batch(
        self, prompts: list[str], batch_size: int = 16
    ) -> list[str]:
        """Generate responses for a list of raw-text prompts.

        Converts each prompt to a single-turn chat message, delegates to
        LLMBackend.generate_batch(), and returns only the text (no latency).
        """
        messages_list = [
            [{"role": "user", "content": p}] for p in prompts
        ]
        results = self._backend.generate_batch(
            messages_list,
            max_batch_size=min(batch_size, 5),
        )
        return [text for text, _latency in results]


def __getattr__(name: str):
    """Expose deprecated symbols while warning on attribute import access."""
    if name == "FastBatchedBackend":
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return _FastBatchedBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FastBatchedBackend"]
