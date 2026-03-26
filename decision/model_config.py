"""ModelConfig dataclass and get_backend() factory.

Phase 16 — Multi-Model Generalizability Study.

Abstracts LLM backend instantiation so that experiments can swap models
by changing a config entry, not code. Supports HuggingFace models (local GPU)
and OpenAI API models (remote, no GPU required).

Usage:
    cfg = ModelConfig(model_id="mistralai/Mistral-7B-Instruct-v0.3")
    backend = get_backend(cfg)
    text, latency = backend.generate(messages)

    cfg = ModelConfig(model_id="gpt-4o", backend_type="openai")
    backend = get_backend(cfg)   # returns OpenAIBackend
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Specification for a single LLM backend.

    Attributes:
        model_id: HuggingFace repo ID or OpenAI model name (e.g. "gpt-4o").
        backend_type: "huggingface" for local GPU inference; "openai" for API.
        context_length: Maximum context window in tokens (informational).
        dtype: Torch dtype string for HuggingFace backends ("float16", "bfloat16").
        quantization: Optional quantization scheme ("4bit", "8bit", None).
            Only applies to HuggingFace backends.
        cache_dir: HuggingFace model cache directory. None = HF default.
        max_new_tokens: Maximum tokens to generate per call.
        temperature: Sampling temperature.
        max_agents: Max population size for this model (for cost/memory control).
        max_rounds: Max simulation rounds for this model.
    """

    model_id: str
    backend_type: Literal["huggingface", "openai"] = "huggingface"
    context_length: int = 4096
    dtype: str = "float16"
    quantization: str | None = None
    cache_dir: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.7
    max_agents: int = 50
    max_rounds: int = 20

    @classmethod
    def mistral_7b(cls, cache_dir: str | None = None) -> "ModelConfig":
        """Mistral-7B-Instruct-v0.3 — primary BGF model."""
        return cls(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            backend_type="huggingface",
            context_length=32768,
            dtype="float16",
            cache_dir=cache_dir or "/mnt/raid/workspace/lucastourinho/models",
        )

    @classmethod
    def qwen2_5_7b(cls, cache_dir: str | None = None) -> "ModelConfig":
        """Qwen2.5-7B-Instruct — second open-weights model for cross-model validation.

        Non-gated alternative to meta-llama/Llama-3.1-8B-Instruct, comparable in
        scale and RLHF training. Freely accessible without HF approval.
        """
        return cls(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            backend_type="huggingface",
            context_length=131072,
            dtype="float16",
            cache_dir=cache_dir or "/mnt/raid/workspace/lucastourinho/models",
        )

    @classmethod
    def gpt4o_mini(cls) -> "ModelConfig":
        """GPT-4o-mini — external validation via OpenAI API (small-scale only)."""
        return cls(
            model_id="gpt-4o-mini",
            backend_type="openai",
            context_length=128000,
            max_new_tokens=256,
            temperature=0.7,
            max_agents=20,
            max_rounds=10,
        )


def get_backend(config: ModelConfig):
    """Instantiate the appropriate LLM backend for the given ModelConfig.

    Returns an object conforming to LLMBackendProtocol:
        - LLMBackend for HuggingFace models
        - OpenAIBackend for OpenAI API models

    The backend is NOT loaded/authenticated here — call backend.load()
    (HuggingFace) or ensure OPENAI_API_KEY is set (OpenAI) before use.

    Args:
        config: ModelConfig specifying the model and inference parameters.

    Returns:
        Backend instance conforming to LLMBackendProtocol.

    Raises:
        ValueError: If backend_type is unrecognized.
    """
    if config.backend_type == "huggingface":
        from decision.llm_backend import LLMBackend
        return LLMBackend(
            model_id=config.model_id,
            dtype=config.dtype,
            device_map="auto",
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            cache_dir=config.cache_dir,
        )

    if config.backend_type == "openai":
        from decision.openai_backend import OpenAIBackend
        return OpenAIBackend(
            model_id=config.model_id,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            max_retries=1,     # reduce explosion
            min_delay=0.25,    # throttle
        )

    raise ValueError(
        f"Unknown backend_type: {config.backend_type!r}. "
        "Valid values: 'huggingface', 'openai'."
    )
