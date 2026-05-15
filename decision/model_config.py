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

import os
from dataclasses import dataclass
from typing import Literal

from decision.token_budget import DEFAULT_MAX_TOKENS, budget_for_model


def _default_model_cache_dir() -> str | None:
    """Resolve the HuggingFace model cache directory.

    Resolution order (first non-empty wins):
      1. BGF_MODEL_CACHE_DIR (project-level override)
      2. HF_HOME (HuggingFace's own convention)
      3. None (HuggingFace uses its default ~/.cache/huggingface)
    """
    return os.environ.get("BGF_MODEL_CACHE_DIR") or os.environ.get("HF_HOME") or None


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
        prompt_budget: Maximum prompt tokens passed to trim_to_budget().
            Balances RAG context richness against inference speed. Set
            per-model via budget_for_model(); override here if needed.
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
    prompt_budget: int = DEFAULT_MAX_TOKENS
    # Alignment-bias provenance tag (Phase 4): "instruct" (default RLHF-tuned),
    # "base" (pre-RLHF), or "uncensored". Recorded so results can attribute
    # behavior to model alignment vs emergent grounding. Informational only —
    # it does not itself change weights; pair it with the matching model_id.
    model_variant: Literal["instruct", "base", "uncensored"] = "instruct"

    def __post_init__(self) -> None:
        # ── Env-var overrides (Phase 4: model agnosticism) ────────────────
        # Make base/instruct/uncensored and backend swappable WITHOUT code
        # edits, so alignment bias can be isolated from emergent behavior.
        # Unset env ⇒ caller-provided values are preserved exactly (backward
        # compatible, including the .mistral_7b()/.gpt4o_mini() classmethods).
        env_model = os.environ.get("BGF_MODEL_ID")
        if env_model:
            self.model_id = env_model
        env_backend = os.environ.get("BGF_BACKEND_TYPE")
        if env_backend:
            if env_backend not in ("huggingface", "openai"):
                raise ValueError(f"BGF_BACKEND_TYPE must be 'huggingface' or 'openai'; got {env_backend!r}")
            self.backend_type = env_backend  # type: ignore[assignment]
        env_variant = os.environ.get("BGF_MODEL_VARIANT")
        if env_variant:
            if env_variant not in ("instruct", "base", "uncensored"):
                raise ValueError(f"BGF_MODEL_VARIANT must be 'instruct', 'base', or 'uncensored'; got {env_variant!r}")
            self.model_variant = env_variant  # type: ignore[assignment]

        # Auto-derive budget from model_id when the caller used the default
        # (recomputed after any env override so it tracks the actual model).
        if self.prompt_budget == DEFAULT_MAX_TOKENS:
            self.prompt_budget = budget_for_model(self.model_id)

    @classmethod
    def mistral_7b(cls, cache_dir: str | None = None) -> ModelConfig:
        """Mistral-7B-Instruct-v0.3 — primary BGF model.

        prompt_budget=4096: quality sweet spot for 7B attention heads; leaves
        ample GPU headroom while fitting full RAG context in most prompts.
        """
        return cls(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            backend_type="huggingface",
            context_length=32768,
            dtype="float16",
            quantization="4bit",
            cache_dir=cache_dir or _default_model_cache_dir(),
            prompt_budget=4096,
        )

    @classmethod
    def qwen2_5_7b(cls, cache_dir: str | None = None) -> ModelConfig:
        """Qwen2.5-7B-Instruct — second open-weights model for cross-model validation.

        Non-gated alternative to meta-llama/Llama-3.1-8B-Instruct, comparable in
        scale and RLHF training. Freely accessible without HF approval.

        prompt_budget=6144: Qwen2.5 handles long contexts reliably; richer
        population_context and social_context improve ESS grounding quality.
        """
        return cls(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            backend_type="huggingface",
            context_length=131072,
            dtype="float16",
            quantization="4bit",
            cache_dir=cache_dir or _default_model_cache_dir(),
            prompt_budget=6144,
        )

    @classmethod
    def gpt4o_mini(cls) -> ModelConfig:
        """GPT-4o-mini — external validation via OpenAI API (small-scale only).

        prompt_budget=8192: API cost is bounded by max_agents=20; maximising
        context window gives the richest RAG injection for the fewest runs.
        """
        return cls(
            model_id="gpt-4o-mini",
            backend_type="openai",
            context_length=128000,
            max_new_tokens=256,
            temperature=0.7,
            max_agents=20,
            max_rounds=10,
            prompt_budget=8192,
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
            context_length=config.context_length,
            quantization=config.quantization,
            inference_timeout=getattr(config, "inference_timeout", 120),
            max_retries=getattr(config, "max_retries", 2),
        )

    if config.backend_type == "openai":
        from decision.openai_backend import OpenAIBackend

        return OpenAIBackend(
            model_id=config.model_id,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            max_retries=1,  # reduce explosion
            min_delay=0.25,  # throttle
        )

    raise ValueError(f"Unknown backend_type: {config.backend_type!r}. Valid values: 'huggingface', 'openai'.")
