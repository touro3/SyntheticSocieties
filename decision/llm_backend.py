"""
LLM backend for agent decision inference on Tesla P100 GPUs.

Handles model loading and text generation using HuggingFace transformers.
Designed for P100 constraints (float16, eager attention, no bfloat16).
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import torch


class LLMBackend:
    """
    Singleton-style LLM inference backend.

    Loads a HuggingFace causal LM once and provides a generate() method
    for reuse across agents and rounds.
    """

    _instance: Optional["LLMBackend"] = None

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        dtype: str = "float16",
        device_map: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.dtype = getattr(torch, dtype, torch.float16)
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cache_dir = cache_dir

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Load model and tokenizer. Call once before inference."""
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading LLM: {self.model_id}")
        print(f"  dtype: {self.dtype}, device_map: {self.device_map}")
        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            attn_implementation="eager",  # P100 doesn't support flash attention
        )

        self.model.eval()
        self._loaded = True

        elapsed = time.time() - start
        print(f"  Model loaded in {elapsed:.1f}s")

        # Report GPU usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                if allocated > 0.1:
                    print(f"  GPU {i}: {allocated:.1f}/{total:.1f} GB used")

    def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[str, float]:
        """
        Generate text from chat messages.

        Args:
            messages: Chat-format messages [{"role": "...", "content": "..."}].
            temperature: Sampling temperature (overrides default).
            max_new_tokens: Max tokens to generate (overrides default).

        Returns:
            Tuple of (generated_text, latency_seconds).
        """
        if not self._loaded:
            self.load()

        temp = temperature if temperature is not None else self.temperature
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        # Move to model device
        if hasattr(self.model, "device"):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp if temp > 0 else 1.0,
                do_sample=temp > 0,
                top_p=0.9 if temp > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        latency = time.time() - start

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return result.strip(), latency

    @classmethod
    def get_instance(cls, **kwargs) -> "LLMBackend":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        if cls._instance is not None:
            del cls._instance.model
            del cls._instance.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cls._instance = None
