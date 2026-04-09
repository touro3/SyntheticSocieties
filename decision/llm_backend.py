"""
LLM backend for agent decision inference on Tesla P100 GPUs.

Handles model loading and text generation using HuggingFace transformers.
Designed for P100 constraints (float16, eager attention, no bfloat16).
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

SLOW_INFERENCE_THRESHOLD_S: float = 30.0  # Warn if a single generate() takes longer


class LLMBackend:
    """
    Singleton-style LLM inference backend.

    Loads a HuggingFace causal LM once and provides a generate() method
    for reuse across agents and rounds.
    """

    _instance: Optional[LLMBackend] = None

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        dtype: str = "float16",
        device_map: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        cache_dir: Optional[str] = None,
        context_length: int = 4096,
        inference_timeout: int = 120,
        max_retries: int = 2,
    ):
        self.model_id = model_id
        self.dtype = getattr(torch, dtype, torch.float16)
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.context_length = context_length
        self.inference_timeout = inference_timeout
        self.max_retries = max_retries

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
            local_files_only=True,
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
            local_files_only=True,
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

    def _timed_generate(self, fn, *args, **kwargs):
        """Run fn(*args, **kwargs) in a thread; raise TimeoutError if it exceeds inference_timeout."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=self.inference_timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "LLM generate() timed out after %ds — returning fallback.",
                    self.inference_timeout,
                )
                raise TimeoutError(f"LLM inference exceeded {self.inference_timeout}s")

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
            max_length=self.context_length,
        )

        # Move to model device
        if hasattr(self.model, "device"):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            temperature=temp if temp > 0 else 1.0,
            do_sample=temp > 0,
            top_p=0.9 if temp > 0 else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        for attempt in range(self.max_retries + 1):
            start = time.time()
            try:
                with torch.no_grad():
                    outputs = self._timed_generate(self.model.generate, **inputs, **gen_kwargs)
                latency = time.time() - start
                if latency > SLOW_INFERENCE_THRESHOLD_S:
                    logger.warning(
                        "Slow inference: %.1fs exceeds threshold %.0fs",
                        latency, SLOW_INFERENCE_THRESHOLD_S,
                    )
                new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                return result.strip(), latency
            except TimeoutError:
                if attempt < self.max_retries:
                    logger.warning("generate() attempt %d timed out, retrying…", attempt + 1)
                else:
                    logger.error("generate() timed out after %d attempts, returning fallback.", self.max_retries + 1)
                    return (
                        '{"action_type": "work", "reasoning_summary": "[backend timeout fallback]", "confidence": 0.3}',
                        self.inference_timeout * (self.max_retries + 1),
                    )

    def generate_batch(
        self,
        messages_list: list[list[dict]],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        max_batch_size: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Batch-generate text from multiple chat message sets using memory-safe sub-batching.

        Tokenizes prompts, runs batched forward pass in chunks of `max_batch_size` 
        to prevent CUDA Out-Of-Memory errors from KV cache bloat, and clears memory aggressively.

        Args:
            messages_list: List of chat-format messages.
            temperature: Sampling temperature.
            max_new_tokens: Max tokens per generation.
            max_batch_size: Chunk size to prevent OOM fragmentation.

        Returns:
            List of (generated_text, latency_seconds) tuples.
        """
        if not self._loaded:
            self.load()

        if not messages_list:
            return []

        temp = temperature if temperature is not None else self.temperature
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        all_results = []
        
        for i in range(0, len(messages_list), max_batch_size):
            sub_messages = messages_list[i : i + max_batch_size]
            prompt_texts = []
            
            for messages in sub_messages:
                pt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                prompt_texts.append(pt)

            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.context_length,
            )

            # Move to model device
            if hasattr(self.model, "device"):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                temperature=temp if temp > 0 else 1.0,
                do_sample=temp > 0,
                top_p=0.9 if temp > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            start = time.time()
            batch_ok = False

            try:
                with torch.no_grad():
                    outputs = self._timed_generate(self.model.generate, **inputs, **gen_kwargs)
                total_latency = time.time() - start
                per_item_latency = total_latency / len(sub_messages)

                input_len = inputs["input_ids"].shape[1]
                for j in range(len(sub_messages)):
                    new_tokens = outputs[j][input_len:]
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    all_results.append((text, per_item_latency))
                del outputs
                batch_ok = True

            except TimeoutError:
                logger.warning(
                    "Batch generate timed out (sub-batch %d–%d). Falling back to sequential.",
                    i, i + len(sub_messages) - 1,
                )

            finally:
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not batch_ok:
                # Sequential fallback: process each item individually with timeout
                for single_messages in sub_messages:
                    try:
                        text, lat = self.generate(
                            single_messages, temperature=temp, max_new_tokens=max_tokens
                        )
                    except Exception as exc:
                        logger.error("Sequential fallback also failed: %s", exc)
                        text, lat = (
                            '{"action_type": "work", "reasoning_summary": "[sequential fallback]", "confidence": 0.3}',
                            float(self.inference_timeout),
                        )
                    all_results.append((text, lat))
                continue  # skip the 'del outputs' below (already cleaned up)

        return all_results

    @classmethod
    def get_instance(cls, **kwargs) -> LLMBackend:
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
