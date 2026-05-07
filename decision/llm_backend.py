"""
LLM backend for agent decision inference on Tesla P100 GPUs.

Handles model loading and text generation using HuggingFace transformers.
Designed for P100 constraints (float16, eager attention, no bfloat16).

Supports optional 4-bit quantization via bitsandbytes to reduce model
footprint from ~14.5GB to ~4GB on 16GB GPUs, leaving ample headroom
for KV cache with large agent populations (100+).
"""

from __future__ import annotations

import concurrent.futures
import gc
import logging
import os
import random
import time
from typing import Optional

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# Enable the expandable-segments allocator *before* any CUDA tensor is
# allocated.  This lets PyTorch reuse fragmented reserved-but-unallocated
# blocks instead of failing with OOM even when free memory exists.
# Must be set before the first CUDA call (i.e. at import time).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)

SLOW_INFERENCE_THRESHOLD_S: float = 30.0  # Warn if a single generate() takes longer


def _is_cuda_oom(exc: Exception) -> bool:
    """Return True if *exc* is a CUDA out-of-memory error."""
    if not _TORCH_AVAILABLE:
        return False
    return isinstance(exc, (torch.cuda.OutOfMemoryError,)) or (
        isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)
    )


class LLMBackend:
    """
    Singleton-style LLM inference backend.

    Loads a HuggingFace causal LM once and provides a generate() method
    for reuse across agents and rounds.
    """

    _instance: Optional[LLMBackend] = None

    # Persistent one-worker executor — avoids the 1–5 ms OS thread-creation
    # overhead on every generate() call.  Shared across all instances because
    # only one generate() ever runs at a time on a single GPU.  Replaced on
    # timeout (the zombie GPU thread keeps the worker busy) and on reset().
    _executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    @classmethod
    def _get_executor(cls) -> concurrent.futures.ThreadPoolExecutor:
        if cls._executor is None or cls._executor._shutdown:
            cls._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return cls._executor

    @classmethod
    def _recycle_executor(cls) -> None:
        """Abandon the current executor after a timeout and create a fresh one.

        When a GPU generate() times out the underlying thread is still running
        (CUDA kernels cannot be cancelled). Submitting to that executor would
        queue behind the zombie, making every subsequent retry time out
        instantly.  Creating a new executor gives the next call a fresh idle
        thread without waiting for the zombie to finish.
        """
        # Do not wait (wait=False) — the zombie thread will complete eventually
        # and be garbage-collected without blocking the simulation.
        if cls._executor is not None:
            cls._executor.shutdown(wait=False)
        # Release any CUDA memory that was freed by completed GPU ops but not
        # yet returned to the pool.  Calling this before creating the new
        # executor avoids the new thread starting with a fragmented pool.
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        cls._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        dtype: str = "float16",
        device_map: str = "auto",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        cache_dir: Optional[str] = None,
        context_length: int = 4096,
        inference_timeout: int = 120,
        max_retries: int = 2,
        quantization: Optional[str] = None,
        allow_remote_code: bool = False,
        allow_remote_downloads: bool = False,
    ):
        self.model_id = model_id
        # Reject bfloat16 before any work: P100 (CC 6.0) has no hardware bfloat16.
        # accelerate silently falls back to CPU emulation, causing 30-50s/token.
        if dtype == "bfloat16":
            raise ValueError(
                "dtype='bfloat16' is not supported on Tesla P100 (CC 6.0). "
                "Use dtype='float16'. bfloat16 triggers silent CPU offload via "
                "accelerate, producing action collapse from garbled outputs."
            )
        self.dtype = getattr(torch, dtype, torch.float16) if _TORCH_AVAILABLE else dtype
        self.device_map = device_map
        # Agent decisions are short JSON (~40 tokens).  128 gives ample
        # headroom while halving KV-cache allocation vs the old 256.
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # Honour BGF_MODEL_CACHE env var so generated configs can use null
        # and still find a custom model directory without hardcoding paths.
        self.cache_dir = cache_dir or __import__("os").environ.get("BGF_MODEL_CACHE")
        self.context_length = context_length
        self.inference_timeout = inference_timeout
        self.max_retries = max_retries
        self.quantization = quantization  # "4bit", "8bit", or None
        # trust_remote_code allows arbitrary Python execution from the model
        # repo.  Only enable this when you control the model source.
        self.allow_remote_code = allow_remote_code
        self.allow_remote_downloads = allow_remote_downloads

        self.model = None
        self.tokenizer = None
        self._loaded = False

        # Safe default batch size for generate_batch().  With 4-bit quant the
        # model occupies ~4 GB, leaving ~12 GB for KV cache — batch 16 fits
        # comfortably.  In fp16 (~14.5 GB) only 1–2 GB remain, so default to 4.
        self._max_batch_size: int = 16 if quantization else 4

        # Exponential backoff parameters (MiroFish retry_with_backoff pattern)
        self._backoff_initial_delay: float = 1.0
        self._backoff_factor: float = 2.0
        self._backoff_max_delay: float = 30.0
        # Reduce temperature by this much per retry to get more deterministic
        # outputs when the model is struggling.
        self._retry_temp_reduction: float = 0.1

    def load(self):
        """Load model and tokenizer. Call once before inference.

        When ``self.quantization`` is set to ``"4bit"`` or ``"8bit"``, the
        model is loaded via bitsandbytes quantization, drastically reducing
        VRAM usage (e.g. Mistral-7B: 14.5GB fp16 → ~4GB 4-bit).  This
        leaves ample KV-cache headroom for batches of 100+ agents on a
        single 16GB GPU.
        """
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM: %s", self.model_id)
        logger.info("  dtype: %s, device_map: %s", self.dtype, self.device_map)
        if self.quantization:
            logger.info("  quantization: %s", self.quantization)
        if self.allow_remote_code:
            logger.warning(
                "trust_remote_code=True for %s — only use with models you control.",
                self.model_id,
            )
        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=self.allow_remote_code,
            local_files_only=not self.allow_remote_downloads,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Quantization config ──────────────────────────────────────────────
        # max_memory: pin the model to the current visible GPU only.
        # Setting cpu to "0GiB" hard-blocks accelerate from offloading layers
        # to CPU, which would raise ValueError and invalidate the experiment.
        # With 4-bit NF4 a 7B model fits in ~4 GB; 14 GiB leaves 10 GB for KV-cache.
        _visible_gpu_idx = 0  # CUDA_VISIBLE_DEVICES remaps the target to device 0
        _max_memory: dict = {_visible_gpu_idx: "14GiB", "cpu": "0GiB"}

        model_kwargs: dict = dict(
            device_map=self.device_map,
            cache_dir=self.cache_dir,
            trust_remote_code=self.allow_remote_code,
            local_files_only=not self.allow_remote_downloads,
            attn_implementation="eager",  # P100 doesn't support flash attention
        )

        if self.quantization in ("4bit", "4"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "4-bit quantization requires the bitsandbytes package. "
                    "Install it with:  pip install bitsandbytes>=0.43"
                ) from exc

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["max_memory"] = _max_memory
            # Explicitly override the model's config.json torch_dtype (bfloat16 for
            # Qwen2.5 and Mistral) so that unquantized layers (embed_tokens, lm_head,
            # norm) load in float16 instead of bfloat16. P100 (CC 6.0) has no hardware
            # bfloat16; without this override those layers trigger a CUDA device-side
            # assert in the NF4 dequantization kernel, poisoning the CUDA context and
            # crashing the next model load.
            model_kwargs["torch_dtype"] = self.dtype
            logger.info(
                "  Loading with NF4 4-bit quantization (bitsandbytes) | torch_dtype=%s | max_memory=%s",
                self.dtype,
                _max_memory,
            )

        elif self.quantization in ("8bit", "8"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "8-bit quantization requires the bitsandbytes package. "
                    "Install it with:  pip install bitsandbytes>=0.43"
                ) from exc

            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["max_memory"] = _max_memory
            model_kwargs["torch_dtype"] = self.dtype  # same bfloat16 override reason
            logger.info(
                "  Loading with INT8 quantization (bitsandbytes) | torch_dtype=%s | max_memory=%s",
                self.dtype,
                _max_memory,
            )

        else:
            # Standard fp16/bf16 loading — no max_memory constraint needed
            model_kwargs["torch_dtype"] = self.dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

        self.model.eval()
        self._loaded = True

        # Fail-fast: any CPU-offloaded parameters mean quantization didn't work.
        self._assert_no_cpu_offload()

        # Register tokenizer for exact token counting in prompt budget trimming.
        # Without this, token_budget falls back to a ±25% char heuristic, causing
        # prompts to silently exceed the model's context window on some inputs.
        from decision.token_budget import set_tokenizer as _set_tok

        _set_tok(self.tokenizer)

        elapsed = time.time() - start
        try:
            actual_dtype = next(self.model.parameters()).dtype
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            logger.info(
                "Model loaded in %.1fs | dtype=%s | GPU=%s | VRAM=%.1fGB",
                elapsed,
                actual_dtype,
                gpu_name,
                gpu_mem_gb,
            )
        except StopIteration:
            logger.info("Model loaded in %.1fs", elapsed)

    def _assert_no_cpu_offload(self) -> None:
        """Raise immediately if any model parameters landed on CPU or disk.

        With 4-bit NF4 + max_memory cpu=0GiB all tensors must be on CUDA.
        Any CPU/meta device means the max_memory constraint was ignored or the
        model is too large for the GPU — either way the experiment is invalid.
        """
        off_device = [(name, str(p.device)) for name, p in self.model.named_parameters() if p.device.type != "cuda"]
        if off_device:
            sample = off_device[:5]
            raise RuntimeError(
                f"CPU/disk offload detected after model load — "
                f"{len(off_device)} parameter(s) are not on CUDA (sample: {sample}). "
                "Ensure quantization='4bit', dtype='float16', and that "
                "max_memory is set correctly for this GPU."
            )

        # Report GPU usage after confirming everything is on-device
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                if allocated > 0.1:
                    logger.info("GPU %d: %.1f/%.1f GB used", i, allocated, total)

    def _timed_generate(self, fn, *args, **kwargs):
        """Run fn(*args, **kwargs) in the persistent thread; raise TimeoutError on timeout.

        Uses a class-level ThreadPoolExecutor (one worker) so we never pay the
        OS thread-creation cost (1–5 ms) on every generate() call.
        """
        executor = self.__class__._get_executor()
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=self.inference_timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "LLM generate() timed out after %ds — returning fallback.",
                self.inference_timeout,
            )
            # Replace the executor so the next retry gets a fresh idle thread
            # rather than queueing behind the zombie GPU kernel.
            self.__class__._recycle_executor()
            raise TimeoutError(f"LLM inference exceeded {self.inference_timeout}s")

    def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        return_logprobs: bool = False,
    ) -> tuple[str, float] | tuple[str, float, float]:
        """
        Generate text from chat messages.

        Args:
            messages: Chat-format messages [{"role": "...", "content": "..."}].
            temperature: Sampling temperature (overrides default).
            max_new_tokens: Max tokens to generate (overrides default).
            return_logprobs: If True, return (text, latency, logprob) where
                logprob is the normalized log-probability of the first
                generated token (from log_softmax over the output logits).
                This provides mathematically grounded confidence instead
                of LLM-hallucinated "confidence" values.

        Returns:
            Tuple of (generated_text, latency_seconds) or
            (generated_text, latency_seconds, first_token_logprob).
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

        # Move to model device.
        # Use next(parameters()) rather than model.device: with device_map="auto"
        # the model has no single .device attribute (it's sharded), and the attr
        # may resolve to "meta" or raise. The embedding layer — first param — is
        # always on the primary CUDA device for quantized models.
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # InfNanRemoveLogitsProcessor: guards against float16 logit overflow.
        # Qwen2.5-7B (vocab_size=152064) + 4-bit NF4 + float16 can produce lm_head
        # logits > 65504 (float16 max) that become Inf, then NaN after softmax,
        # crashing torch.multinomial with "probability tensor contains inf/nan".
        # This processor replaces Inf→max_float16 and NaN→-inf before sampling.
        try:
            from transformers.generation.logits_process import (
                InfNanRemoveLogitsProcessor,
                LogitsProcessorList,
            )

            _logits_processor = LogitsProcessorList([InfNanRemoveLogitsProcessor()])
        except ImportError:
            _logits_processor = None

        gen_kwargs_base = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if _logits_processor is not None:
            gen_kwargs_base["logits_processor"] = _logits_processor

        # Enable score output for logprob extraction
        if return_logprobs:
            gen_kwargs_base["output_scores"] = True
            gen_kwargs_base["return_dict_in_generate"] = True

        delay = self._backoff_initial_delay

        # inference_mode() is strictly superior to no_grad(): it additionally
        # prevents view ops from producing unnecessary intermediate tensors,
        # reducing peak VRAM usage ~10–30% with no accuracy cost.
        for attempt in range(self.max_retries + 1):
            # Progressive temperature reduction: cooler on each retry to
            # produce more deterministic output when the model struggles.
            retry_temp = max(0.1, temp - (attempt * self._retry_temp_reduction))

            gen_kwargs = {
                **gen_kwargs_base,
                "temperature": retry_temp if retry_temp > 0 else 1.0,
                "do_sample": retry_temp > 0,
                "top_p": 0.9 if retry_temp > 0 else 1.0,
            }

            start = time.time()
            _timed_out = False
            try:
                with torch.inference_mode():
                    outputs = self._timed_generate(self.model.generate, **inputs, **gen_kwargs)
                latency = time.time() - start
                if latency > SLOW_INFERENCE_THRESHOLD_S:
                    logger.warning(
                        "Slow inference: %.1fs exceeds threshold %.0fs",
                        latency,
                        SLOW_INFERENCE_THRESHOLD_S,
                    )

                # Extract sequences — handle both dict-style and tensor outputs
                if return_logprobs and hasattr(outputs, "sequences"):
                    sequences = outputs.sequences
                else:
                    sequences = outputs

                input_len = inputs["input_ids"].shape[1]
                new_tokens = sequences[0][input_len:]
                result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

                if return_logprobs and hasattr(outputs, "scores") and outputs.scores:
                    # Extract normalized log-probability of the first generated token
                    first_step_logits = outputs.scores[0]  # shape: [batch, vocab]
                    log_probs = torch.nn.functional.log_softmax(first_step_logits, dim=-1)
                    first_token_id = new_tokens[0].item()
                    logprob = log_probs[0, first_token_id].item()
                    del outputs, sequences, new_tokens
                    return result.strip(), latency, logprob

                del outputs, sequences, new_tokens
                return result.strip(), latency

            except TimeoutError:
                _timed_out = True
                if attempt < self.max_retries:
                    # Jittered exponential backoff (MiroFish pattern)
                    current_delay = min(delay, self._backoff_max_delay)
                    jittered = current_delay * (0.5 + random.random())
                    next_temp = max(0.1, temp - ((attempt + 1) * self._retry_temp_reduction))
                    logger.warning(
                        "generate() attempt %d/%d timed out; retrying in %.1fs (temp %.2f → %.2f)",
                        attempt + 1,
                        self.max_retries + 1,
                        jittered,
                        retry_temp,
                        next_temp,
                    )
                    time.sleep(jittered)
                    delay *= self._backoff_factor
                else:
                    logger.error("generate() timed out after %d attempts, returning fallback.", self.max_retries + 1)
                    fallback = (
                        '{"action_type": "work", "reasoning_summary": "[backend timeout fallback]", "confidence": 0.3}',
                        self.inference_timeout * (self.max_retries + 1),
                    )
                    if return_logprobs:
                        return fallback[0], fallback[1], float("-inf")
                    return fallback
            finally:
                # Release input tensors on every attempt. Skip synchronize after
                # timeout — the GPU kernel may still be running and synchronize()
                # would block until it finishes, defeating the timeout.
                if not _timed_out and torch.cuda.is_available():
                    torch.cuda.synchronize()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ── VRAM probe ────────────────────────────────────────────────────────────

    @staticmethod
    def _free_vram_gb() -> float:
        """Return currently free VRAM on device 0 in GB, or inf if no CUDA."""
        if not torch.cuda.is_available():
            return float("inf")
        props = torch.cuda.get_device_properties(0)
        reserved = torch.cuda.memory_reserved(0)
        # Use reserved as the high-water mark for the pool.
        free = props.total_memory - reserved
        return free / 1e9

    @staticmethod
    def _vram_safe_batch_size(max_batch_size: int) -> int:
        """Clamp max_batch_size to what current free VRAM can sustain.

        Heuristic: each agent in a Mistral-7B batch consumes ~150-300 MB of
        KV cache at context_length=4096.  We reserve 1 GB as safety headroom
        and estimate 250 MB per batch item.
        """
        free_gb = LLMBackend._free_vram_gb()
        headroom_gb = 1.0
        per_item_gb = 0.25
        vram_safe = max(1, int((free_gb - headroom_gb) / per_item_gb))
        clamped = min(max_batch_size, vram_safe)
        if clamped < max_batch_size:
            logger.debug(
                "VRAM-safe batch size: %d (free=%.1fGB, requested=%d)",
                clamped,
                free_gb,
                max_batch_size,
            )
        return clamped

    def generate_batch(
        self,
        messages_list: list[list[dict]],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        max_batch_size: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Batch-generate text from multiple chat message sets using memory-safe
        sub-batching with automatic OOM recovery.

        Strategy:
          1. Chunk the population into mini-batches of ``max_batch_size``.
          2. Attempt a batched ``model.generate()`` for each chunk.
          3. On CUDA OOM → halve the chunk, clear cache, retry with smaller
             sub-batches.  Continues halving until batch_size == 1.
          4. If batch_size == 1 still OOMs → fall back to the sequential
             ``generate()`` path (which has its own retry + backoff).

        This guarantees the simulation never crashes from KV-cache bloat,
        regardless of population size.  On a 16GB P100 with fp16 Mistral-7B
        the practical max chunk is ~3–5; with 4-bit quantization it can
        sustain 10–15.

        Args:
            messages_list: List of chat-format messages.
            temperature: Sampling temperature.
            max_new_tokens: Max tokens per generation.
            max_batch_size: Initial chunk size (will be halved on OOM).

        Returns:
            List of (generated_text, latency_seconds) tuples.
        """
        if not self._loaded:
            self.load()

        if not messages_list:
            return []

        temp = temperature if temperature is not None else self.temperature
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        all_results: list[tuple[str, float]] = []
        # Clamp to VRAM-safe size before the first chunk attempt.
        effective_batch_size = self._vram_safe_batch_size(max_batch_size)

        offset = 0
        while offset < len(messages_list):
            sub_messages = messages_list[offset : offset + effective_batch_size]
            chunk_results = self._try_batch_chunk(
                sub_messages,
                temp,
                max_tokens,
                effective_batch_size,
            )

            if chunk_results is not None:
                all_results.extend(chunk_results)
                offset += len(sub_messages)
            else:
                # OOM or timeout — halve batch size and retry from same offset
                if effective_batch_size > 1:
                    new_size = max(1, effective_batch_size // 2)
                    logger.warning(
                        "Batch OOM/timeout at chunk_size=%d; halving to %d and retrying.",
                        effective_batch_size,
                        new_size,
                    )
                    effective_batch_size = new_size
                    # Don't advance offset — retry same chunk with smaller size
                else:
                    # batch_size == 1 failed → fall back to sequential generate()
                    logger.warning(
                        "Batch size 1 failed — falling back to sequential generate() for %d messages.",
                        len(sub_messages),
                    )
                    for single_messages in sub_messages:
                        try:
                            text, lat = self.generate(
                                single_messages,
                                temperature=temp,
                                max_new_tokens=max_tokens,
                            )
                        except Exception as exc:
                            logger.error("Sequential fallback also failed: %s", exc)
                            text, lat = (
                                '{"action_type": "work", "reasoning_summary": "[sequential fallback]", "confidence": 0.3}',
                                float(self.inference_timeout),
                            )
                        all_results.append((text, lat))
                    offset += len(sub_messages)

        if effective_batch_size < max_batch_size:
            logger.info(
                "Batch size was auto-reduced from %d to %d due to memory pressure.",
                max_batch_size,
                effective_batch_size,
            )

        return all_results

    def _try_batch_chunk(
        self,
        sub_messages: list[list[dict]],
        temp: float,
        max_tokens: int,
        chunk_size: int,
    ) -> list[tuple[str, float]] | None:
        """Attempt a single sub-batch generate.  Returns results or None on OOM/timeout."""
        prompt_texts = []
        for messages in sub_messages:
            pt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
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

        # Move to model device (same reasoning as generate() — use parameters()).
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            from transformers.generation.logits_process import (
                InfNanRemoveLogitsProcessor,
                LogitsProcessorList,
            )

            _lp = LogitsProcessorList([InfNanRemoveLogitsProcessor()])
        except ImportError:
            _lp = None

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            temperature=temp if temp > 0 else 1.0,
            do_sample=temp > 0,
            top_p=0.9 if temp > 0 else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if _lp is not None:
            gen_kwargs["logits_processor"] = _lp

        start = time.time()
        _timed_out = False
        try:
            # inference_mode() > no_grad(): prevents intermediate tensor views,
            # saving additional VRAM on top of disabling autograd tracking.
            with torch.inference_mode():
                outputs = self._timed_generate(
                    self.model.generate,
                    **inputs,
                    **gen_kwargs,
                )
            total_latency = time.time() - start
            per_item_latency = total_latency / len(sub_messages)

            results: list[tuple[str, float]] = []
            input_len = inputs["input_ids"].shape[1]
            for j in range(len(sub_messages)):
                new_tokens = outputs[j][input_len:]
                # Guard: all-padding / zero-length output is a silent failure.
                if len(new_tokens) == 0 or new_tokens.sum().item() == 0:
                    logger.warning(
                        "Batch item %d produced empty/zero token output — using work fallback.",
                        j,
                    )
                    results.append(
                        (
                            '{"action_type": "work", "reasoning_summary": "[empty output]", "confidence": 0.3}',
                            per_item_latency,
                        )
                    )
                    continue
                text = self.tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                ).strip()
                # Guard: decoded to empty string despite non-empty tokens.
                if not text:
                    logger.warning("Batch item %d decoded to empty string — using work fallback.", j)
                    text = '{"action_type": "work", "reasoning_summary": "[empty decode]", "confidence": 0.3}'
                results.append((text, per_item_latency))

            del outputs
            return results

        except TimeoutError:
            _timed_out = True
            logger.warning(
                "Sub-batch of %d timed out after %ds.",
                len(sub_messages),
                self.inference_timeout,
            )
            return None

        except Exception as exc:
            if _is_cuda_oom(exc):
                logger.warning(
                    "CUDA OOM on sub-batch of %d (%.1f GB allocated). Will retry with smaller batch.",
                    len(sub_messages),
                    torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                )
                return None
            raise  # Re-raise non-OOM exceptions

        finally:
            del inputs
            if torch.cuda.is_available():
                # Skip synchronize after timeout — the GPU kernel may still be
                # running and synchronize() would block until it finishes,
                # defeating the timeout mechanism entirely.
                if not _timed_out:
                    torch.cuda.synchronize()
                torch.cuda.empty_cache()

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
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            cls._instance = None
        # Shut down and replace the persistent executor so the next test
        # or seed gets a clean thread with no stale state.
        if cls._executor is not None:
            cls._executor.shutdown(wait=False)
            cls._executor = None
        # Unregister tokenizer from token_budget so tests use the heuristic.
        # set_tokenizer(None) also clears the static-section LRU cache so the
        # next test run doesn't see stale tokenizer-based counts.
        from decision.token_budget import set_tokenizer as _set_tok

        _set_tok(None)

    @classmethod
    def between_seeds(cls) -> None:
        """Release transient CUDA allocations between back-to-back seed runs.

        The singleton model stays loaded (no reload penalty) but all KV-cache
        fragments and intermediate tensors from the previous run are freed.
        Call this after each seed's simulation finishes and before the next
        seed begins.

        Steps:
          1. Python GC — drops any lingering tensor references in Python objects.
          2. CUDA synchronize — waits for all in-flight GPU ops to complete so
             their allocations become reclaimable.
          3. empty_cache — returns freed blocks to the OS/CUDA memory pool.
          4. reset_peak_memory_stats — clears the high-water mark counter so
             logging for the next seed is accurate.
        """
        gc.collect()
        if torch.cuda.is_available():
            # empty_cache BEFORE synchronize: async OOM from inference surfaces at
            # synchronize() — draining first reduces the chance of a spurious raise.
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning(
                        "CUDA OOM at between_seeds synchronize (deferred async error"
                        " from previous inference); draining cache and continuing: %s",
                        exc,
                    )
                    torch.cuda.empty_cache()
                else:
                    raise
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(
                "between_seeds cleanup: %.1f GB allocated, %.1f GB reserved",
                allocated,
                reserved,
            )
