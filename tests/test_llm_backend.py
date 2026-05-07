"""Unit tests for LLMBackend timeout, retry, and batch-fallback logic.

No GPU required — model.generate() is mocked to simulate hangs and normal returns.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — skipping LLM backend tests")

from decision.llm_backend import LLMBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(inference_timeout: int = 2, max_retries: int = 1) -> LLMBackend:
    """Return a pre-loaded LLMBackend with mocked model and tokenizer."""
    backend = LLMBackend(
        model_id="test-model",
        inference_timeout=inference_timeout,
        max_retries=max_retries,
    )

    # Mock tokenizer
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.pad_token_id = 0
    tok.eos_token = "</s>"
    tok.apply_chat_template.return_value = "<prompt>"
    # tokenizer(text) → dict with input_ids tensor of shape [1, 5]
    tok.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }

    # batch tokenizer call (list of texts) → shape [N, 5]
    def _batch_tok(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.zeros(n, 5, dtype=torch.long),
            "attention_mask": torch.ones(n, 5, dtype=torch.long),
        }

    tok.side_effect = lambda *a, **kw: (
        _batch_tok(a[0], **kw)
        if isinstance(a[0], list)
        else {
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
    )
    tok.decode.return_value = '{"action": "cooperate"}'

    # Mock model
    model = MagicMock()
    model.device = torch.device("cpu")
    # parameters() must yield an object with a real .device so that
    # next(self.model.parameters()).device resolves to a torch.device.
    # Use side_effect (callable) so repeated calls each return a fresh iterator.
    _mock_param = MagicMock()
    _mock_param.device = torch.device("cpu")
    model.parameters.side_effect = lambda: iter([_mock_param])
    # Normal output: shape [1, 8] — first 5 are input tokens, last 3 are new
    model.generate.return_value = torch.zeros(1, 8, dtype=torch.long)

    backend.tokenizer = tok
    backend.model = model
    backend._loaded = True
    return backend


# ---------------------------------------------------------------------------
# _timed_generate
# ---------------------------------------------------------------------------


class TestTimedGenerate:
    def test_returns_result_on_success(self):
        backend = _make_backend(inference_timeout=5)
        result = backend._timed_generate(lambda: "ok")
        assert result == "ok"

    def test_raises_timeout_error_when_slow(self):
        backend = _make_backend(inference_timeout=1)

        def slow():
            time.sleep(5)
            return "never"

        with pytest.raises(TimeoutError, match="exceeded 1s"):
            backend._timed_generate(slow)

    def test_propagates_non_timeout_exception(self):
        backend = _make_backend(inference_timeout=5)

        def boom():
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            backend._timed_generate(boom)


# ---------------------------------------------------------------------------
# generate() — single-agent path
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_normal_return(self):
        backend = _make_backend(inference_timeout=5)
        text, latency = backend.generate([{"role": "user", "content": "hi"}])
        assert isinstance(text, str)
        assert latency >= 0

    def test_fallback_on_timeout_after_all_retries(self):
        backend = _make_backend(inference_timeout=1, max_retries=1)

        def slow(**kwargs):
            time.sleep(5)
            return torch.zeros(1, 8)

        backend.model.generate.side_effect = slow

        text, latency = backend.generate([{"role": "user", "content": "hi"}])
        assert '"action_type": "work"' in text
        # latency = timeout * (retries + 1) = 1 * 2 = 2
        assert latency == pytest.approx(2.0)

    def test_retries_before_fallback(self):
        backend = _make_backend(inference_timeout=1, max_retries=2)
        call_count = {"n": 0}

        def slow(**kwargs):
            call_count["n"] += 1
            time.sleep(5)
            return torch.zeros(1, 8)

        backend.model.generate.side_effect = slow
        backend.generate([{"role": "user", "content": "hi"}])

        # Should have tried max_retries + 1 = 3 times total
        assert call_count["n"] == 3

    def test_succeeds_on_second_attempt(self):
        backend = _make_backend(inference_timeout=1, max_retries=2)
        call_count = {"n": 0}

        def flaky(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                time.sleep(5)  # first attempt hangs
            return torch.zeros(1, 8)  # second succeeds

        backend.model.generate.side_effect = flaky
        text, _ = backend.generate([{"role": "user", "content": "hi"}])

        assert call_count["n"] == 2
        assert text != '{"action": "work"}'  # got a real decode


# ---------------------------------------------------------------------------
# generate_batch() — batch path
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    def _messages(self, n: int) -> list[list[dict]]:
        return [[{"role": "user", "content": f"msg {i}"}] for i in range(n)]

    def test_normal_batch_return(self):
        backend = _make_backend(inference_timeout=5)
        # Output shape [3, 8] for a batch of 3
        backend.model.generate.return_value = torch.zeros(3, 8, dtype=torch.long)

        results = backend.generate_batch(self._messages(3), max_batch_size=3)
        assert len(results) == 3
        for text, lat in results:
            assert isinstance(text, str)
            assert lat >= 0

    def test_empty_input_returns_empty(self):
        backend = _make_backend()
        assert backend.generate_batch([]) == []

    def test_batch_timeout_falls_back_to_sequential(self):
        backend = _make_backend(inference_timeout=1, max_retries=1)
        call_count = {"batch": 0, "single": 0}

        # Batch generate hangs; sequential generate returns immediately
        def slow_batch(**kwargs):
            n = kwargs.get("input_ids", torch.zeros(1, 5)).shape[0]
            if n > 1:
                call_count["batch"] += 1
                time.sleep(5)
            call_count["single"] += 1
            return torch.zeros(n, 8, dtype=torch.long)

        backend.model.generate.side_effect = slow_batch

        results = backend.generate_batch(self._messages(3), max_batch_size=3)
        assert len(results) == 3
        assert call_count["batch"] >= 1  # batch was attempted

    def test_sequential_fallback_returns_work_on_total_failure(self):
        backend = _make_backend(inference_timeout=1, max_retries=0)

        def always_slow(**kwargs):
            time.sleep(5)
            return torch.zeros(1, 8)

        backend.model.generate.side_effect = always_slow

        results = backend.generate_batch(self._messages(2), max_batch_size=2)
        assert len(results) == 2
        for text, _ in results:
            assert '"action_type": "work"' in text

    def test_sub_batching_aggregates_all_results(self):
        backend = _make_backend(inference_timeout=5)

        def dynamic_output(**kwargs):
            n = kwargs["input_ids"].shape[0]
            return torch.zeros(n, 8, dtype=torch.long)

        backend.model.generate.side_effect = dynamic_output

        results = backend.generate_batch(self._messages(7), max_batch_size=3)
        assert len(results) == 7


# ---------------------------------------------------------------------------
# local_files_only is set at load time (no network calls)
# ---------------------------------------------------------------------------


class TestLoadLocalFilesOnly:
    def teardown_method(self):
        """Reset singleton + global tokenizer to prevent cross-test pollution."""
        LLMBackend.reset()

    def test_from_pretrained_called_with_local_files_only(self):
        """load() must pass local_files_only=True to prevent mid-run HTTPS CLOSE_WAIT hangs."""
        backend = LLMBackend(model_id="fake/model", inference_timeout=5)

        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.device = torch.device("cpu")
        _mp = MagicMock(); _mp.device = torch.device("cpu")
        mock_model.parameters.side_effect = lambda: iter([_mp])

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok) as tok_call,
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as model_call,
            patch("torch.cuda.is_available", return_value=False),
        ):
            backend.load()

        _, tok_kwargs = tok_call.call_args
        assert tok_kwargs.get("local_files_only") is True, (
            "AutoTokenizer.from_pretrained must use local_files_only=True"
        )

        _, model_kwargs = model_call.call_args
        assert model_kwargs.get("local_files_only") is True, (
            "AutoModelForCausalLM.from_pretrained must use local_files_only=True"
        )


# ---------------------------------------------------------------------------
# Logprobs-based uncertainty calibration
# ---------------------------------------------------------------------------


class TestLogprobsCalibration:
    """generate() extracts real logprob confidence from output scores."""

    def test_generate_with_logprobs_returns_three_tuple(self):
        """When return_logprobs=True, generate() returns (text, latency, logprob)."""
        backend = _make_backend(inference_timeout=5)

        # Mock model.generate to return a GenerateDecoderOutput-style object
        # with scores attribute
        output_ids = torch.zeros(1, 8, dtype=torch.long)
        scores = (torch.randn(1, 100),)  # One step of logits over vocab

        mock_output = MagicMock()
        mock_output.__getitem__ = lambda self, idx: output_ids[idx]
        mock_output.sequences = output_ids
        mock_output.scores = scores

        backend.model.generate.return_value = mock_output

        result = backend.generate(
            [{"role": "user", "content": "test"}],
            return_logprobs=True,
        )

        assert len(result) == 3, f"Expected 3-tuple (text, latency, logprob), got {len(result)}"
        text, latency, logprob = result
        assert isinstance(text, str)
        assert isinstance(latency, float)
        assert isinstance(logprob, float)
        assert -float("inf") < logprob <= 0.0, "Logprob must be a non-positive float"

    def test_generate_without_logprobs_returns_two_tuple(self):
        """Default generate() still returns (text, latency) — backward compatible."""
        backend = _make_backend(inference_timeout=5)
        result = backend.generate([{"role": "user", "content": "test"}])
        assert len(result) == 2

    def test_logprob_is_normalized(self):
        """The logprob should be from log_softmax, not raw logits."""
        backend = _make_backend(inference_timeout=5)

        # Create logits where token 0 has high probability
        logits = torch.full((1, 100), -10.0)
        logits[0, 0] = 5.0  # Token 0 is very likely
        scores = (logits,)

        output_ids = torch.zeros(1, 8, dtype=torch.long)
        mock_output = MagicMock()
        mock_output.__getitem__ = lambda self, idx: output_ids[idx]
        mock_output.sequences = output_ids
        mock_output.scores = scores

        backend.model.generate.return_value = mock_output

        _, _, logprob = backend.generate(
            [{"role": "user", "content": "test"}],
            return_logprobs=True,
        )

        # Token 0 has logits=5.0 vs all others=-10.0
        # log_softmax should give a value close to 0 (high confidence)
        assert logprob > -1.0, "High-confidence token should have logprob close to 0"
