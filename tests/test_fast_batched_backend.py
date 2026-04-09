"""Tests for FastBatchedBackend — deprecated wrapper around LLMBackend.

All tests mock the underlying LLMBackend to avoid GPU dependency.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest


class TestFastBatchedBackendImport:
    def test_importing_emits_deprecation_warning(self):
        """Importing the module must emit a DeprecationWarning."""
        import importlib
        import sys

        # Remove cached module to force re-import
        for key in list(sys.modules.keys()):
            if "fast_batched_backend" in key:
                del sys.modules[key]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import decision.fast_batched_backend  # noqa: F401
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)


class TestFastBatchedBackend:
    """Tests using a fully mocked inner LLMBackend."""

    def _make(self) -> "FastBatchedBackend":
        from decision.fast_batched_backend import FastBatchedBackend
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fb = FastBatchedBackend(model_id="fake/model", temperature=0.5)
        return fb

    def test_generate_batch_returns_strings(self):
        fb = self._make()
        fb._backend = MagicMock()
        fb._backend.generate_batch.return_value = [("response text", 0.1)]

        results = fb.generate_batch(["hello world"])

        assert isinstance(results, list)
        assert results == ["response text"]

    def test_generate_batch_converts_to_chat_format(self):
        fb = self._make()
        fb._backend = MagicMock()
        fb._backend.generate_batch.return_value = [("ok", 0.1), ("ok", 0.1)]

        fb.generate_batch(["prompt A", "prompt B"])

        call_args = fb._backend.generate_batch.call_args
        messages_list = call_args[0][0] if call_args[0] else call_args[1]["messages_list"]
        assert messages_list[0] == [{"role": "user", "content": "prompt A"}]
        assert messages_list[1] == [{"role": "user", "content": "prompt B"}]

    def test_empty_input_returns_empty(self):
        fb = self._make()
        fb._backend = MagicMock()
        fb._backend.generate_batch.return_value = []

        result = fb.generate_batch([])
        assert result == []

    def test_batch_size_capped_at_5(self):
        """max_batch_size passed to LLMBackend must not exceed 5."""
        fb = self._make()
        fb._backend = MagicMock()
        fb._backend.generate_batch.return_value = [("r", 0.1)] * 3

        fb.generate_batch(["a", "b", "c"], batch_size=100)

        call_kwargs = fb._backend.generate_batch.call_args[1]
        assert call_kwargs.get("max_batch_size", 999) <= 5

    def test_load_delegates_to_backend(self):
        fb = self._make()
        fb._backend = MagicMock()
        fb.load()
        fb._backend.load.assert_called_once()

    def test_generate_batch_strips_latency_from_output(self):
        fb = self._make()
        fb._backend = MagicMock()
        fb._backend.generate_batch.return_value = [
            ("text1", 99.9),
            ("text2", 0.001),
        ]
        result = fb.generate_batch(["p1", "p2"])
        # Returned list should contain only the text, not tuples
        assert result == ["text1", "text2"]
