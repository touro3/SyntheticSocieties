"""Tests for LLM backend protocol conformance."""

import pytest
from decision.backend_protocol import LLMBackendProtocol, BatchLLMBackendProtocol


class MockSingleBackend:
    """Conforms to LLMBackendProtocol."""

    def generate(self, messages, temperature=None):
        return ("output", 0.1)


class MockBatchBackend:
    """Conforms to BatchLLMBackendProtocol."""

    def generate(self, messages, temperature=None):
        return ("output", 0.1)

    def generate_batch(self, messages_list, temperature=None):
        return [("output", 0.1) for _ in messages_list]


class NonConformingBackend:
    """Missing generate method."""

    def run(self, text):
        return text


class TestLLMBackendProtocol:
    def test_single_backend_conforms(self):
        assert isinstance(MockSingleBackend(), LLMBackendProtocol)

    def test_batch_backend_conforms_to_single(self):
        assert isinstance(MockBatchBackend(), LLMBackendProtocol)

    def test_batch_backend_conforms_to_batch(self):
        assert isinstance(MockBatchBackend(), BatchLLMBackendProtocol)

    def test_single_backend_does_not_conform_to_batch(self):
        assert not isinstance(MockSingleBackend(), BatchLLMBackendProtocol)

    def test_non_conforming_fails(self):
        assert not isinstance(NonConformingBackend(), LLMBackendProtocol)
