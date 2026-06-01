"""Tests for LLMPolicy — no GPU required."""

from decision.llm_policy import LLMPolicy


def test_llm_policy_stores_prompt_budget():
    policy = LLMPolicy(prompt_budget=4096)
    assert policy.prompt_budget == 4096


def test_llm_policy_default_prompt_budget_is_none():
    policy = LLMPolicy()
    assert policy.prompt_budget is None
