"""Tests for the PolicyProtocol structural contract."""
import pytest
from decision.policy_protocol import PolicyProtocol
from decision.mock_policy import MockPolicy
from decision.random_policy import RandomPolicy
from decision.rule_based_policy import RuleBasedPolicy
from decision.template_policy import TemplatePolicy
from decision.auditable_random_policy import AuditableRandomPolicy
from decision.data_driven_policy import DataDrivenPolicy


BASELINE_POLICIES = [
    MockPolicy,
    RandomPolicy,
    RuleBasedPolicy,
    TemplatePolicy,
]


@pytest.mark.parametrize("policy_cls", BASELINE_POLICIES)
def test_baseline_policy_satisfies_protocol(policy_cls):
    instance = policy_cls()
    assert isinstance(instance, PolicyProtocol)


def test_auditable_random_satisfies_protocol():
    instance = AuditableRandomPolicy(seed=42, condition_name="test")
    assert isinstance(instance, PolicyProtocol)


def test_non_conforming_class_fails():
    class BadPolicy:
        def do_something(self):
            pass

    assert not isinstance(BadPolicy(), PolicyProtocol)


def test_missing_method_fails():
    class AlmostPolicy:
        def propose_action(self, profile, state, memory):  # missing context, round_id
            pass

    # Protocol runtime check only validates method name exists,
    # not full signature — but the method must exist.
    # A class completely without propose_action should fail.
    class NoMethod:
        pass

    assert not isinstance(NoMethod(), PolicyProtocol)
