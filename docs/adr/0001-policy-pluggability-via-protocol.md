# 1. Policy pluggability via `PolicyProtocol` (PEP 544)

- Status: accepted
- Date: 2026-02-12
- Deciders: BGF core
- Tags: architecture, pep-544, llm-agnostic

## Context

BGF needs to evaluate at least four families of agent decision policies
side-by-side: pure LLM (Condition A), ESS-RAG-grounded LLM (Condition B),
generative-agents proxy (Condition C), and deterministic rule-based ESS
(Condition D). The simulation kernel must not know which is in use —
swapping policies at the config layer is the entire point of the matrix
sweep in `scripts/run_experiment_matrix.py`.

Three options were on the table:

1. **Common abstract base class** (`class Policy(ABC)` with `@abstractmethod`).
   Forces inheritance; awkward for the rule-based and template policies
   that have no shared state with the LLM policies.
2. **Duck typing**, no formal interface. Easy to ship; impossible to
   type-check; reviewers can't tell at a glance what a policy must
   provide.
3. **Structural typing via `typing.Protocol`** (PEP 544). No inheritance
   required, type checker enforces the contract, runtime
   `isinstance(policy, Protocol)` works for backend dispatch
   (`SimulationKernel._can_use_batched_mode`).

## Decision

We use a `PolicyProtocol` (PEP 544 `runtime_checkable` Protocol).
Concrete policies declare the required methods (`decide`,
`_fallback_action`, optional `graph_rag_context` / `sql_rag_context`)
without inheriting from any base. The kernel dispatches on
`isinstance(policy, LLMPolicy)` only where the batched-inference path
diverges from the per-agent path; elsewhere everything is duck-typed
through the protocol.

## Consequences

**Positive**

- New policies (placebo, padded, generative-agents) drop in without any
  base-class plumbing.
- `mypy` enforces the contract at type-check time.
- The rule-based policy stays a free function under a thin class
  wrapper — no inheritance ceremony for a decision tree.

**Negative**

- Slightly less newcomer-friendly than an ABC; the contract is
  documented in `decision/policy_protocol.py` rather than enforced by
  Python at import time.
- `isinstance(policy, PolicyProtocol)` is `True` for *any* object that
  happens to have the right attribute names — typo-prone if a policy
  exposes the wrong-shaped method.

**Mitigation**

- Tests assert protocol conformance for every shipped policy
  (`tests/test_policy_protocol.py`).
- `mypy` configuration in `pyproject.toml` enables
  `check_untyped_defs = true` on `decision.*` so the contract surface
  is type-checked.
