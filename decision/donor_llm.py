"""LLM-driven donor decision for the cultural-evolution loop.

Bridges any ``LLMBackendProtocol`` backend (local Mistral, GPT-4o, Claude,
Gemini) to the donor-game ``decision_fn`` signature expected by
``simulation.evolution_kernel.EvolutionRunner`` and
``environment.donor_game.DonorGameEngine``.

The prompt is intentionally short (~150 tokens) to bound API cost across the
10 gen × 12 agents × 12 rounds × seeds × models sweep. It carries:
  * the recipient's **reputation** (recent donation rate + window), and
  * the donor's **inherited strategy** text (cultural transmission), and
  * optionally the donor's **ESS persona** (grounding × evolution arm).
"""

from __future__ import annotations

import re

from environment.donor_game import DONATE, KEEP, DonorDecision, ReputationView

_SYSTEM = (
    "You are an agent in an iterated Donor Game. Each round you are paired with another agent. "
    "As the donor you may DONATE (it costs you, but benefits the recipient twice as much) or KEEP "
    "(you lose nothing, the recipient gets nothing). Your goal is to maximize your own accumulated "
    "resources over many rounds. Reply with exactly one word on the first line: DONATE or KEEP."
)


def _persona_line(donor) -> str:
    p = donor.profile
    bits = []
    if getattr(p, "trust_people", None) is not None:
        bits.append(f"trust-in-others {p.trust_people:.2f}")
    if getattr(p, "risk_tolerance", None) is not None:
        bits.append(f"risk-tolerance {p.risk_tolerance:.2f}")
    if getattr(p, "country", None):
        bits.append(f"country {p.country}")
    return ("Your background: " + ", ".join(bits) + ".") if bits else ""


def build_donor_prompt(donor, recipient, rep: ReputationView, *, grounded: bool) -> list[dict]:
    if rep.n_observed == 0:
        rep_line = "You have no information about this recipient (first encounter)."
    else:
        rep_line = f"This recipient donated in {rep.donate_rate:.0%} of their last {rep.n_observed} observed decisions."
    strategy = getattr(donor, "inherited_strategy", "") or ""
    strategy_line = f"Advice inherited from your predecessor: {strategy}" if strategy else ""
    persona_line = _persona_line(donor) if grounded else ""

    user = "\n".join(
        line
        for line in [
            rep_line,
            persona_line,
            strategy_line,
            "Do you DONATE or KEEP? Answer DONATE or KEEP, then a short reason.",
        ]
        if line
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def parse_donor_response(text: str) -> str:
    """Return DONATE or KEEP from a model response. Defaults to KEEP (the Nash action)."""
    t = (text or "").strip().lower()
    # First explicit keyword wins.
    m = re.search(r"\b(donate|keep)\b", t)
    if m:
        return DONATE if m.group(1) == "donate" else KEEP
    return KEEP


def make_llm_donor_decider(backend, *, grounded: bool = False, temperature: float | None = None):
    """Return a ``decision_fn(donor, recipient, rep, round_id) -> DonorDecision``."""

    def _decide(donor, recipient, rep: ReputationView, round_id: int) -> DonorDecision:
        messages = build_donor_prompt(donor, recipient, rep, grounded=grounded)
        text, _ = backend.generate(messages, temperature=temperature)
        action = parse_donor_response(text)
        reason = (text or "").strip().splitlines()
        reasoning = reason[-1][:200] if reason else ""
        return DonorDecision(action, reasoning=reasoning)

    return _decide
