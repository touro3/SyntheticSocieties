"""
Output parser for extracting structured ProposedAction from LLM text.

Handles JSON extraction, validation, and fallback when the LLM
produces malformed output.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Optional

from decision.constants import (
    DEFAULT_COOPERATE_AMOUNT,
    DEFAULT_KEYWORD_CONFIDENCE,
    DEFAULT_WORK_AMOUNT,
    MAX_ACTION_AMOUNT,
)
from decision.schemas import ProposedAction


VALID_ACTIONS = {"work", "save", "cooperate"}

# ── Parse statistics tracking ────────────────────────────────────────────────
# Module-level counters for diagnosing LLM output quality across runs.
_parse_stats: dict[str, int] = {}


def get_parse_stats() -> dict[str, int]:
    """Return a copy of the cumulative parse method statistics."""
    return dict(_parse_stats)


def reset_parse_stats() -> None:
    """Reset all parse statistics counters to zero."""
    _parse_stats.clear()


def parse_llm_output(
    raw_text: str,
    neighbors: Optional[list[str]] = None,
) -> tuple[Optional[ProposedAction], dict]:
    """
    Parse LLM output text into a ProposedAction.

    Attempts multiple strategies:
      1. Direct JSON parse of the entire text
      2. Regex extraction of JSON block
      3. Keyword-based fallback

    Args:
        raw_text: Raw text output from the LLM.
        neighbors: List of valid neighbor agent IDs for validation.

    Returns:
        Tuple of (ProposedAction or None, parse_metadata dict).
    """
    metadata = {
        "raw_text": raw_text,
        "parse_method": None,
        "parse_success": False,
        "parse_error": None,
    }

    if not raw_text or not raw_text.strip():
        metadata["parse_error"] = "Empty LLM output"
        return None, metadata

    # Strategy 1: Direct JSON parse
    action = _try_direct_json(raw_text.strip())
    if action:
        metadata["parse_method"] = "direct_json"
        metadata["parse_success"] = True
        result = _validate_action(action, neighbors, metadata)
        if result[0] is not None:
            _parse_stats["direct_json"] = _parse_stats.get("direct_json", 0) + 1
            return result
        # Validation failed (e.g. invalid action_type) — fall through
        _parse_stats["failed"] = _parse_stats.get("failed", 0) + 1
        return result

    # Strategy 2: Regex extraction
    action = _try_regex_json(raw_text)
    if action:
        metadata["parse_method"] = "regex_json"
        metadata["parse_success"] = True
        result = _validate_action(action, neighbors, metadata)
        if result[0] is not None:
            _parse_stats["regex_json"] = _parse_stats.get("regex_json", 0) + 1
            return result
        _parse_stats["failed"] = _parse_stats.get("failed", 0) + 1
        return result

    # Strategy 3: Keyword fallback
    action = _try_keyword_fallback(raw_text, neighbors)
    if action:
        metadata["parse_method"] = "keyword_fallback"
        metadata["parse_success"] = True
        _parse_stats["keyword_fallback"] = _parse_stats.get("keyword_fallback", 0) + 1
        return action, metadata

    metadata["parse_error"] = "All parsing strategies failed"
    _parse_stats["failed"] = _parse_stats.get("failed", 0) + 1
    return None, metadata


def _try_direct_json(text: str) -> Optional[dict]:
    """Try parsing the entire text as JSON."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action_type" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _try_regex_json(text: str) -> Optional[dict]:
    """Extract JSON block from text using regex."""
    # Match JSON objects (greedy, handles nested)
    patterns = [
        r'\{[^{}]*"action_type"[^{}]*\}',  # Simple flat JSON
        r'\{.*?"action_type".*?\}',          # More permissive
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if "action_type" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                continue

    return None


def _try_keyword_fallback(text: str, neighbors: Optional[list[str]]) -> Optional[ProposedAction]:
    """Last resort: infer action from keywords in the text.

    Uses word-boundary regex to avoid false positives (e.g., "network"
    matching "work"). When multiple action keywords are found, scores
    each action and picks the highest.
    """
    text_lower = text.lower()

    # Word-boundary patterns to avoid substring false positives
    _ACTION_PATTERNS = {
        "cooperate": [r'\bcooperat\w*\b', r'\bhelp\b', r'\bshar\w*\b', r'\bgive\b'],
        "save": [r'\bsav\w*\b', r'\bconserv\w*\b', r'\brest\b', r'\bpreserv\w*\b'],
        "work": [r'\bwork\b', r'\bearn\b', r'\blabor\b', r'\bincome\b'],
    }

    scores: dict[str, int] = {}
    for action_type, patterns in _ACTION_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score > 0:
            scores[action_type] = score

    if not scores:
        return None

    # Pick the action with the highest keyword score
    best_action = max(scores, key=scores.get)

    if best_action == "cooperate":
        target = neighbors[0] if neighbors else None
        return ProposedAction(
            action_type="cooperate",
            target_agent_id=target,
            amount=DEFAULT_COOPERATE_AMOUNT,
            reasoning_summary="[parsed from keywords: cooperation detected]",
            confidence=DEFAULT_KEYWORD_CONFIDENCE,
        )

    if best_action == "save":
        return ProposedAction(
            action_type="save",
            amount=DEFAULT_COOPERATE_AMOUNT,
            reasoning_summary="[parsed from keywords: saving detected]",
            confidence=DEFAULT_KEYWORD_CONFIDENCE,
        )

    # best_action == "work"
    return ProposedAction(
        action_type="work",
        amount=DEFAULT_WORK_AMOUNT,
        reasoning_summary="[parsed from keywords: work detected]",
        confidence=DEFAULT_KEYWORD_CONFIDENCE,
    )


def _validate_action(
    data: dict,
    neighbors: Optional[list[str]],
    metadata: dict,
) -> tuple[Optional[ProposedAction], dict]:
    """Validate and construct ProposedAction from parsed dict."""
    action_type = data.get("action_type", "").strip().lower()

    if action_type not in VALID_ACTIONS:
        metadata["parse_error"] = f"Invalid action_type: {action_type}"
        return None, metadata

    target = data.get("target_agent_id")
    amount = data.get("amount")
    reasoning = data.get("reasoning_summary", "LLM decision")
    confidence = data.get("confidence")

    # Validate cooperate has a valid target
    if action_type == "cooperate":
        if target is None and neighbors:
            target = neighbors[0]
        elif target and neighbors and target not in neighbors:
            # LLM picked an invalid neighbor — use first valid one
            target = neighbors[0] if neighbors else None

    # Clamp amount to valid range [0, MAX_ACTION_AMOUNT]
    if amount is not None:
        try:
            amount = float(amount)
            amount = max(0.0, min(amount, MAX_ACTION_AMOUNT))
        except (ValueError, TypeError):
            amount = DEFAULT_COOPERATE_AMOUNT
    else:
        amount = DEFAULT_WORK_AMOUNT if action_type == "work" else DEFAULT_COOPERATE_AMOUNT

    # Clamp confidence
    if confidence is not None:
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(confidence, 1.0))
        except (ValueError, TypeError):
            confidence = 0.5

    try:
        action = ProposedAction(
            action_type=action_type,
            target_agent_id=target,
            amount=amount,
            reasoning_summary=str(reasoning)[:500],
            confidence=confidence,
        )
        return action, metadata
    except Exception as e:
        metadata["parse_error"] = f"ProposedAction construction failed: {e}"
        return None, metadata
