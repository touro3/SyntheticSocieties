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

from decision.schemas import ProposedAction


VALID_ACTIONS = {"work", "save", "cooperate"}


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
        return _validate_action(action, neighbors, metadata)

    # Strategy 2: Regex extraction
    action = _try_regex_json(raw_text)
    if action:
        metadata["parse_method"] = "regex_json"
        metadata["parse_success"] = True
        return _validate_action(action, neighbors, metadata)

    # Strategy 3: Keyword fallback
    action = _try_keyword_fallback(raw_text, neighbors)
    if action:
        metadata["parse_method"] = "keyword_fallback"
        metadata["parse_success"] = True
        return action, metadata

    metadata["parse_error"] = "All parsing strategies failed"
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
    """Last resort: infer action from keywords in the text."""
    text_lower = text.lower()

    if "cooperate" in text_lower or "help" in text_lower or "share" in text_lower:
        target = neighbors[0] if neighbors else None
        return ProposedAction(
            action_type="cooperate",
            target_agent_id=target,
            amount=5.0,
            reasoning_summary="[parsed from keywords: cooperation detected]",
            confidence=0.3,
        )

    if "save" in text_lower or "conserve" in text_lower or "preserve" in text_lower:
        return ProposedAction(
            action_type="save",
            amount=5.0,
            reasoning_summary="[parsed from keywords: saving detected]",
            confidence=0.3,
        )

    if "work" in text_lower or "earn" in text_lower or "labor" in text_lower:
        return ProposedAction(
            action_type="work",
            amount=10.0,
            reasoning_summary="[parsed from keywords: work detected]",
            confidence=0.3,
        )

    return None


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

    # Clamp amount
    if amount is not None:
        try:
            amount = float(amount)
            amount = max(0.0, min(amount, 20.0))
        except (ValueError, TypeError):
            amount = 5.0
    else:
        amount = 10.0 if action_type == "work" else 5.0

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
