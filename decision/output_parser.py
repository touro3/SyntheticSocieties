"""
Output parser for extracting structured ProposedAction from LLM text.

Handles JSON extraction, validation, and fallback when the LLM
produces malformed output.
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from decision.constants import (
    DEFAULT_COOPERATE_AMOUNT,
    DEFAULT_KEYWORD_CONFIDENCE,
    DEFAULT_WORK_AMOUNT,
    MAX_ACTION_AMOUNT,
)
from decision.schemas import ProposedAction

VALID_ACTIONS = {"work", "save", "cooperate", "communicate"}

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
        if not neighbors:
            # cooperate requires a target; fall back to work when no neighbors available
            return ProposedAction(
                action_type="work",
                amount=DEFAULT_WORK_AMOUNT,
                reasoning_summary="[parsed from keywords: cooperation detected but no neighbors — defaulting to work]",
                confidence=DEFAULT_KEYWORD_CONFIDENCE,
            )
        return ProposedAction(
            action_type="cooperate",
            target_agent_id=neighbors[0],
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


def _repair_json(text: str) -> str:
    """Attempt lightweight repairs on malformed JSON strings.

    Handles the most common LLM output defects:
    1. Trailing commas before closing brace/bracket.
    2. Embedded raw newlines inside string values (MiroFish pattern).
    3. Control characters that break JSON parsing (MiroFish pattern).
    4. Unclosed strings / braces / brackets (appends missing closers).

    This is intentionally conservative — it only fixes patterns that are
    safe to repair without changing the semantic content.
    """
    # Strip leading/trailing whitespace and markdown code fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        json.loads(text)
        return text  # already valid, no repair needed
    except json.JSONDecodeError:
        pass

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Normalize embedded newlines inside JSON string values (MiroFish pattern).
    # Raw \n / \r inside a JSON string are illegal and break parsers.
    try:
        def _fix_str_newlines(m: re.Match) -> str:
            s = m.group(0)
            s = s.replace('\n', ' ').replace('\r', ' ')
            return re.sub(r'  +', ' ', s)
        text = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', _fix_str_newlines, text)
    except re.error:
        pass  # regex can fail on pathological/truncated input

    # Strip control characters that prevent JSON parsing (MiroFish pattern).
    # Preserve \t (0x09), \n (0x0a), \r (0x0d) — valid JSON whitespace.
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)

    # Balance unmatched braces/brackets.
    # Close any dangling open string before appending closers (MiroFish pattern).
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_braces > 0 and text and text[-1] not in '",}]':
        text += '"'
    text += "}" * max(open_braces, 0)
    text += "]" * max(open_brackets, 0)

    return text


def _field_extract_action(
    text: str,
    neighbors: Optional[list[str]] = None,
) -> Optional[ProposedAction]:
    """Ultimate fallback: extract ProposedAction fields via targeted regex.

    Called after all JSON-repair strategies are exhausted.  Mirrors
    MiroFish's ``_try_fix_json`` field-level extraction: even when the
    outer JSON structure is irreparable, individual field values can
    usually be recovered with targeted patterns.
    """
    action_match = re.search(r'"action_type"\s*:\s*"(\w+)"', text)
    if not action_match:
        return None
    action_type = action_match.group(1).strip().lower()
    if action_type not in VALID_ACTIONS:
        return None

    amount_match = re.search(r'"amount"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    conf_match = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    reason_match = re.search(r'"reasoning_summary"\s*:\s*"([^"]*)"', text)
    target_match = re.search(r'"target_agent_id"\s*:\s*"([^"]+)"', text)

    amount = float(amount_match.group(1)) if amount_match else (
        DEFAULT_WORK_AMOUNT if action_type == "work" else DEFAULT_COOPERATE_AMOUNT
    )
    confidence = float(conf_match.group(1)) if conf_match else DEFAULT_KEYWORD_CONFIDENCE
    reasoning = reason_match.group(1) if reason_match else "[field extraction fallback]"
    target = target_match.group(1) if target_match else None

    if action_type in ("cooperate", "communicate"):
        if target and neighbors and target not in neighbors:
            target = neighbors[0] if neighbors else None
        elif target is None and neighbors:
            target = neighbors[0]

    try:
        return ProposedAction(
            action_type=action_type,
            target_agent_id=target,
            amount=max(0.0, min(float(amount), MAX_ACTION_AMOUNT)),
            confidence=max(0.0, min(float(confidence), 1.0)),
            reasoning_summary=reasoning[:500],
        )
    except Exception:
        return None


def parse_llm_output_with_retry(
    raw_text: str,
    neighbors: Optional[list[str]] = None,
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> tuple[Optional[ProposedAction], dict]:
    """Parse LLM output with JSON repair and exponential backoff retry.

    On each failed attempt, applies _repair_json() to the text before
    retrying the full parse pipeline. Uses exponential backoff (base_delay *
    2^attempt) between attempts to avoid hammering a struggling LLM backend.

    This is the recommended entry point for production use.  The lower-level
    ``parse_llm_output`` remains available for testing and cases where retry
    overhead is undesirable.

    Args:
        raw_text:    Raw text output from the LLM.
        neighbors:   List of valid neighbor agent IDs for validation.
        max_attempts: Maximum parse attempts (default 3, matching MiroFish).
        base_delay:   Base delay in seconds for exponential backoff (default 2s).

    Returns:
        Tuple of (ProposedAction or None, parse_metadata dict).
    """
    last_action: Optional[ProposedAction] = None
    last_metadata: dict = {}
    candidate = raw_text

    for attempt in range(max_attempts):
        action, metadata = parse_llm_output(candidate, neighbors)

        if action is not None:
            metadata["retry_attempts"] = attempt
            _parse_stats["retry_success"] = _parse_stats.get("retry_success", 0) + 1
            return action, metadata

        last_action = action
        last_metadata = metadata

        if attempt < max_attempts - 1:
            # Apply JSON repair before next attempt
            repaired = _repair_json(candidate)
            if repaired != candidate:
                candidate = repaired
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)

    # Ultimate fallback: field-level regex extraction (MiroFish pattern).
    # Even when JSON structure is irreparable, individual field values can
    # often be recovered with targeted patterns.
    field_action = _field_extract_action(raw_text, neighbors)
    if field_action is not None:
        last_metadata["parse_method"] = "field_extract"
        last_metadata["parse_success"] = True
        last_metadata["retry_attempts"] = max_attempts
        _parse_stats["field_extract"] = _parse_stats.get("field_extract", 0) + 1
        return field_action, last_metadata

    last_metadata["retry_attempts"] = max_attempts
    last_metadata["retry_exhausted"] = True
    _parse_stats["retry_exhausted"] = _parse_stats.get("retry_exhausted", 0) + 1
    return last_action, last_metadata


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

    # Validate cooperate/communicate has a valid target
    if action_type in ("cooperate", "communicate"):
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
