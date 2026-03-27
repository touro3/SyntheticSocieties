"""
Prompt and output logger for LLM decision auditing.

Stores every prompt + LLM response as JSONL for reproducibility
and analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class PromptLogger:
    """Log prompts and LLM outputs to a JSONL file."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def log(
        self,
        round_id: int,
        agent_id: str,
        prompt: str,
        raw_output: str,
        parsed_action: Optional[dict],
        latency_ms: float,
        parse_metadata: Optional[dict] = None,
        rag_context: Optional[dict] = None,
    ):
        """Append one prompt/output record to the JSONL file.

        Args:
            rag_context: Optional dict with RAG presence flags, e.g.
                {"sql_rag_present": True, "graph_rag_present": False}.
                Enables post-hoc verification that grounding was active.
        """
        record = {
            "round_id": round_id,
            "agent_id": agent_id,
            "prompt": prompt,
            "raw_output": raw_output,
            "parsed_action": parsed_action,
            "latency_ms": round(latency_ms, 2),
            "parse_metadata": {
                k: v for k, v in (parse_metadata or {}).items()
                if k != "raw_text"  # avoid duplication
            },
            "rag_context": rag_context,
        }

        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._count += 1

    @property
    def count(self) -> int:
        return self._count
