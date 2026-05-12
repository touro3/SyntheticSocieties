"""
ReACT-style report agent for cross-experiment synthesis.

Implements the Reasoning + Acting loop from Yao et al. (2022):
  THINK → ACT (tool call) → OBSERVE → ... → THINK → FINAL ANSWER

Tools expose the DuckDB experiment tracker so the agent can synthesise
insights across all completed runs without pre-scripted metric scripts.

Usage (OpenAI-compatible endpoint required):
    from analysis.react_report_agent import ReportAgent

    agent = ReportAgent(api_key="...", base_url="...")
    report = agent.generate_report("Compare grounded vs ungrounded LLM performance")
    print(report)

    # Or run as CLI:
    # python analysis/react_report_agent.py --query "Summarise Condition A vs B"
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
import time
from pathlib import Path
from typing import Any, Optional

import duckdb

logger = logging.getLogger(__name__)

# ── Tool registry ─────────────────────────────────────────────────────────────

DEFAULT_INDEX = "tracker/experiment_index.parquet"
MAX_ITERATIONS = 8  # guard against infinite loops
TOOL_CALL_PAUSE = 0.2  # seconds between tool calls
_MAX_CONFLICTS = 2  # max consecutive tool+FinalAnswer conflicts before downgrade
_MAX_LOOP_TOKENS = 800  # max completion tokens per ReACT iteration (cost guard)
_MAX_HISTORY_TURNS = 12  # keep only the N most-recent turns to cap context growth


class _TrackerTools:
    """Retrieval tools backed by the DuckDB experiment index."""

    def __init__(self, index_path: str = DEFAULT_INDEX):
        self._index = index_path

    def _conn(self) -> duckdb.DuckDBPyConnection:
        path = Path(self._index)
        if not path.exists():
            raise FileNotFoundError(f"Experiment index not found: {path}. Run a simulation first to populate it.")
        conn = duckdb.connect()
        if path.suffix != ".parquet":
            raise ValueError(f"Experiment index must be a .parquet file: {path}")
        safe_path = str(path.resolve()).replace("'", "''")
        conn.execute(f"CREATE VIEW experiments AS SELECT * FROM read_parquet('{safe_path}')")
        return conn

    # ── Individual tools ──────────────────────────────────────────────────────

    def policy_comparison(self) -> str:
        """Aggregate mean wealth, Gini, and stress by policy type."""
        try:
            conn = self._conn()
            df = conn.execute("""
                SELECT
                    policy_type,
                    COUNT(*) AS n_runs,
                    ROUND(AVG(wealth_mean), 3)   AS avg_wealth,
                    ROUND(STDDEV(wealth_mean), 3) AS std_wealth,
                    ROUND(AVG(wealth_gini), 3)   AS avg_gini,
                    ROUND(AVG(stress_mean), 3)   AS avg_stress
                FROM experiments
                GROUP BY policy_type
                ORDER BY avg_wealth DESC
            """).fetchdf()
            return df.to_string(index=False)
        except Exception as exc:
            return f"ERROR: {exc}"

    def seed_variance(self, policy: str = "llm") -> str:
        """Show per-seed wealth variance for a given policy."""
        try:
            conn = self._conn()
            df = conn.execute(
                """
                SELECT experiment_id, seed, wealth_mean, wealth_gini, stress_mean
                FROM experiments
                WHERE policy_type = ?
                ORDER BY seed
            """,
                [policy],
            ).fetchdf()
            if df.empty:
                return f"No experiments found for policy='{policy}'."
            return df.to_string(index=False)
        except Exception as exc:
            return f"ERROR: {exc}"

    def ablation_comparison(self) -> str:
        """Compare ablation levels (experiment IDs starting with 'ablation_')."""
        try:
            conn = self._conn()
            df = conn.execute("""
                SELECT
                    REGEXP_EXTRACT(experiment_id, 'ablation_([^_]+(?:_[^_]+)?)', 1) AS ablation_mode,
                    COUNT(*) AS n_runs,
                    ROUND(AVG(wealth_mean), 3)    AS avg_wealth,
                    ROUND(STDDEV(wealth_mean), 3) AS std_wealth,
                    ROUND(AVG(wealth_gini), 3)    AS avg_gini,
                    ROUND(AVG(stress_mean), 3)    AS avg_stress
                FROM experiments
                WHERE experiment_id LIKE 'ablation_%'
                GROUP BY ablation_mode
                ORDER BY avg_wealth DESC
            """).fetchdf()
            if df.empty:
                return "No ablation experiments found."
            return df.to_string(index=False)
        except Exception as exc:
            return f"ERROR: {exc}"

    def experiment_detail(self, experiment_id: str) -> str:
        """Return all columns for a specific experiment by ID."""
        try:
            conn = self._conn()
            df = conn.execute("SELECT * FROM experiments WHERE experiment_id = ?", [experiment_id]).fetchdf()
            if df.empty:
                return f"Experiment '{experiment_id}' not found."
            return df.T.to_string(header=False)
        except Exception as exc:
            return f"ERROR: {exc}"

    def run_sql(self, sql: str) -> str:
        """Execute an arbitrary read-only SQL query against the experiment index.

        Only SELECT statements are allowed. The table is named 'experiments'.
        """
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT"):
            return "ERROR: Only SELECT statements are permitted."
        try:
            conn = self._conn()
            df = conn.execute(sql).fetchdf()
            if df.empty:
                return "(empty result set)"
            return df.to_string(index=False)
        except Exception as exc:
            return f"ERROR: {exc}"

    def list_experiments(self, limit: int = 20) -> str:
        """List recent experiments with key metadata."""
        try:
            conn = self._conn()
            df = conn.execute(f"""
                SELECT experiment_id, policy_type, seed, wealth_mean, wealth_gini
                FROM experiments
                ORDER BY experiment_id DESC
                LIMIT {int(limit)}
            """).fetchdf()
            if df.empty:
                return "No experiments found."
            return df.to_string(index=False)
        except Exception as exc:
            return f"ERROR: {exc}"

    def panorama_search(self, metric: str = "wealth_mean", top_n: int = 10) -> str:
        """Comprehensive historical search across all experiments (MiroFish panorama pattern).

        Unlike ``policy_comparison`` which averages across all runs, panorama_search
        surfaces the full distribution: best runs, worst runs, outliers, and the
        temporal arc of a metric.  Includes experiments of every status — the
        "panorama" — rather than just the latest aggregate snapshot.

        Args:
            metric: Column to rank experiments by (default: wealth_mean).
            top_n:  How many top/bottom experiments to surface.
        """
        safe_metric = re.sub(r"[^\w]", "", metric)  # prevent SQL injection
        try:
            conn = self._conn()
            # Check the column exists
            cols = [r[0] for r in conn.execute("DESCRIBE experiments").fetchall()]
            if safe_metric not in cols:
                safe_metric = "wealth_mean"

            top_df = conn.execute(f"""
                SELECT experiment_id, policy_type, seed,
                       ROUND({safe_metric}, 3) AS metric_value,
                       ROUND(wealth_gini, 3) AS gini, ROUND(stress_mean, 4) AS stress
                FROM experiments
                ORDER BY {safe_metric} DESC
                LIMIT {int(top_n)}
            """).fetchdf()

            bot_df = conn.execute(f"""
                SELECT experiment_id, policy_type, seed,
                       ROUND({safe_metric}, 3) AS metric_value,
                       ROUND(wealth_gini, 3) AS gini, ROUND(stress_mean, 4) AS stress
                FROM experiments
                ORDER BY {safe_metric} ASC
                LIMIT {int(top_n)}
            """).fetchdf()

            dist_df = conn.execute(f"""
                SELECT
                    policy_type,
                    COUNT(*) AS n,
                    ROUND(MIN({safe_metric}), 3)    AS min_val,
                    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {safe_metric}), 3) AS p25,
                    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {safe_metric}), 3) AS median,
                    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {safe_metric}), 3) AS p75,
                    ROUND(MAX({safe_metric}), 3)    AS max_val
                FROM experiments
                GROUP BY policy_type
                ORDER BY median DESC
            """).fetchdf()

            return (
                f"=== PANORAMA: {safe_metric} ===\n\n"
                f"Full distribution by policy:\n{dist_df.to_string(index=False)}\n\n"
                f"Top {top_n} experiments:\n{top_df.to_string(index=False)}\n\n"
                f"Bottom {top_n} experiments:\n{bot_df.to_string(index=False)}"
            )
        except Exception as exc:
            return f"ERROR: {exc}"

    def trend_analysis(self, metric: str = "wealth_mean", window: int = 3) -> str:
        """Analyse how a metric evolves across experiments ordered by experiment_id.

        Computes a rolling average to smooth noise and detect regime changes.
        Useful for identifying whether performance improves over seeds or
        degrades over time (e.g., action collapse onset).

        Args:
            metric: Column to track (default: wealth_mean).
            window: Rolling average window size.
        """
        safe_metric = re.sub(r"[^\w]", "", metric)
        try:
            conn = self._conn()
            cols = [r[0] for r in conn.execute("DESCRIBE experiments").fetchall()]
            if safe_metric not in cols:
                safe_metric = "wealth_mean"

            df = conn.execute(f"""
                SELECT
                    experiment_id, policy_type, seed,
                    ROUND({safe_metric}, 3) AS value
                FROM experiments
                ORDER BY experiment_id
            """).fetchdf()

            if df.empty:
                return "(no experiments)"

            # Rolling average per policy
            parts: list[str] = []
            for policy, group in df.groupby("policy_type"):
                group = group.sort_values("experiment_id")
                group = group.copy()
                group["rolling_avg"] = group["value"].rolling(window, min_periods=1).mean().round(3)
                first_avg = group["rolling_avg"].iloc[0]
                last_avg = group["rolling_avg"].iloc[-1]
                trend = "improving" if last_avg > first_avg else "declining"
                parts.append(
                    f"Policy '{policy}': {len(group)} experiments, "
                    f"trend={trend}, latest_avg={last_avg}\n"
                    f"{group[['experiment_id', 'value', 'rolling_avg']].to_string(index=False)}"
                )
            return f"=== TREND ANALYSIS: {safe_metric} (window={window}) ===\n\n" + "\n\n".join(parts)
        except Exception as exc:
            return f"ERROR: {exc}"

    def insight_forge(self, question: str, client: Any, model: str, temperature: float = 0.3) -> str:
        """Deep analysis via sub-question decomposition (MiroFish insight_forge pattern).

        Breaks a complex research question into focused sub-questions, answers
        each one individually using the available tracker tools, then synthesises
        all partial answers into a final insight string.

        This method requires a live LLM client and is called by the ReportAgent
        when the 'insight_forge' tool is dispatched.

        Args:
            question:    Complex research question to decompose.
            client:      OpenAI-compatible client instance.
            model:       Model name to use for decomposition and synthesis.
            temperature: Sampling temperature.
        """
        # ── Stage A: decompose into sub-questions ─────────────────────────────
        decompose_prompt = textwrap.dedent(f"""
            You are a research analyst for the BGF simulation project.
            Break the following research question into 3-5 focused sub-questions
            that can each be answered by querying experiment data.
            Return ONLY a JSON array of strings, e.g.:
            ["sub-question 1", "sub-question 2", "sub-question 3"]

            Question: {question}
        """).strip()

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": decompose_prompt}],
                temperature=temperature,
                max_tokens=150,  # expects a JSON array of 3-5 short strings (~40-80 tokens)
            )
            raw = resp.choices[0].message.content or "[]"
            # Extract JSON array robustly
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            sub_questions: list[str] = json.loads(match.group(0)) if match else [question]
        except Exception:
            sub_questions = [question]

        # ── Stage B: answer each sub-question via tools ───────────────────────
        partial_answers: list[str] = []
        for sq in sub_questions[:5]:  # cap at 5 sub-questions
            sq_lower = sq.lower()
            # Route sub-question to the most relevant tool
            if any(k in sq_lower for k in ("ablat", "condition", "grounding", "rag")):
                obs = self.ablation_comparison()
            elif any(k in sq_lower for k in ("seed", "variance", "replicate", "reproducib")):
                obs = self.seed_variance()
            elif any(k in sq_lower for k in ("policy", "baseline", "random", "template")):
                obs = self.policy_comparison()
            elif any(k in sq_lower for k in ("experiment", "run", "specific")):
                obs = self.list_experiments(limit=10)
            else:
                # Generic: run a best-effort SQL query
                obs = self.policy_comparison()
            partial_answers.append(f"Sub-question: {sq}\nData:\n{obs}")

        # ── Stage C: synthesise partial answers ───────────────────────────────
        # Truncate each observation before injecting — DataFrame dumps can be
        # thousands of chars each; 5 × 800 chars keeps the synthesis prompt
        # under ~1,500 tokens regardless of experiment count.
        _MAX_OBS_CHARS = 800
        truncated_answers = [
            pa[:_MAX_OBS_CHARS] + ("…" if len(pa) > _MAX_OBS_CHARS else "")
            for pa in partial_answers
        ]
        synthesis_prompt = textwrap.dedent(f"""
            You are a research analyst for the BGF simulation project.
            Synthesise the following data observations into a concise, precise insight.
            Use specific numbers. Be publication-quality but under 200 words.

            Original question: {question}

            Observations:
            {chr(10).join(f"--- {i + 1} ---{chr(10)}{pa}" for i, pa in enumerate(truncated_answers))}
        """).strip()

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=temperature,
                max_tokens=350,  # 200 words ≈ 270 tokens + headroom
            )
            return resp.choices[0].message.content or "(synthesis failed)"
        except Exception as exc:
            return f"(synthesis error: {exc})\n\n" + "\n\n".join(partial_answers)

    def _insight_forge_fallback(self, question: str) -> str:
        """Rule-based insight synthesis when no LLM client is available.

        Produces a structured data dump that assembles all available tool
        outputs so the caller still gets useful context even in offline /
        no-API environments.
        """
        parts = [
            f"Question: {question}\n",
            "--- Policy Overview ---",
            self.policy_comparison(),
            "\n--- Ablation Overview ---",
            self.ablation_comparison(),
            "\n--- Trend ---",
            self.trend_analysis(),
        ]
        return "\n".join(parts)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    TOOL_DESCRIPTIONS = {
        "policy_comparison": "Compare mean wealth, Gini, and stress across all policy types. No arguments.",
        "seed_variance": "Show per-seed variance for a policy. Argument: policy (str, default='llm').",
        "ablation_comparison": "Compare ablation conditions. No arguments.",
        "experiment_detail": "Full details of one experiment. Argument: experiment_id (str).",
        "run_sql": "Run a custom SELECT query against the 'experiments' table. Argument: sql (str).",
        "list_experiments": "List recent experiments. Argument: limit (int, default=20).",
        "panorama_search": (
            "Full distribution analysis of a metric across ALL experiments (best/worst/percentiles). "
            "Use when you need the complete picture, not just averages. "
            "Arguments: metric (str, default='wealth_mean'), top_n (int, default=10)."
        ),
        "insight_forge": (
            "Deep analysis via sub-question decomposition. Best for complex, multi-faceted questions. "
            "Argument: question (str). Automatically decomposes into sub-questions, answers each, "
            "then synthesises. Use this when a simple tool call won't capture the full answer."
        ),
        "trend_analysis": (
            "Track how a metric evolves across experiments over time / seeds. "
            "Computes rolling averages to detect regime changes (e.g., action collapse onset). "
            "Arguments: metric (str, default='wealth_mean'), window (int, default=3)."
        ),
    }

    def call(self, tool_name: str, args: dict[str, Any], _client: Any = None, _model: str = "gpt-4o-mini") -> str:
        """Dispatch a tool call by name with keyword arguments."""
        dispatch = {
            "policy_comparison": lambda: self.policy_comparison(),
            "seed_variance": lambda: self.seed_variance(**args),
            "ablation_comparison": lambda: self.ablation_comparison(),
            "experiment_detail": lambda: self.experiment_detail(**args),
            "run_sql": lambda: self.run_sql(**args),
            "list_experiments": lambda: self.list_experiments(**args),
            "panorama_search": lambda: self.panorama_search(**args),
            "insight_forge": lambda: self.insight_forge(
                question=args.get("question", ""),
                client=_client,
                model=_model,
            ),
            "trend_analysis": lambda: self.trend_analysis(**args),
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return f"ERROR: Unknown tool '{tool_name}'. Available: {list(dispatch)}"
        if tool_name == "insight_forge" and _client is None:
            return self._insight_forge_fallback(args.get("question", ""))
        return fn()


# ── System prompt ─────────────────────────────────────────────────────────────


def _build_system_prompt(tools: _TrackerTools) -> str:
    tool_docs = "\n".join(f"  - {name}: {desc}" for name, desc in tools.TOOL_DESCRIPTIONS.items())
    return textwrap.dedent(f"""
        You are a research analyst for the BGF (Behavioral Grounding Framework) simulation project.
        Your job is to synthesise insights across simulation experiments by querying the DuckDB
        experiment tracker and produce a concise, publication-quality report.

        You have access to these retrieval tools:
        {tool_docs}

        ## Format for reasoning and tool use

        Use the following exact format in your responses:

        Thought: <your reasoning about what to do next>
        Action: <tool_name>
        Args: <JSON object with arguments, or {{}} if none>
        Observation: <tool result will be inserted here by the system>

        Repeat Thought/Action/Args/Observation as many times as needed (max {MAX_ITERATIONS} iterations).
        When you have enough information, end with:

        Thought: I have gathered enough data to write the final report.
        Final Answer:
        <your complete markdown report here>

        ## Report style
        - Use markdown headings, tables, and bullet lists.
        - Cite specific numbers from tool observations.
        - Compare Condition A (ungrounded LLM, policy_type='llm' without RAG) vs Condition B
          (grounded LLM with ESS RAG) when relevant.
        - Highlight statistical significance and effect sizes where available.
        - Be concise but precise — this is for a research paper.
    """).strip()


# ── ReACT loop ────────────────────────────────────────────────────────────────


class ReportAgent:
    """
    ReACT-style report synthesis agent backed by the BGF DuckDB tracker.

    Requires an OpenAI-compatible chat endpoint (local or API). Defaults
    to GPT-4o-mini but works with any model that follows the chat format.

    Args:
        api_key:    API key for the LLM endpoint.
        base_url:   Base URL for OpenAI-compatible endpoint.
                    Defaults to the official OpenAI API.
        model:      Model name to use.
        index_path: Path to the experiment index parquet file.
        max_iterations: Maximum ReACT loop iterations before forcing answer.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        index_path: str = DEFAULT_INDEX,
        max_iterations: int = MAX_ITERATIONS,
        temperature: float = 0.3,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.tools = _TrackerTools(index_path)

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError as exc:
            raise ImportError("openai package is required for ReportAgent. Install with: pip install openai") from exc

    # ── Parsing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_action(text: str) -> Optional[tuple[str, dict]]:
        """Extract (tool_name, args_dict) from the assistant's response.

        Supports three formats in priority order (MiroFish multi-format pattern):
          1. XML-style:  <tool_call>{"name": "...", "parameters": {...}}</tool_call>
          2. Text-style: Action: <name> / Args: {...}  (original BGF format)
          3. Bare JSON:  {"name": "...", "parameters": {...}} at end of response
        """
        valid_tools = set(_TrackerTools.TOOL_DESCRIPTIONS.keys())

        # Format 1: XML-style <tool_call> (MiroFish primary format)
        xml_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        if xml_match:
            try:
                call_data = json.loads(xml_match.group(1))
                tool_name = call_data.get("name") or call_data.get("tool")
                if tool_name in valid_tools:
                    args = call_data.get("parameters") or call_data.get("params") or call_data.get("args") or {}
                    return tool_name, (args if isinstance(args, dict) else {})
            except json.JSONDecodeError:
                pass

        # Format 2: Text-style Action:/Args: (original BGF format)
        action_match = re.search(r"Action:\s*(\w+)", text)
        args_match = re.search(r"Args:\s*(\{.*?\})", text, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1).strip()
            args: dict = {}
            if args_match:
                try:
                    args = json.loads(args_match.group(1))
                except json.JSONDecodeError:
                    pass
            return tool_name, args

        # Format 3: Bare JSON at end of response (MiroFish fallback)
        stripped = text.strip()
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        bare_match = re.search(json_pattern, stripped, re.DOTALL)
        if not bare_match and stripped.startswith("{") and stripped.endswith("}"):
            bare_match = re.match(r"(\{.*\})", stripped, re.DOTALL)
        if bare_match:
            try:
                call_data = json.loads(bare_match.group(1))
                tool_name = call_data.get("name") or call_data.get("tool")
                if tool_name in valid_tools:
                    args = call_data.get("parameters") or call_data.get("params") or call_data.get("args") or {}
                    return tool_name, (args if isinstance(args, dict) else {})
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _extract_final_answer(text: str) -> Optional[str]:
        """Extract everything after 'Final Answer:' marker."""
        match = re.search(r"Final Answer:\s*\n?(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    # ── Main loop ─────────────────────────────────────────────────────────────

    def generate_report(
        self,
        query: str,
        verbose: bool = False,
    ) -> str:
        """
        Run the ReACT loop to answer `query` and return a markdown report.

        Args:
            query:   Natural-language research question.
            verbose: If True, print each Thought/Action/Observation step.

        Returns:
            Markdown string containing the synthesised report.
        """
        system_prompt = _build_system_prompt(self.tools)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        conflict_retries = 0  # consecutive conflict counter (MiroFish pattern)

        for iteration in range(self.max_iterations):
            # Trim history: always keep system + user (first 2), then last N turns.
            # Prevents unbounded context growth that multiplies cost each iteration.
            if len(messages) > 2 + _MAX_HISTORY_TURNS:
                messages = messages[:2] + messages[-(  _MAX_HISTORY_TURNS):]
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=_MAX_LOOP_TOKENS,
                    stop=["Observation:"],
                )
                assistant_text = (response.choices[0].message.content if response and response.choices else "") or ""
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "rate_limit" in err_str.lower() or "rate limit" in err_str.lower():
                    _wait = 2 ** min(iteration, 4)  # 1, 2, 4, 8, 16s — caps at 16s
                    logger.warning(
                        "ReACT iteration %d: 429 rate limit — sleeping %ds before continuing.",
                        iteration + 1,
                        _wait,
                    )
                    time.sleep(_wait)
                else:
                    logger.warning("ReACT iteration %d: LLM call failed (%s).", iteration + 1, exc)
                assistant_text = ""

            # Handle empty / None response (MiroFish None-guard pattern)
            if not assistant_text.strip():
                logger.warning("ReACT iteration %d: empty response.", iteration + 1)
                if iteration < self.max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(empty response)"})
                    messages.append({"role": "user", "content": "Please continue."})
                    continue
                break

            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
                print(assistant_text)

            final = self._extract_final_answer(assistant_text)
            parsed = self._parse_action(assistant_text)

            # Conflict: both tool call and Final Answer in same response
            # (MiroFish conflict resolution pattern)
            if parsed is not None and final is not None:
                conflict_retries += 1
                logger.warning(
                    "ReACT iteration %d: conflict — tool call and Final Answer in same response (conflict #%d/%d).",
                    iteration + 1,
                    conflict_retries,
                    _MAX_CONFLICTS,
                )
                if conflict_retries <= _MAX_CONFLICTS:
                    messages.append({"role": "assistant", "content": assistant_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your response contained both a tool call and a Final Answer. "
                                "Please do only one per reply: either call a tool OR write your "
                                "Final Answer — not both."
                            ),
                        }
                    )
                    continue
                else:
                    # Downgrade after repeated conflicts: prefer Final Answer
                    logger.warning("ReACT: conflict threshold exceeded — using Final Answer directly.")
                    conflict_retries = 0
                    return final

            # No conflict — proceed normally
            if final:
                return final

            if parsed is None:
                logger.warning(
                    "ReACT iteration %d: no Action found, using response as final answer.",
                    iteration + 1,
                )
                return assistant_text.strip()

            tool_name, args = parsed
            conflict_retries = 0  # reset on clean tool call
            observation = self.tools.call(tool_name, args, _client=self._client, _model=self.model)

            if verbose:
                print(f"Observation ({tool_name}): {observation[:500]}...")

            # Truncate large DataFrame observations before adding to history —
            # panorama_search/trend_analysis can return thousands of chars which
            # are re-sent on every subsequent iteration, inflating costs linearly.
            _MAX_OBS_CHARS = 2000
            obs_trimmed = observation[:_MAX_OBS_CHARS] + (
                "\n…[truncated]" if len(observation) > _MAX_OBS_CHARS else ""
            )
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation:\n{obs_trimmed}\n\nContinue.",
                }
            )

            time.sleep(TOOL_CALL_PAUSE)

        # Max iterations reached — ask for a final answer with current context
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the maximum number of tool calls. "
                    "Write your Final Answer now based on the information gathered."
                ),
            }
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1200,
            )
            final_text = (response.choices[0].message.content if response and response.choices else "") or ""
        except Exception as exc:
            logger.error("ReACT final synthesis call failed: %s", exc)
            final_text = ""
        final = self._extract_final_answer(final_text)
        return final or final_text.strip()

    def generate_sections(
        self,
        outline: list[str],
        context_query: str = "",
        verbose: bool = False,
    ) -> dict[str, str]:
        """
        Generate a multi-section report by running a separate ReACT loop per
        section. Mirrors MiroFish's section-by-section generation strategy.

        Args:
            outline:       List of section titles to generate.
            context_query: Optional global context injected into each section query.
            verbose:       Print progress.

        Returns:
            Dict mapping section title → markdown content.
        """
        sections: dict[str, str] = {}
        for title in outline:
            query = f"Write the '{title}' section of the BGF research report."
            if context_query:
                query += f" Global context: {context_query}"
            if verbose:
                print(f"\n=== Generating section: {title} ===")
            sections[title] = self.generate_report(query, verbose=verbose)
        return sections

    def assemble_report(
        self,
        sections: dict[str, str],
        title: str = "BGF Simulation Analysis Report",
    ) -> str:
        """Assemble section dict into a single markdown document."""
        parts = [f"# {title}\n"]
        for section_title, content in sections.items():
            parts.append(f"## {section_title}\n\n{content}")
        return "\n\n---\n\n".join(parts)


# ── CLI entry point ───────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="BGF ReACT report agent — synthesise insights from DuckDB tracker")
    parser.add_argument("--query", default="Summarise the key findings across all experiments.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--out", default=None, help="Save report to file path")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--sections",
        nargs="*",
        default=None,
        help="Generate multi-section report with given section titles",
    )
    args = parser.parse_args()

    agent = ReportAgent(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        index_path=args.index,
    )

    if args.sections:
        sections = agent.generate_sections(args.sections, context_query=args.query, verbose=args.verbose)
        report = agent.assemble_report(sections)
    else:
        report = agent.generate_report(args.query, verbose=args.verbose)

    print("\n" + "=" * 72)
    print(report)
    print("=" * 72)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    _cli()
