"""Tests for ReportAgent and _TrackerTools (analysis/react_report_agent.py)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from analysis.react_report_agent import ReportAgent, _TrackerTools


@pytest.fixture()
def tracker_with_parquet(tmp_path):
    """Write a minimal experiment parquet and return a _TrackerTools pointing at it."""
    df = pd.DataFrame(
        {
            "experiment_id": ["exp_llm_s1", "exp_llm_s2", "exp_random_s1"],
            "policy_type": ["llm", "llm", "random"],
            "seed": [1, 2, 1],
            "wealth_mean": [55.0, 60.0, 40.0],
            "wealth_gini": [0.3, 0.35, 0.2],
            "stress_mean": [0.4, 0.38, 0.5],
        }
    )
    parquet_path = tmp_path / "experiment_index.parquet"
    df.to_parquet(parquet_path, index=False)
    return _TrackerTools(index_path=str(parquet_path))


def test_tracker_tools_policy_comparison_offline(tracker_with_parquet):
    result = tracker_with_parquet.policy_comparison()
    assert isinstance(result, str)
    assert "llm" in result or "random" in result


def test_tracker_tools_seed_variance_offline(tracker_with_parquet):
    result = tracker_with_parquet.seed_variance("llm")
    assert isinstance(result, str)
    assert "llm" in result or "55" in result or "60" in result


def test_tracker_tools_panorama_search_offline(tracker_with_parquet):
    result = tracker_with_parquet.panorama_search("wealth_mean")
    assert "PANORAMA" in result


def test_tracker_tools_run_sql_blocks_non_select(tracker_with_parquet):
    result = tracker_with_parquet.run_sql("DROP TABLE experiments")
    assert result.startswith("ERROR:")


def test_tracker_tools_run_sql_select_works(tracker_with_parquet):
    result = tracker_with_parquet.run_sql("SELECT COUNT(*) AS n FROM experiments")
    assert "3" in result or "n" in result


def test_tracker_tools_list_experiments_offline(tracker_with_parquet):
    result = tracker_with_parquet.list_experiments(limit=5)
    assert "exp_llm" in result or "exp_random" in result


def test_parse_action_text_style():
    text = "Thought: I need data.\nAction: policy_comparison\nArgs: {}"
    result = ReportAgent._parse_action(text)
    assert result is not None
    tool_name, args = result
    assert tool_name == "policy_comparison"
    assert args == {}


def test_parse_action_xml_style():
    text = 'Thought: compare seeds.\n<tool_call>{"name": "seed_variance", "parameters": {"policy": "llm"}}</tool_call>'
    result = ReportAgent._parse_action(text)
    assert result is not None
    tool_name, args = result
    assert tool_name == "seed_variance"
    assert args == {"policy": "llm"}


def test_parse_action_returns_none_when_no_action():
    result = ReportAgent._parse_action("Just a plain text response with no tool calls.")
    assert result is None


def test_extract_final_answer():
    text = "Thought: done.\nFinal Answer:\n# My Report\n\nContent here."
    result = ReportAgent._extract_final_answer(text)
    assert result == "# My Report\n\nContent here."


def test_report_agent_returns_final_answer_from_mock(tmp_path):
    """ReportAgent should return the text after 'Final Answer:' when the mock LLM returns it."""
    df = pd.DataFrame(
        {
            "experiment_id": ["exp_llm_s1"],
            "policy_type": ["llm"],
            "seed": [1],
            "wealth_mean": [55.0],
            "wealth_gini": [0.3],
            "stress_mean": [0.4],
        }
    )
    parquet_path = tmp_path / "experiment_index.parquet"
    df.to_parquet(parquet_path, index=False)

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Thought: done.\nFinal Answer:\nMock report content."

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("analysis.react_report_agent.ReportAgent.__init__", lambda self, **kw: None):
        agent = ReportAgent.__new__(ReportAgent)
        agent.model = "gpt-4o-mini"
        agent.max_iterations = 5
        agent.temperature = 0.3
        agent.tools = _TrackerTools(index_path=str(parquet_path))
        agent._client = mock_client

    result = agent.generate_report("Summarise results")
    assert result == "Mock report content."
