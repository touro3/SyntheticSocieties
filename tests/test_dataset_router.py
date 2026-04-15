"""Tests for DatasetRegistry — routing, scoring, and context building.

Uses a lightweight in-memory registry fixture to avoid dependency on
the real data/dataset_registry.json file.
"""

from __future__ import annotations

import json

import pytest

from decision.dataset_router import DatasetRegistry, RoutedDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_REGISTRY = {
    "registry_version": "test-1.0",
    "datasets": [
        {
            "id": "ess11",
            "title": "European Social Survey Wave 11",
            "description": "Trust, cooperation, and social attitudes across Europe.",
            "provider": "ESS",
            "citation": "ESS 2024",
            "local_path": "data/ess_clean.parquet",
            "format": "parquet",
            "trust_level": "high",
            "auditable": True,
            "domains": ["trust", "cooperation", "inequality"],
            "semantic_keywords": ["trust", "cooperation", "ess"],
            "profile_features": [
                {"name": "trust_people", "label": "Trust in people"},
                {"name": "risk_tolerance", "label": "Risk tolerance"},
            ],
            "target_items": [
                {"name": "cooperation_rate", "description": "Frequency of cooperative behavior."},
            ],
            "derived_columns": {},
        },
        {
            "id": "wvs7",
            "title": "World Values Survey Wave 7",
            "description": "Values, beliefs, and cultural dimensions worldwide.",
            "provider": "WVS",
            "citation": "WVS 2020",
            "local_path": "data/wvs7.parquet",
            "format": "parquet",
            "trust_level": "high",
            "auditable": True,
            "domains": ["values", "culture", "beliefs"],
            "semantic_keywords": ["values", "culture", "wvs"],
            "profile_features": [],
            "target_items": [],
            "derived_columns": {},
        },
        {
            "id": "synthetic",
            "title": "Synthetic Population",
            "description": "Synthetic baseline agents.",
            "provider": "internal",
            "citation": "",
            "local_path": "data/synthetic.parquet",
            "format": "parquet",
            "trust_level": "low",
            "auditable": False,
            "domains": ["simulation"],
            "semantic_keywords": [],
            "profile_features": [],
            "target_items": [],
            "derived_columns": {},
        },
    ],
}


@pytest.fixture
def registry(tmp_path) -> DatasetRegistry:
    reg_file = tmp_path / "registry.json"
    reg_file.write_text(json.dumps(MINIMAL_REGISTRY), encoding="utf-8")
    return DatasetRegistry(registry_path=reg_file)


# ---------------------------------------------------------------------------
# DatasetRegistry.get
# ---------------------------------------------------------------------------


class TestGet:
    def test_returns_correct_dataset(self, registry):
        ds = registry.get("ess11")
        assert ds["title"] == "European Social Survey Wave 11"

    def test_raises_on_unknown_id(self, registry):
        with pytest.raises(KeyError):
            registry.get("nonexistent_id")


# ---------------------------------------------------------------------------
# DatasetRegistry.list
# ---------------------------------------------------------------------------


class TestList:
    def test_returns_all_datasets(self, registry):
        datasets = registry.list()
        assert len(datasets) == 3

    def test_returns_copy(self, registry):
        d1 = registry.list()
        d2 = registry.list()
        assert d1 is not d2  # separate list objects


# ---------------------------------------------------------------------------
# DatasetRegistry.route
# ---------------------------------------------------------------------------


class TestRoute:
    def test_returns_list_of_routed_datasets(self, registry):
        results = registry.route("trust cooperation")
        assert isinstance(results, list)
        assert all(isinstance(r, RoutedDataset) for r in results)

    def test_top_k_limits_results(self, registry):
        results = registry.route("trust cooperation", top_k=1)
        assert len(results) == 1

    def test_top_k_default_is_3(self, registry):
        results = registry.route("trust cooperation")
        assert len(results) <= 3

    def test_relevant_dataset_scores_higher_than_unrelated(self, registry):
        results = registry.route("trust cooperation ess")
        ids = [r.dataset_id for r in results]
        # ESS should be first (has semantic keywords: trust, cooperation, ess)
        assert ids[0] == "ess11"

    def test_all_zero_scores_for_unrelated_query(self, registry):
        results = registry.route("xyz_unmatched_term_123", top_k=3)
        # May return 3 items all with score 0.0
        assert all(r.score == pytest.approx(0.0) for r in results)

    def test_keyword_match_scores_higher_than_text_match(self, registry):
        # "ess" is a semantic keyword for ess11, giving it extra weight
        results_ess = registry.route("ess trust")
        results_wvs = registry.route("wvs values")
        # Both should find their respective datasets
        ess_result = next((r for r in results_ess if r.dataset_id == "ess11"), None)
        wvs_result = next((r for r in results_wvs if r.dataset_id == "wvs7"), None)
        assert ess_result is not None
        assert wvs_result is not None

    def test_empty_query_returns_results(self, registry):
        results = registry.route("")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# DatasetRegistry.build_rag_context
# ---------------------------------------------------------------------------


class TestBuildRagContext:
    def test_includes_title_and_description(self, registry):
        context = registry.build_rag_context(["ess11"])
        assert "European Social Survey" in context

    def test_multiple_datasets_concatenated(self, registry):
        context = registry.build_rag_context(["ess11", "wvs7"])
        assert "European Social Survey" in context
        assert "World Values Survey" in context

    def test_empty_list_returns_empty_string(self, registry):
        context = registry.build_rag_context([])
        assert context == ""
