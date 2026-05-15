"""Tests for OOD country-holdout split resolution + filtered sampling (Phase 2).

Guarantees the audit-response properties:
  - train/eval cluster sets are disjoint and non-empty,
  - country↔cluster resolution works both ways,
  - sample_empirical_rows honors country filters,
  - an AT-only filter empties the frame and is surfaced loudly (not silent),
  - LOCO holds out exactly the named cluster.

CPU-only, no LLM. Uses the real benchmarks JSON + AT-only parquet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from population.country_clusters import load_clusters
from population.ood_split import OODSplit, leave_one_cluster_out, resolve_split
from population.sampling import sample_empirical_rows

REPO = Path(__file__).resolve().parents[1]
ESS_PARQUET = REPO / "data" / "ess_clean.parquet"


def test_resolve_split_disjoint_and_nonempty():
    split = resolve_split({"clusters": ["nordic", "southern"]}, {"clusters": ["eastern"]})
    train = {c.name for c in split.train_clusters}
    ev = {c.name for c in split.eval_clusters}
    assert train == {"nordic", "southern"}
    assert ev == {"eastern"}
    assert not (train & ev)
    assert split.train_countries and split.eval_countries


def test_overlapping_split_rejected():
    with pytest.raises(ValueError, match="not disjoint"):
        resolve_split({"clusters": ["nordic"]}, {"clusters": ["nordic"]})


def test_empty_side_rejected():
    with pytest.raises(ValueError):
        resolve_split({"clusters": []}, {"clusters": ["eastern"]})


def test_country_spec_maps_back_to_cluster():
    # NO is a Nordic ISO code in cross_cultural_benchmarks.json.
    split = resolve_split({"countries": ["NO"]}, {"clusters": ["eastern"]})
    assert {c.name for c in split.train_clusters} == {"nordic"}


def test_unknown_cluster_rejected():
    with pytest.raises(ValueError, match="Unknown cluster"):
        resolve_split({"clusters": ["atlantis"]}, {"clusters": ["eastern"]})


def test_leave_one_cluster_out_holds_out_named():
    for held in [c.name for c in load_clusters()]:
        split = leave_one_cluster_out(held)
        assert {c.name for c in split.eval_clusters} == {held}
        assert held not in {c.name for c in split.train_clusters}
        assert isinstance(split, OODSplit)


def test_sampling_no_filter_is_backward_compatible():
    rows = sample_empirical_rows(str(ESS_PARQUET), n=10, seed=1)
    assert len(rows) == 10


def test_sampling_country_filter_kept():
    rows = sample_empirical_rows(str(ESS_PARQUET), n=8, seed=1, country_filter=["AT"])
    assert len(rows) == 8
    assert all(r["country"] == "AT" for r in rows)


def test_sampling_filter_emptying_frame_raises_loudly():
    # AT-only parquet → filtering to a non-AT country must error, not silently
    # resample the wrong cohort (the BRM-circularity trap this guards against).
    with pytest.raises(ValueError, match="eliminated all rows"):
        sample_empirical_rows(str(ESS_PARQUET), n=5, seed=1, country_filter=["NO"])


def test_sampling_exclude_countries():
    with pytest.raises(ValueError, match="eliminated all rows"):
        sample_empirical_rows(str(ESS_PARQUET), n=5, seed=1, exclude_countries=["AT"])
