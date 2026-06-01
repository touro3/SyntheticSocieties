"""Out-of-distribution (OOD) country-holdout split resolution.

Research motivation
-------------------
The BRM realism metric is *circular* if the same ESS source that conditions
the agents is also the target it is scored against — the framework could be
rewarded merely for echoing its conditioning distribution. To break this we
need a genuine out-of-distribution test: condition on one part of the
empirical world, evaluate realism on a disjoint part whose benchmark never
informed the model.

Data constraint (honest scoping)
--------------------------------
The local ``data/ess_clean.parquet`` contains **only Austrian (AT) microdata**
(866 rows, single country). A microdata-level country holdout is therefore
impossible. The codebase already resolves this exactly the way the
cross-cultural module does: each ESS country *cluster* (Nordic / Southern /
Eastern) is approximated by filtering the AT sample to the cluster's
``trust_band`` and validated against the cluster's **published ESS-11
benchmark** (`data/cross_cultural_benchmarks.json`).

Cross-country variation therefore exists only at the **cluster-benchmark
level**, so that is where the OOD split operates. The honest, non-circular
test available with this data is **leave-one-cluster-out (LOCO)**: fit the
trust→behavior calibration on the train clusters, then predict the held-out
cluster whose published benchmark was *never* used in the fit.

This module just resolves a split spec into concrete cluster/country lists,
reusing :mod:`population.country_clusters`. The LOCO fit + realism scoring
lives in ``scripts/run_ood_validation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from population.country_clusters import (
    CANONICAL_CLUSTER_ORDER,
    CountryCluster,
    load_clusters,
)


@dataclass
class OODSplit:
    """A resolved train/eval partition over ESS country clusters.

    Attributes:
        train_clusters: CountryCluster objects used for conditioning + fitting.
        eval_clusters:  Held-out CountryCluster objects scored OOD.
        train_countries: Flattened ISO codes of all train clusters.
        eval_countries:  Flattened ISO codes of all eval clusters.
    """

    train_clusters: list[CountryCluster]
    eval_clusters: list[CountryCluster]
    train_countries: list[str]
    eval_countries: list[str]

    def __post_init__(self) -> None:
        train_names = {c.name for c in self.train_clusters}
        eval_names = {c.name for c in self.eval_clusters}
        overlap = train_names & eval_names
        if overlap:
            raise ValueError(f"OOD split is not disjoint — clusters in both train and eval: {sorted(overlap)}")
        if not train_names or not eval_names:
            raise ValueError(
                f"OOD split must be non-empty on both sides (train={sorted(train_names)}, eval={sorted(eval_names)})."
            )


def _resolve_cluster_names(spec: dict, all_clusters: dict[str, CountryCluster]) -> list[str]:
    """Resolve a split spec dict to a list of cluster names.

    Accepts ``{"clusters": [...]}`` directly, or ``{"countries": [...]}`` which
    is mapped back to the clusters those ISO codes belong to.
    """
    if "clusters" in spec:
        names = list(spec["clusters"])
    elif "countries" in spec:
        wanted = set(spec["countries"])
        names = [name for name, c in all_clusters.items() if wanted & set(c.countries)]
    else:
        raise ValueError(f"Split spec must contain 'clusters' or 'countries'; got keys {sorted(spec)}")

    unknown = [n for n in names if n not in all_clusters]
    if unknown:
        raise ValueError(f"Unknown cluster(s) {unknown}. Available: {sorted(all_clusters)}")
    return names


def resolve_split(
    train_spec: dict,
    eval_spec: dict,
    benchmarks_path: str | Path = "data/cross_cultural_benchmarks.json",
) -> OODSplit:
    """Resolve explicit train/eval split specs into an :class:`OODSplit`.

    Args:
        train_spec: e.g. ``{"clusters": ["nordic", "southern"]}``.
        eval_spec:  e.g. ``{"clusters": ["eastern"]}``.
        benchmarks_path: Path to the cluster benchmarks JSON.
    """
    by_name = {c.name: c for c in load_clusters(benchmarks_path)}
    train_names = _resolve_cluster_names(train_spec, by_name)
    eval_names = _resolve_cluster_names(eval_spec, by_name)

    train_clusters = [by_name[n] for n in train_names]
    eval_clusters = [by_name[n] for n in eval_names]
    return OODSplit(
        train_clusters=train_clusters,
        eval_clusters=eval_clusters,
        train_countries=[iso for c in train_clusters for iso in c.countries],
        eval_countries=[iso for c in eval_clusters for iso in c.countries],
    )


def leave_one_cluster_out(
    held_out: str,
    benchmarks_path: str | Path = "data/cross_cultural_benchmarks.json",
) -> OODSplit:
    """Build a LOCO split: ``held_out`` is the eval cluster, the rest train.

    This is the primary OOD design for the current AT-only dataset — the
    held-out cluster's published benchmark never enters the calibration fit.
    """
    by_name = {c.name: c for c in load_clusters(benchmarks_path)}
    if held_out not in by_name:
        raise ValueError(f"Unknown cluster '{held_out}'. Available: {sorted(by_name)}")
    train = [n for n in CANONICAL_CLUSTER_ORDER if n in by_name and n != held_out]
    return resolve_split({"clusters": train}, {"clusters": [held_out]}, benchmarks_path)
