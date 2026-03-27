"""Country cluster definitions for Phase 27 cross-cultural validation.

These clusters are grounded in ESS-11 published aggregate statistics.
The local parquet only contains Austrian microdata; simulation parameters
are set via trust_people_band to approximate each cluster's trust profile.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CountryCluster:
    """A cultural country cluster defined by ESS-11 published trust norms.

    Attributes:
        name: Cluster key (e.g., "nordic", "southern", "eastern").
        countries: ISO country codes in this cluster (informational).
        ess_mean_trust: ESS-11 published mean interpersonal trust [0, 1].
        ess_sd_trust: Published standard deviation.
        trust_band: SocietySpec trust_people_band value used for simulation.
        description: Human-readable cluster description.
    """

    name: str
    countries: list[str]
    ess_mean_trust: float
    ess_sd_trust: float
    trust_band: str
    description: str


# Ordered lowest to highest trust so a positive Spearman r confirms gradient recovery.
CANONICAL_CLUSTER_ORDER: list[str] = ["eastern", "southern", "nordic"]


def load_clusters(
    benchmarks_path: str | Path = "data/cross_cultural_benchmarks.json",
) -> list[CountryCluster]:
    """Load cluster definitions from the ESS-11 benchmarks JSON.

    Args:
        benchmarks_path: Path to cross_cultural_benchmarks.json.

    Returns:
        List of CountryCluster instances, unordered.
    """
    path = Path(benchmarks_path)
    with path.open() as f:
        data = json.load(f)
    clusters = []
    for name, c in data["clusters"].items():
        clusters.append(
            CountryCluster(
                name=name,
                countries=c["countries"],
                ess_mean_trust=c["ess_mean_trust_people"],
                ess_sd_trust=c["ess_sd_trust_people"],
                trust_band=c["trust_band"],
                description=c["description"],
            )
        )
    return clusters


def get_cluster_by_name(name: str, benchmarks_path: str | Path = "data/cross_cultural_benchmarks.json") -> CountryCluster:
    """Return a single cluster by name."""
    clusters = {c.name: c for c in load_clusters(benchmarks_path)}
    if name not in clusters:
        raise KeyError(f"Unknown cluster '{name}'. Available: {list(clusters)}")
    return clusters[name]
