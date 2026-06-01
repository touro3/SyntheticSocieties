"""Pydantic validation schema for BGF configuration files.

Catches typos, invalid values, and non-portable paths at load time
rather than deep inside a simulation run.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ProjectConfig(BaseModel):
    name: str = "bgf"
    experiment_id: str = Field(default_factory=lambda: f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    seed: int = 42


class SimulationConfig(BaseModel):
    # Upper bounds cap compute/disk so an oversized config (uploaded YAML or
    # LLM-designed scenario) cannot exhaust GPU/CPU/disk. Headroom over the
    # largest shipped pipeline (phase_d: 500 agents × 100 rounds).
    rounds: int = Field(default=3, ge=1, le=200)
    population_size: int = Field(default=5, ge=1, le=1000)


class PolicyConfig(BaseModel):
    type: Literal[
        "mock",
        "random",
        "template",
        "rule_based",
        "llm",
        "conditioned_llm",
        "generative_agents",
        "ablated_llm",
        "data_driven",
    ] = "mock"


class PopulationConfig(BaseModel):
    # "empirical" = ESS-coherent grounding; "placebo" = scrambled-but-valid
    # semantic-isolation control; "synthetic" = config-default unconditioned.
    source: Literal["empirical", "synthetic", "placebo"] = "empirical"


class DataConfig(BaseModel):
    ess_interview_path: str = "data/ESS11INTe04_1.csv"
    ess_main_path: str = "data/ESS11MD_e01_2.csv"
    ess_clean_path: str = "data/ess_clean.parquet"
    distributions_path: str = "data/empirical_distributions.json"
    sample_mode: Literal["resample", "subsample"] = "resample"
    # OOD country-holdout splits. Each is a dict like
    # {"clusters": ["nordic", "southern"]} or {"countries": ["AT"]}.
    # None (default) = no split → fully backward compatible.
    train_split: Optional[dict] = None
    eval_split: Optional[dict] = None


class LLMConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    cache_dir: Optional[str] = None
    dtype: str = "float16"
    device_map: str = "auto"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=256, ge=1, le=8192)
    memory_window: int = Field(default=5, ge=1, le=100)
    max_retries: int = Field(default=2, ge=0, le=10)
    inference_timeout: int = Field(default=120, ge=1, le=600)

    @field_validator("cache_dir", mode="before")
    @classmethod
    def resolve_cache_dir(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        expanded = os.path.expanduser(os.path.expandvars(v))
        return expanded


class NetworkConfig(BaseModel):
    type: Literal["random", "small_world"] = "random"
    edge_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    k: int = Field(default=2, ge=1)
    rewiring_prob: float = Field(default=0.3, ge=0.0, le=1.0)


class EnvironmentConfig(BaseModel):
    public_signal: dict = Field(default_factory=lambda: {"economy": "stable"})
    prices: dict = Field(default_factory=lambda: {"food": 1.0})
    resources: dict = Field(default_factory=lambda: {"jobs": 100.0})


class AgentDefaultsConfig(BaseModel):
    min_age: int = Field(default=25, ge=0)
    max_age: int = Field(default=60, ge=0)
    base_income: float = Field(default=1000.0, ge=0.0)
    income_step: float = Field(default=100.0, ge=0.0)
    education: str = "college"
    occupation: str = "worker"
    location: str = "italy"
    political_preference: str = "center"
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    social_class: str = "middle"
    initial_wealth: float = Field(default=50.0, ge=0.0)
    wealth_step: float = Field(default=10.0, ge=0.0)
    memory_size: int = Field(default=10, ge=1)
    memory_persistent: bool = Field(
        default=False,
        description="Opt-in: mirror agent memory to a disk-persistent semantic "
        "store (experiments/<exp_id>/memory/<agent>.db). Default off — "
        "preserves byte-identical behavior for the M0–M3 ablation.",
    )
    embedding_model: str = Field(default="all-MiniLM-L6-v2")


class BGFConfig(BaseModel):
    """Top-level configuration schema for the BGF simulation framework."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    agent_defaults: AgentDefaultsConfig = Field(default_factory=AgentDefaultsConfig)
