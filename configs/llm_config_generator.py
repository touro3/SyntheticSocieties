"""
LLM-generated simulation configuration.

Converts a natural-language scenario description into a validated BGFConfig
using 4 sequential LLM calls (MiroFish staged-generation pattern).  Each
stage focuses on one configuration domain, preventing token overflow and
allowing targeted validation before the next stage runs.

Stages
──────
  1. Population  — countries, trust levels, age profile, population size
  2. Economy     — wealth distribution, shocks, public signal, job levels
  3. Network     — topology (random / small_world), density, rewiring
  4. Experiment  — rounds, policy type, seed, experiment id

Usage
─────
    from configs.llm_config_generator import LLMConfigGenerator

    gen = LLMConfigGenerator(api_key="...", base_url="...")
    config_dict = gen.generate("A Nordic welfare state under fiscal austerity, 50 agents, 30 rounds")
    # config_dict is a plain dict ready for yaml.dump() or BGFConfig(**config_dict)

    # CLI
    python configs/llm_config_generator.py \\
        --scenario "Nordic welfare state under austerity" \\
        --out configs/generated/nordic_austerity.yaml
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ── Stage prompt templates ─────────────────────────────────────────────────────

_STAGE1_POPULATION = textwrap.dedent("""
    You are configuring a BGF (Behavioral Grounding Framework) agent-based simulation.

    Scenario: {scenario}

    Generate ONLY the population and agent_defaults configuration as a JSON object.
    Use these exact keys (no others):

    {{
      "population": {{
        "source": "empirical" or "synthetic"
      }},
      "simulation": {{
        "population_size": <integer 5-500>
      }},
      "agent_defaults": {{
        "min_age": <int>,
        "max_age": <int>,
        "risk_tolerance": <float 0.0-1.0>,
        "initial_wealth": <float>,
        "memory_size": <int 5-20>
      }}
    }}

    Rules:
    - Use "empirical" source only if the scenario mentions real ESS countries (Nordic, Eastern, Southern Europe).
    - population_size: keep <= 100 for local runs, <= 500 for cluster runs.
    - risk_tolerance: low (<0.35) for conservative/welfare societies, high (>0.65) for competitive/liberal ones.
    - Return ONLY the JSON object, no commentary.
""").strip()

_STAGE2_ECONOMY = textwrap.dedent("""
    You are configuring the economic environment for a BGF simulation.

    Scenario: {scenario}
    Population already configured: {population_summary}

    Generate ONLY the environment configuration as a JSON object.
    Use these exact keys:

    {{
      "environment": {{
        "public_signal": {{
          "economy": "stable" | "recession" | "growth" | "crisis"
        }},
        "prices": {{
          "food": <float, 1.0 = normal, >1.5 = scarcity>
        }},
        "resources": {{
          "jobs": <float, 100.0 = full employment, 50.0 = high unemployment>
        }}
      }}
    }}

    Rules:
    - "crisis" + food > 1.5 + jobs < 60 for austerity/crisis scenarios.
    - "growth" + food ~1.0 + jobs ~120 for boom scenarios.
    - "stable" + food ~1.0 + jobs ~100 for baseline scenarios.
    - Return ONLY the JSON object.
""").strip()

_STAGE3_NETWORK = textwrap.dedent("""
    You are configuring the social network topology for a BGF simulation.

    Scenario: {scenario}
    Population size: {population_size}

    Generate ONLY the network configuration as a JSON object.
    Use these exact keys:

    {{
      "network": {{
        "type": "random" | "small_world",
        "edge_prob": <float 0.1-0.8>,
        "k": <int 2-6>,
        "rewiring_prob": <float 0.05-0.5>
      }}
    }}

    Rules:
    - "small_world" for tight community scenarios (Nordic, village-based).
    - "random" for atomised / urban / crisis scenarios.
    - edge_prob: 0.2-0.3 sparse, 0.5-0.7 dense.
    - k (small_world only): 2 = sparse lattice, 4-6 = dense neighbourhood.
    - rewiring_prob: 0.05-0.1 = high clustering, 0.3-0.5 = more random.
    - Return ONLY the JSON object.
""").strip()

_STAGE4_EXPERIMENT = textwrap.dedent("""
    You are finalising the experiment configuration for a BGF simulation.

    Scenario: {scenario}
    Previous stages: population_size={population_size}, economy={economy_signal}, network={network_type}

    Generate ONLY the project/simulation/policy/llm configuration as a JSON object.
    Use these exact keys:

    {{
      "project": {{
        "name": "bgf",
        "experiment_id": "<short_snake_case_id_max_30_chars>",
        "seed": <int 1-9999>
      }},
      "simulation": {{
        "rounds": <int 10-100>
      }},
      "policy": {{
        "type": "llm" | "rule_based" | "random"
      }},
      "llm": {{
        "temperature": <float 0.3-1.0>,
        "max_new_tokens": 256,
        "memory_window": <int 3-10>
      }}
    }}

    Rules:
    - experiment_id: snake_case, descriptive, no spaces (e.g. "nordic_austerity_30r").
    - rounds: 10-20 for quick tests, 30-50 for standard, 100 for long-horizon.
    - policy "llm" for grounded/ungrounded agent experiments; "rule_based" for baselines.
    - temperature: 0.3-0.5 for reproducibility, 0.7-1.0 for behavioural diversity.
    - Return ONLY the JSON object.
""").strip()


# ── JSON repair (reuse logic from output_parser) ──────────────────────────────


def _repair_and_parse(text: str) -> dict:
    """Extract and parse a JSON object from LLM output with light repair."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        # Remove trailing commas before } or ]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(candidate)
    raise ValueError(f"No JSON object found in LLM output: {text[:200]!r}")


# ── Deep merge utility ────────────────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflict)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ── Default fallback config ───────────────────────────────────────────────────

_DEFAULT_CONFIG: dict = {
    "project": {"name": "bgf", "experiment_id": "generated_exp", "seed": 42},
    "simulation": {"rounds": 30, "population_size": 50},
    "policy": {"type": "llm"},
    "population": {"source": "synthetic"},
    "data": {
        "ess_interview_path": "data/ESS11INTe04_1.csv",
        "ess_main_path": "data/ESS11MD_e01_2.csv",
        "ess_clean_path": "data/ess_clean.parquet",
        "distributions_path": "data/empirical_distributions.json",
        "sample_mode": "resample",
    },
    "llm": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "cache_dir": None,
        "dtype": "float16",
        "device_map": "auto",
        "temperature": 0.7,
        "max_new_tokens": 256,
        "memory_window": 5,
        "max_retries": 2,
        "inference_timeout": 120,
    },
    "network": {"type": "random", "edge_prob": 0.5, "k": 2, "rewiring_prob": 0.3},
    "environment": {
        "public_signal": {"economy": "stable"},
        "prices": {"food": 1.0},
        "resources": {"jobs": 100.0},
    },
    "agent_defaults": {
        "min_age": 25,
        "max_age": 60,
        "base_income": 1000.0,
        "income_step": 100.0,
        "education": "college",
        "occupation": "worker",
        "location": "urban",
        "political_preference": "center",
        "risk_tolerance": 0.5,
        "social_class": "middle",
        "initial_wealth": 50.0,
        "wealth_step": 10.0,
        "memory_size": 10,
    },
}


# ── Generator ─────────────────────────────────────────────────────────────────


class LLMConfigGenerator:
    """
    Generates BGF simulation configs from natural-language scenario descriptions.

    Uses 4 sequential LLM calls (one per config domain) to avoid token overflow
    and allow per-stage validation.  Each stage's output is merged into a running
    config dict; if any stage fails (JSON parse error, LLM timeout), the default
    value for that domain is used and a warning is logged.

    Args:
        api_key:     API key for the OpenAI-compatible endpoint.
        base_url:    Base URL. Defaults to the official OpenAI API.
        model:       Model name. Defaults to gpt-4o-mini.
        temperature: Sampling temperature for all stages.
        max_retries: How many times to retry a failed stage before using defaults.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_retries: int = 2,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError as exc:
            raise ImportError("openai package required. Install with: pip install openai") from exc

    def _call_llm(self, prompt: str) -> dict:
        """Call the LLM and parse JSON from the response, with retry."""
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                raw = resp.choices[0].message.content or "{}"
                return _repair_and_parse(raw)
            except Exception as exc:
                last_exc = exc
                logger.warning("Stage LLM call attempt %d failed: %s", attempt + 1, exc)
        raise last_exc

    def generate(self, scenario: str, verbose: bool = False) -> dict:
        """
        Run all 4 stages and return a merged, validated config dict.

        Args:
            scenario: Natural-language scenario description.
            verbose:  Print each stage's output.

        Returns:
            Dict ready for ``yaml.dump()`` or ``BGFConfig(**config)``.
        """
        config = dict(_DEFAULT_CONFIG)

        # ── Stage 1: Population ───────────────────────────────────────────────
        stage1 = self._run_stage(
            name="Population",
            prompt=_STAGE1_POPULATION.format(scenario=scenario),
            verbose=verbose,
        )
        config = _deep_merge(config, stage1)

        # ── Stage 2: Economy ─────────────────────────────────────────────────
        pop_summary = (
            f"size={config['simulation'].get('population_size', 50)}, "
            f"source={config['population'].get('source', 'synthetic')}"
        )
        stage2 = self._run_stage(
            name="Economy",
            prompt=_STAGE2_ECONOMY.format(scenario=scenario, population_summary=pop_summary),
            verbose=verbose,
        )
        config = _deep_merge(config, stage2)

        # ── Stage 3: Network ─────────────────────────────────────────────────
        stage3 = self._run_stage(
            name="Network",
            prompt=_STAGE3_NETWORK.format(
                scenario=scenario,
                population_size=config["simulation"].get("population_size", 50),
            ),
            verbose=verbose,
        )
        config = _deep_merge(config, stage3)

        # ── Stage 4: Experiment ───────────────────────────────────────────────
        stage4 = self._run_stage(
            name="Experiment",
            prompt=_STAGE4_EXPERIMENT.format(
                scenario=scenario,
                population_size=config["simulation"].get("population_size", 50),
                economy_signal=config["environment"]["public_signal"].get("economy", "stable"),
                network_type=config["network"].get("type", "random"),
            ),
            verbose=verbose,
        )
        # Stage 4 also updates population_size from simulation block
        config = _deep_merge(config, stage4)

        # ── Validate against BGFConfig schema ────────────────────────────────
        config = self._validate(config)

        return config

    def _run_stage(self, name: str, prompt: str, verbose: bool) -> dict:
        """Run one LLM stage, returning the parsed dict (or {} on failure)."""
        try:
            result = self._call_llm(prompt)
            if verbose:
                print(f"\n[Stage {name}]\n{json.dumps(result, indent=2)}")
            return result
        except Exception as exc:
            logger.warning("Stage '%s' failed, using defaults: %s", name, exc)
            if verbose:
                print(f"\n[Stage {name}] FAILED ({exc}) — using defaults")
            return {}

    @staticmethod
    def _validate(config: dict) -> dict:
        """Clamp numeric fields to valid ranges after LLM generation."""
        sim = config.setdefault("simulation", {})
        sim["population_size"] = max(1, min(int(sim.get("population_size", 50)), 500))
        sim["rounds"] = max(1, min(int(sim.get("rounds", 30)), 200))

        net = config.setdefault("network", {})
        net["edge_prob"] = max(0.05, min(float(net.get("edge_prob", 0.5)), 0.95))
        net["rewiring_prob"] = max(0.01, min(float(net.get("rewiring_prob", 0.3)), 0.99))
        net["k"] = max(1, min(int(net.get("k", 2)), 10))
        if net.get("type") not in ("random", "small_world"):
            net["type"] = "random"

        env = config.setdefault("environment", {})
        env.setdefault("public_signal", {}).setdefault("economy", "stable")
        prices = env.setdefault("prices", {})
        prices["food"] = max(0.5, min(float(prices.get("food", 1.0)), 5.0))
        resources = env.setdefault("resources", {})
        resources["jobs"] = max(10.0, min(float(resources.get("jobs", 100.0)), 500.0))

        llm = config.setdefault("llm", {})
        llm["temperature"] = max(0.0, min(float(llm.get("temperature", 0.7)), 2.0))
        llm["memory_window"] = max(1, min(int(llm.get("memory_window", 5)), 20))

        ad = config.setdefault("agent_defaults", {})
        ad["risk_tolerance"] = max(0.0, min(float(ad.get("risk_tolerance", 0.5)), 1.0))
        ad["initial_wealth"] = max(0.0, float(ad.get("initial_wealth", 50.0)))

        pol = config.setdefault("policy", {})
        if pol.get("type") not in (
            "mock",
            "random",
            "template",
            "rule_based",
            "llm",
            "conditioned_llm",
            "generative_agents",
            "ablated_llm",
            "data_driven",
        ):
            pol["type"] = "llm"

        return config

    def save(self, config: dict, path: str | Path) -> None:
        """Save a generated config dict to a YAML file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info("Config saved to %s", out)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a BGF simulation config from a natural-language scenario")
    parser.add_argument("--scenario", required=True, help="Natural-language scenario description")
    parser.add_argument("--out", default=None, help="Output YAML path (default: print to stdout)")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    gen = LLMConfigGenerator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    config = gen.generate(args.scenario, verbose=args.verbose)

    if args.out:
        gen.save(config, args.out)
        print(f"Config saved to {args.out}")
    else:
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    _cli()
