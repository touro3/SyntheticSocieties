from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from population.ess_grounding import ESSGrounder
from population.persona_synthesizer import (
    save_persona_records,
    synthesize_ess_personas,
    synthesize_spec_personas,
)
from population.spec_parser import parse_society_prompt
from utils.io import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Build society artifacts from a natural-language prompt.")
    parser.add_argument("--prompt", required=True, type=str, help="Natural-language society description.")
    parser.add_argument("--out", required=True, type=str, help="Output artifact directory.")
    parser.add_argument("--population-size", type=int, default=50, help="Number of personas to synthesize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--ess-path", type=str, default="data/ess_clean.parquet", help="Path to cleaned ESS parquet.")
    parser.add_argument("--min-cohort-size", type=int, default=30, help="Minimum cohort size after progressive relaxation.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out)

    spec = parse_society_prompt(args.prompt, target_population_size=args.population_size)
    grounder = ESSGrounder(args.ess_path, min_cohort_size=args.min_cohort_size)
    grounding = grounder.ground(spec)

    ess_personas = synthesize_ess_personas(
        grounding.matched_df,
        spec,
        n=spec.target_population_size,
        seed=args.seed,
    )
    synthetic_personas = synthesize_spec_personas(
        spec,
        n=spec.target_population_size,
        seed=args.seed,
    )

    save_json(spec.model_dump(), out_dir / "society_spec.json")
    save_json(grounding.report.model_dump(), out_dir / "grounding_report.json")

    with (out_dir / "population_context.txt").open("w", encoding="utf-8") as f:
        f.write(grounding.population_context)

    save_persona_records(ess_personas, out_dir / "ess_personas.jsonl")
    save_persona_records(synthetic_personas, out_dir / "synthetic_personas.jsonl")
    grounding.matched_df.head(min(500, len(grounding.matched_df))).to_csv(out_dir / "grounding_sample.csv", index=False)

    manifest = {
        "society_spec_path": str(out_dir / "society_spec.json"),
        "grounding_report_path": str(out_dir / "grounding_report.json"),
        "population_context_path": str(out_dir / "population_context.txt"),
        "ess_personas_path": str(out_dir / "ess_personas.jsonl"),
        "synthetic_personas_path": str(out_dir / "synthetic_personas.jsonl"),
        "grounding_sample_path": str(out_dir / "grounding_sample.csv"),
        "min_cohort_size": args.min_cohort_size,
    }
    save_json(manifest, out_dir / "manifest.json")

    print(
        json.dumps(
            {
                "artifacts_dir": str(out_dir),
                "matched_rows": grounding.report.matched_rows,
                "population_size": spec.target_population_size,
                "selected_datasets": grounding.report.selected_datasets,
                "min_cohort_size": grounding.report.min_cohort_size,
                "warnings": grounding.report.warnings,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
