from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from decision.dataset_router import DatasetRegistry
from decision.fidelity_prompt_builder import (
    build_fidelity_messages,
    parse_fidelity_output,
    prompt_text,
)
from decision.llm_backend import LLMBackend
from metrics.persona_fidelity import (
    compute_fidelity_report,
    summarize_synthetic_runs,
    write_report_files,
)
from population.profile_tree import (
    build_profile_text,
    fit_profile_tree,
    prepare_benchmark_frame,
)
from utils.io import ensure_dir, save_json, set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run persona fidelity benchmark using survey-grounded profiles.")
    parser.add_argument("--society-prompt", required=True, type=str, help="Natural-language description used for semantic dataset routing.")
    parser.add_argument("--registry-path", type=str, default="data/dataset_registry.json")
    parser.add_argument("--dataset-id", type=str, default=None, help="Force one dataset id instead of semantic routing.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="artifacts/persona_fidelity")
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--model-id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--include-justification", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    return parser.parse_args()


def _run_id(args) -> str:
    if args.run_id:
        return args.run_id
    return f"persona_fidelity_{int(time.time())}"


def _candidate_dataset_ids(args, registry: DatasetRegistry) -> list[str]:
    if args.dataset_id:
        return [args.dataset_id]
    routed = registry.route(args.society_prompt, top_k=args.top_k)
    return [item.dataset_id for item in routed if item.score > 0] or [registry.list()[0]["id"]]


def _seed_list(base_seed: int, replications: int) -> list[int]:
    return [base_seed + i for i in range(replications)]


def main():
    args = parse_args()
    run_dir = ensure_dir(Path(args.output_root) / _run_id(args))
    registry = DatasetRegistry(args.registry_path)

    routed = registry.route(args.society_prompt, top_k=args.top_k)
    registry.audit_selection(args.society_prompt, routed, run_dir / "dataset_selection.json")

    dataset_ids = _candidate_dataset_ids(args, registry)
    primary_dataset_id = dataset_ids[0]
    dataset = registry.get(primary_dataset_id)

    profile_features = dataset["profile_features"]
    target_items = dataset["target_items"]
    selected_names = [f["name"] for f in profile_features] + [t["name"] for t in target_items]

    required_columns = registry.required_columns(primary_dataset_id, selected_names)
    frame = registry.load_frame(primary_dataset_id, required_columns)

    if "trust_institutions" not in frame.columns and {"trust_parliament", "trust_legal", "trust_police"}.issubset(frame.columns):
        frame["trust_institutions"] = frame[["trust_parliament", "trust_legal", "trust_police"]].mean(axis=1)

    benchmark_frame = prepare_benchmark_frame(frame, profile_features, target_items)
    tree_result = fit_profile_tree(
        benchmark_frame,
        profile_features=profile_features,
        target_items=target_items,
        min_support=args.min_support,
        max_depth=args.max_depth,
        random_state=args.base_seed,
    )

    save_json(
        {
            "run_id": run_dir.name,
            "society_prompt": args.society_prompt,
            "primary_dataset_id": primary_dataset_id,
            "selected_dataset_ids": dataset_ids,
            "profile_features": [f["name"] for f in profile_features],
            "target_items": [t["name"] for t in target_items],
            "min_support": args.min_support,
            "max_depth": args.max_depth,
            "replications": args.replications,
            "base_seed": args.base_seed,
        },
        run_dir / "benchmark_config.json",
    )

    save_json(tree_result.profile_definitions, run_dir / "profile_definitions.json")
    tree_result.profiles_df.to_csv(run_dir / "real_profile_summary.csv", index=False)
    tree_result.frame.to_csv(run_dir / "benchmark_frame.csv", index=False)

    dataset_rag_context = registry.build_rag_context(dataset_ids)
    (run_dir / "dataset_rag_context.txt").write_text(dataset_rag_context, encoding="utf-8")

    if args.skip_llm:
        print(f"Prepared profile benchmark only: {run_dir}")
        return

    backend = LLMBackend.get_instance(
        model_id=args.model_id,
        dtype="float16",
        device_map="auto",
        max_new_tokens=256,
        temperature=args.temperature,
        cache_dir=args.cache_dir,
        inference_timeout=120,
        max_retries=2,
    )
    backend.load()

    seeds = _seed_list(args.base_seed, args.replications)
    llm_records = []

    for profile_def in tree_result.profile_definitions:
        profile_text = build_profile_text(profile_def, profile_features)

        for replication_seed in seeds:
            set_global_seed(replication_seed)
            messages = build_fidelity_messages(
                profile_def=profile_def,
                profile_text=profile_text,
                target_items=target_items,
                dataset_rag_context=dataset_rag_context,
                include_justification=args.include_justification,
            )
            raw_output, latency = backend.generate(messages=messages, temperature=args.temperature)
            parsed, justification = parse_fidelity_output(raw_output, target_items)

            llm_records.append(
                {
                    "profile_id": profile_def["profile_id"],
                    "leaf_id": profile_def["leaf_id"],
                    "n_real": profile_def["n_real"],
                    "replication_seed": replication_seed,
                    "prompt": prompt_text(messages),
                    "raw_output": raw_output,
                    "justification": justification,
                    "latency_ms": latency * 1000.0,
                    **parsed,
                }
            )

    synthetic_runs_df = pd.DataFrame(llm_records)
    synthetic_runs_df.to_json(run_dir / "persona_fidelity_runs.jsonl", orient="records", lines=True, force_ascii=False)

    synthetic_profile_df = summarize_synthetic_runs(synthetic_runs_df, target_items)
    report, per_profile_df = compute_fidelity_report(
        real_profiles_df=tree_result.profiles_df,
        synthetic_profile_df=synthetic_profile_df,
        target_items=target_items,
    )
    write_report_files(run_dir, report, per_profile_df, synthetic_profile_df)

    print(f"Persona fidelity benchmark completed: {run_dir}")
    print(run_dir / "profile_definitions.json")
    print(run_dir / "real_profile_summary.csv")
    print(run_dir / "synthetic_profile_summary.csv")
    print(run_dir / "fidelity_report.json")


if __name__ == "__main__":
    main()
