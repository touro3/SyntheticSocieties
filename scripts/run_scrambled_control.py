import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Run Scrambled-ESS Negative Control (Condition S)")
    parser.add_argument("--seeds", type=str, default="42,123,7")
    parser.add_argument("--agents", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--policy-type", type=str, default="scrambled_rag")
    args = parser.parse_args()

    seeds = args.seeds.split(",")
    for seed in seeds:
        print(f"Running Scrambled Control (Seed {seed})...")
        exp_id = f"scrambled_control_s{seed}"

        cmd = [
            "python",
            "scripts/run_config_simulation.py",
            "--config",
            "configs/base_config.yaml",
            f"project.experiment_id={exp_id}",
            f"project.seed={seed}",
            f"simulation.population_size={args.agents}",
            f"simulation.rounds={args.rounds}",
            f"negative_control.scramble_seed={seed}",
            "population.source=empirical",
        ]

        if args.policy_type == "mock":
            cmd.append("policy.type=mock")
        else:
            cmd.append("policy.type=scrambled_rag")

        subprocess.run(cmd)


if __name__ == "__main__":
    main()
