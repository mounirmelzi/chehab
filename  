import argparse
import csv
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .policy import HierarchicalMaskablePolicy
from .env import fheEnv
from .utils import load_expressions, create_rules
from .__main__ import load_embeddings_from_config


DATA_DIR = Path(__file__).resolve().parent / "datasets"
RL_DIR = Path(__file__).resolve().parents[1]


def evaluate_model(model_path: str, embeddings_model, expressions_file: str, model_label: str):
    expressions = load_expressions(expressions_file)
    rules_list = create_rules(str(RL_DIR / "rules.txt"))
    rules_list["END"] = None
    max_positions = 16

    env = DummyVecEnv([
        lambda: Monitor(fheEnv(rules_list, expressions, max_positions=max_positions, embeddings_model=embeddings_model))
    ])
    model = PPO.load(model_path, env=env, custom_objects={"policy": HierarchicalMaskablePolicy})

    results = []
    num_expr = len(expressions)
    for idx in range(num_expr):
        start = time.perf_counter()
        obs = env.reset()
        wrapper = env.envs[0]
        fhe_env = wrapper.env
        test_expr = fhe_env.initial_expression
        initial_cost = fhe_env.initial_cost

        done = False
        steps = 0
        last_expr = test_expr
        last_cost = initial_cost
        last_noise = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            last_noise = float(info0.get("noise", 0.0))
            wrapper = env.envs[0]
            fhe_env = wrapper.env
            last_expr = fhe_env.expression
            last_cost = fhe_env.current_cost
            steps += 1

        elapsed = time.perf_counter() - start
        results.append({
            "model": model_label,
            "index": idx,
            "initial_expression": test_expr,
            "final_expression": last_expr,
            "initial_cost": initial_cost,
            "final_cost": last_cost,
            "steps": steps,
            "noise_used": last_noise,
            "time_seconds": elapsed,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL agents on benchmarks with noise and timing")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to model .zip files (one or more)")
    parser.add_argument("--labels", nargs="+", help="Optional labels for models (same length as --models)")
    parser.add_argument("--expressions", default=str(DATA_DIR / "benchmarks.txt"), help="Expressions file to evaluate")
    parser.add_argument("--out", default="eval_results.csv", help="Output CSV path")
    args = parser.parse_args()

    embeddings, _ = load_embeddings_from_config()

    labels = args.labels if args.labels and len(args.labels) == len(args.models) else [f"model_{i}" for i in range(len(args.models))]

    all_results = []
    for model_path, label in zip(args.models, labels):
        res = evaluate_model(model_path, embeddings, args.expressions, label)
        all_results.extend(res)

    fieldnames = [
        "model", "index", "initial_expression", "final_expression",
        "initial_cost", "final_cost", "steps", "noise_used", "time_seconds"
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"Wrote {len(all_results)} rows to {args.out}")


if __name__ == "__main__":
    main()


