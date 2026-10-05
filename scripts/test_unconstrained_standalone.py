"""Standalone test for the unconstrained PPO agent on augmented benchmarks.

The unconstrained agent (Bilel & Raouf) was trained with an older observation
space that does NOT include budget_one_hot_encoding.  The current fheEnv always
adds it, but after the backwards-compat fix in policy.py the model loads fine
and simply ignores the budget keys.

We run the agent once (its behavior is budget-independent) and replicate the
per-expression results across all requested test budgets.
"""

import sys, os, importlib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RL"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RL", "pytrs"))

# Old model was saved under package name "fhe_rl_new" — alias it so cloudpickle can deserialize
sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")

from stable_baselines3 import PPO
from pytrs import parse_sexpr, NoiseEstimator
from fhe_rl.utils import load_expressions, load_embeddings, create_rules

BENCHMARKS = "./fhe_rl/datasets/benchmarks.txt"
MODEL_PATH = "./fhe_rl/trained_models/agent_dynamic_llm_data.zip"
OUTPUT = "./test_results/test_aug_unconstrained_ppo.xlsx"
TEST_BUDGETS = [230, 233, 236, 300, 369, 372, 9_000_000]


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), "..", "RL"))

    expressions = load_expressions(BENCHMARKS)
    rules_list = create_rules("rules.txt")
    rules_list["END"] = None
    noise_estimator = NoiseEstimator()
    emb_path = "./fhe_rl/trained_models/embeddings_ROT_15_32_5m_10742576.pth"
    embeddings, _ = load_embeddings(tokenizer_type="dynamic", checkpoint_path=emb_path)

    model = PPO.load(MODEL_PATH)
    obs_keys = list(model.observation_space.spaces.keys()) if hasattr(model.observation_space, 'spaces') else []
    print(f"Loaded model from {MODEL_PATH}")
    print(f"  Policy obs space keys: {obs_keys}")
    print(f"  Num expressions: {len(expressions)}")

    from fhe_rl.env import fheEnv

    env = fheEnv(
        rules_list, expressions, max_positions=16,
        embeddings_model=embeddings,
        budget_options=[9_000_000],
        constraint_method="none",
    )

    model_obs_keys = set(obs_keys)

    per_expr_results = []

    print(f"\nRunning unconstrained agent on {len(expressions)} expressions...")
    for expr_idx in range(len(expressions)):
        obs, info = env.reset(options={"budget": 9_000_000})

        initial_expr = env.initial_expression
        initial_cost = env.initial_cost
        initial_noise = float(noise_estimator.estimate(parse_sexpr(initial_expr)))

        done = False
        steps = 0
        while not done:
            obs_for_model = {k: v for k, v in obs.items() if k in model_obs_keys or k == "action_mask" or k == "observation"}
            action, _ = model.predict(obs_for_model, deterministic=True)
            action = int(action) if hasattr(action, '__len__') and len(action) == 1 else int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        final_expr = env.expression
        final_cost = env.current_cost
        final_noise = float(noise_estimator.estimate(parse_sexpr(final_expr)))
        cost_red = ((initial_cost - final_cost) / initial_cost * 100) if initial_cost > 0 else 0

        per_expr_results.append({
            "expr_idx": expr_idx,
            "initial_cost": initial_cost,
            "final_cost": final_cost,
            "cost_reduction_pct": round(cost_red, 2),
            "initial_noise": round(initial_noise, 2),
            "final_noise": round(final_noise, 2),
            "steps": steps,
        })

        if (expr_idx + 1) % 10 == 0:
            print(f"  {expr_idx+1}/{len(expressions)} done  "
                  f"(last CR={cost_red:.1f}%, noise={final_noise:.1f})")

    all_results = []
    for r in per_expr_results:
        for budget in TEST_BUDGETS:
            all_results.append({
                "Budget": budget,
                "Expression #": r["expr_idx"] + 1,
                "Initial Cost": r["initial_cost"],
                "Final Cost": r["final_cost"],
                "Cost Reduction (%)": r["cost_reduction_pct"],
                "Initial Noise": r["initial_noise"],
                "Final Noise": r["final_noise"],
                "Budget Violated": "YES" if r["final_noise"] > budget else "NO",
                "Noise Margin": round(budget - r["final_noise"], 2),
                "Steps": r["steps"],
            })

    df = pd.DataFrame(all_results)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)

        summary_rows = []
        for b in TEST_BUDGETS:
            bdf = df[df["Budget"] == b]
            viol = (bdf["Budget Violated"] == "YES").sum()
            summary_rows.append({
                "Budget": b,
                "Num Expressions": len(bdf),
                "Avg Cost Reduction (%)": round(bdf["Cost Reduction (%)"].mean(), 2),
                "Avg Final Noise": round(bdf["Final Noise"].mean(), 2),
                "Violations": int(viol),
                "Violation Rate (%)": round(viol / len(bdf) * 100, 1) if len(bdf) > 0 else 0,
                "Agent": "Unconstrained PPO (no budget)",
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary_per_budget", index=False)

    print(f"\nResults written to {OUTPUT}")
    print(f"\nSummary:")
    print(f"  {'Budget':>10}  {'N':>4}  {'CR%':>8}  {'Viol':>5}  {'Viol%':>6}")
    for s in summary_rows:
        print(f"  {s['Budget']:>10}  {s['Num Expressions']:>4}  "
              f"{s['Avg Cost Reduction (%)']:>8.1f}  {s['Violations']:>5}  "
              f"{s['Violation Rate (%)']:>6.1f}")


if __name__ == "__main__":
    main()
