#!/usr/bin/env python3
"""Test unconstrained agent on threshold benchmarks with safety rollback.

Runs the agent once per expression (budget-unaware), then evaluates the
trajectory against each budget post-hoc.  Accepts --benchmark and --budgets
CLI arguments for flexibility.
"""

import sys
import os
import argparse
import importlib

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RL"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RL", "pytrs"))

sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")

from stable_baselines3 import PPO
from pytrs import parse_sexpr, NoiseEstimator
from fhe_rl.utils import load_expressions, load_embeddings, create_rules
from fhe_rl.env import fheEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=str,
                    default="./fhe_rl/datasets/benchmarks.txt")
    ap.add_argument("--budgets", type=str, default="230,236,369")
    ap.add_argument("--output", type=str,
                    default="./test_results/threshold_unconstrained.xlsx")
    ap.add_argument("--model", type=str,
                    default="./fhe_rl/trained_models/agent_dynamic_llm_data.zip")
    args = ap.parse_args()

    test_budgets = [int(b.strip()) for b in args.budgets.split(",")]

    os.chdir(os.path.join(os.path.dirname(__file__), "..", "RL"))

    expressions = load_expressions(args.benchmark)
    rules_list = create_rules("rules.txt")
    rules_list["END"] = None
    noise_estimator = NoiseEstimator()
    emb_path = "./fhe_rl/trained_models/embeddings_ROT_15_32_5m_10742576.pth"
    embeddings, _ = load_embeddings(tokenizer_type="dynamic",
                                    checkpoint_path=emb_path)

    model = PPO.load(args.model)
    obs_keys = (list(model.observation_space.spaces.keys())
                if hasattr(model.observation_space, "spaces") else [])
    print(f"Model: {args.model}")
    print(f"  Obs keys: {obs_keys}")
    print(f"  Expressions: {len(expressions)}")
    print(f"  Budgets: {test_budgets}")

    env = fheEnv(
        rules_list, expressions, max_positions=16,
        embeddings_model=embeddings,
        budget_options=[9_000_000],
        constraint_method="none",
    )

    model_obs_keys = set(obs_keys)

    trajectories = []
    print(f"\nRunning unconstrained agent on {len(expressions)} expressions...")
    for expr_idx in range(len(expressions)):
        obs, info = env.reset(options={"budget": 9_000_000})

        initial_expr = env.initial_expression
        initial_cost = env.initial_cost
        initial_noise = float(noise_estimator.estimate(parse_sexpr(initial_expr)))

        checkpoints = [{
            "step": 0, "expression": initial_expr,
            "cost": initial_cost, "noise": initial_noise,
        }]

        done = False
        steps = 0
        while not done:
            obs_for_model = {
                k: v for k, v in obs.items()
                if k in model_obs_keys or k in ("action_mask", "observation")
            }
            action, _ = model.predict(obs_for_model, deterministic=True)
            action = int(action) if hasattr(action, "__len__") and len(action) == 1 else int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            checkpoints.append({
                "step": steps,
                "expression": info["expression"],
                "cost": info["cost"],
                "noise": float(info["noise"]),
            })

        trajectories.append({
            "expr_idx": expr_idx,
            "initial_expr": initial_expr,
            "initial_cost": initial_cost,
            "initial_noise": initial_noise,
            "checkpoints": checkpoints,
            "total_steps": steps,
        })

        if (expr_idx + 1) % 10 == 0:
            agent_cr = ((initial_cost - checkpoints[-1]["cost"]) / initial_cost * 100) if initial_cost > 0 else 0
            print(f"  {expr_idx+1}/{len(expressions)} done  "
                  f"(agent CR={agent_cr:.1f}%, steps={steps})")

    all_results = []
    for traj in trajectories:
        checkpoints = traj["checkpoints"]
        agent_final = checkpoints[-1]
        initial_cost = traj["initial_cost"]
        initial_noise = traj["initial_noise"]

        for budget in test_budgets:
            is_feasible = initial_noise <= budget

            valid_cps = [cp for cp in checkpoints if cp["noise"] <= budget]
            if valid_cps:
                best = min(valid_cps, key=lambda cp: cp["cost"])
                safe_cost = best["cost"]
                safe_noise = best["noise"]
                best_valid_step = best["step"]
                safety_activated = (best["step"] != agent_final["step"])
            else:
                safe_cost = initial_cost
                safe_noise = initial_noise
                best_valid_step = 0
                safety_activated = True

            agent_cr = ((initial_cost - agent_final["cost"]) / initial_cost * 100) if initial_cost > 0 else 0
            safe_cr = ((initial_cost - safe_cost) / initial_cost * 100) if initial_cost > 0 else 0

            all_results.append({
                "Budget": budget,
                "Expression #": traj["expr_idx"] + 1,
                "Is Feasible": is_feasible,
                "Initial Cost": initial_cost,
                "Initial Noise": round(initial_noise, 2),
                "Agent Final Cost": agent_final["cost"],
                "Agent Final Noise": round(agent_final["noise"], 2),
                "Agent Cost Reduction (%)": round(agent_cr, 2),
                "Agent Violated": agent_final["noise"] > budget,
                "Safe Final Cost": safe_cost,
                "Safe Final Noise": round(safe_noise, 2),
                "Safe Cost Reduction (%)": round(safe_cr, 2),
                "Safe Violated": safe_noise > budget,
                "Safety Activated": safety_activated,
                "Best Valid Step": best_valid_step,
                "Total Steps": traj["total_steps"],
                "Noise Margin": round(budget - safe_noise, 2),
                "Trajectory Costs": "|".join(str(cp["cost"]) for cp in checkpoints),
                "Trajectory Noises": "|".join(f'{cp["noise"]:.2f}' for cp in checkpoints),
            })

    df = pd.DataFrame(all_results)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)

        summary_rows = []
        for b in test_budgets:
            bdf = df[df["Budget"] == b]
            summary_rows.append({
                "Budget": b,
                "Num Expressions": len(bdf),
                "Feasible Count": int(bdf["Is Feasible"].sum()),
                "Infeasible Count": int((~bdf["Is Feasible"]).sum()),
                "Agent Avg Cost Red (%)": round(bdf["Agent Cost Reduction (%)"].mean(), 2),
                "Agent Violations": int(bdf["Agent Violated"].sum()),
                "Agent Violation Rate (%)": round(bdf["Agent Violated"].mean() * 100, 1),
                "Safe Avg Cost Red (%)": round(bdf["Safe Cost Reduction (%)"].mean(), 2),
                "Safe Violations": int(bdf["Safe Violated"].sum()),
                "Safe Violation Rate (%)": round(bdf["Safe Violated"].mean() * 100, 1),
                "Safety Activations": int(bdf["Safety Activated"].sum()),
                "Avg Noise Margin": round(bdf["Noise Margin"].mean(), 2),
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary_per_budget", index=False)

        for label, filt in [("feasible_summary", True), ("infeasible_summary", False)]:
            rows = []
            for b in test_budgets:
                sub = df[(df["Budget"] == b) & (df["Is Feasible"] == filt)]
                if len(sub) == 0:
                    continue
                kind = "Feasible" if filt else "Infeasible"
                row = {
                    "Budget": b,
                    f"{kind} Expressions": len(sub),
                    "Agent Avg Cost Red (%)": round(sub["Agent Cost Reduction (%)"].mean(), 2),
                    "Agent Violations": int(sub["Agent Violated"].sum()),
                    "Agent Violation Rate (%)": round(sub["Agent Violated"].mean() * 100, 1),
                    "Safe Avg Cost Red (%)": round(sub["Safe Cost Reduction (%)"].mean(), 2),
                    "Safe Violations": int(sub["Safe Violated"].sum()),
                    "Safe Violation Rate (%)": round(sub["Safe Violated"].mean() * 100, 1),
                    "Safety Activations": int(sub["Safety Activated"].sum()),
                }
                rows.append(row)
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name=label, index=False)

    print(f"\nResults written to {args.output}")
    print(f"\nFeasible Summary:")
    print(f"  {'Budget':>10}  {'N':>4}  {'AgentCR%':>9}  {'SafeCR%':>8}  {'SafeViol':>8}")
    for b in test_budgets:
        fdf = df[(df["Budget"] == b) & df["Is Feasible"]]
        if len(fdf) == 0:
            continue
        print(f"  {b:>10}  {len(fdf):>4}  "
              f"{fdf['Agent Cost Reduction (%)'].mean():>9.2f}  "
              f"{fdf['Safe Cost Reduction (%)'].mean():>8.2f}  "
              f"{int(fdf['Safe Violated'].sum()):>8}")


if __name__ == "__main__":
    main()
