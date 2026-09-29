import os
import sys
import time
import importlib
import pandas as pd
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from pytrs import parse_sexpr, NoiseEstimator
from .utils import load_expressions, load_expressions_named, create_rules
from .env import fheEnv
from .policy import HierarchicalMaskablePolicy


def _resolve_train_budgets(budget_options, test_budgets):
    """The one-hot size/order MUST match the checkpoint's training budgets.
    Falling back to the test budgets silently breaks that, so fall back to the
    env defaults instead and warn."""
    if budget_options is None:
        budget_options = list(fheEnv.DEFAULT_BUDGET_OPTIONS)
        print(f"WARNING: --train_budgets not given; assuming the model was trained on "
              f"{budget_options}. Pass --train_budgets explicitly if that is wrong.")
    return budget_options

# ═══════════════════════════════════════════════════════════════════════════════
#  test_agent (v1) — Standard Testing
# ═══════════════════════════════════════════════════════════════════════════════

def test_agent(
    expressions_file: str,
    embeddings_model,
    model_filepath: str,
    noise_budget: int = 300,
    budget_options: list = None,
    test_budgets: list = None,
    constraint_method: str = "lagrangian_pid",
    output_file: str = None,
):
    expressions = load_expressions(expressions_file)
    rules_list = create_rules("rules.txt", "rotations_rules.txt")
    rules_list["END"] = None
    max_positions = 16

    if test_budgets is None:
        test_budgets = [noise_budget]
    budget_options = _resolve_train_budgets(budget_options, test_budgets)

    env = DummyVecEnv([
        lambda: Monitor(
            fheEnv(
                rules_list,
                expressions,
                max_positions=max_positions,
                embeddings_model=embeddings_model,
                budget_options=budget_options,
                constraint_method=constraint_method,
            )
        )
    ])

    model = PPO(policy=HierarchicalMaskablePolicy, env=env)
    # Compatibility hack for loading older models
    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    model = model.load(model_filepath)
    noise_estimator = NoiseEstimator()

    all_results = []

    for budget in test_budgets:
        print(f"\n{'='*60}")
        print(f"  Testing with budget = {budget}")
        print(f"{'='*60}")

        for expr_idx in range(len(expressions)):
            wrapper = env.envs[0]
            fhe_env = wrapper.env
            fhe_env.current_index = expr_idx  # pin cursor: neutralize DummyVecEnv auto-reset double-advance
            # FIX: DummyVecEnv consumes the options on the first reset() and clears them,
            # so the budget must be re-applied before EVERY reset.
            env.set_options({"budget": budget})
            obs = env.reset()
            assert fhe_env.budget == budget, f"budget not applied: {fhe_env.budget} != {budget}"

            test_expr = fhe_env.initial_expression
            initial_exec = fhe_env.initial_ops
            initial_noise = noise_estimator.estimate(parse_sexpr(test_expr))

            done = False
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                done = bool(dones[0])
                steps += 1

            terminal = infos[0]
            last_expr = terminal["expression"]
            last_exec = terminal["c_exec"]
            final_noise = float(terminal["noise"])
            cost_reduction = ((initial_exec - last_exec) / initial_exec * 100) if initial_exec > 0 else 0
            budget_violated = final_noise > budget

            all_results.append({
                "Budget": budget,
                "Expression #": expr_idx + 1,
                "Initial Expression": test_expr,
                "Final Expression": last_expr,
                "Initial Exec Cost": initial_exec,
                "Final Exec Cost": last_exec,
                "Exec Cost Reduction (%)": round(cost_reduction, 2),
                "Initial Noise": round(initial_noise, 2),
                "Final Noise": round(final_noise, 2),
                "Budget Violated": budget_violated,
                "Noise Margin": round(budget - final_noise, 2),
                "Steps": steps,
            })

            status = "VIOLATED" if budget_violated else "OK"
            print(f"  [{status}] Expr {expr_idx+1}: Exec {initial_exec}->{last_exec} "
                  f"({cost_reduction:+.1f}%), noise {initial_noise:.0f}->{final_noise:.0f}")

    df = pd.DataFrame(all_results)
    if output_file is None:
        model_name = os.path.basename(model_filepath).replace(".zip", "")
        output_file = f"test_results_{model_name}.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)
        print(f"\nResults written to: {output_file}")


# ═══════════════════════════════════════════════════════════════════════════════
#  test_agent_v2 — Trajectory Checkpointing + Safety Rollback
# ═══════════════════════════════════════════════════════════════════════════════

def test_agent_v2(
    expressions_file: str,
    embeddings_model,
    model_filepath: str,
    noise_budget: int = 300,
    budget_options: list = None,
    test_budgets: list = None,
    constraint_method: str = "lagrangian_pid",
    output_file: str = None,
    save_optimized: str = None,
):
    named_pairs = load_expressions_named(expressions_file)
    expr_names = [name for _, name in named_pairs]
    expressions = [expr for expr, _ in named_pairs]
    
    rules_list = create_rules("rules.txt", "rotations_rules.txt")
    rules_list["END"] = None
    max_positions = 16

    if test_budgets is None:
        test_budgets = [noise_budget]
    budget_options = _resolve_train_budgets(budget_options, test_budgets)

    env = DummyVecEnv([
        lambda: Monitor(
            fheEnv(
                rules_list,
                expressions,
                max_positions=max_positions,
                embeddings_model=embeddings_model,
                budget_options=budget_options,
                constraint_method=constraint_method,
            )
        )
    ])

    model = PPO(policy=HierarchicalMaskablePolicy, env=env)
    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    model = model.load(model_filepath)
    noise_estimator = NoiseEstimator()

    all_results = []

    for budget in test_budgets:
        print(f"\n{'='*60}")
        print(f"  Testing (v2) with budget = {budget}")
        print(f"{'='*60}")

        for expr_idx in range(len(expressions)):
            t0 = time.perf_counter()

            wrapper = env.envs[0]
            fhe_env = wrapper.env
            fhe_env.current_index = expr_idx  # pin cursor: neutralize auto-reset double-advance
            # FIX: re-apply the budget before EVERY reset (options are cleared after one reset)
            env.set_options({"budget": budget})
            obs = env.reset()
            assert fhe_env.budget == budget, f"budget not applied: {fhe_env.budget} != {budget}"

            w_vec = fhe_env.current_w
            n_budget = fhe_env.n_budget

            test_expr = fhe_env.initial_expression
            initial_exec = fhe_env.initial_ops
            initial_keys = fhe_env.initial_keys
            initial_noise = float(noise_estimator.estimate(parse_sexpr(test_expr)))

            is_feasible = initial_noise <= budget

            checkpoints = [{
                "step": 0,
                "expression": test_expr,
                "c_exec": initial_exec,
                "c_keys": initial_keys,
                "noise": initial_noise,
            }]

            done = False
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                done = bool(dones[0])
                steps += 1

                if done:
                    terminal = infos[0]
                    cp_expr = terminal["expression"]
                    cp_exec = terminal["c_exec"]
                    cp_keys = terminal["c_keys"]
                    cp_noise = float(terminal["noise"])
                else:
                    cp_expr = fhe_env.expression
                    cp_exec = fhe_env.curr_ops
                    cp_keys = fhe_env.curr_keys
                    cp_noise = float(noise_estimator.estimate(parse_sexpr(cp_expr)))

                checkpoints.append({
                    "step": steps,
                    "expression": cp_expr,
                    "c_exec": cp_exec,
                    "c_keys": cp_keys,
                    "noise": cp_noise,
                })

            rl_time_ms = (time.perf_counter() - t0) * 1000
            agent_final = checkpoints[-1]

            # ── MORL Safety Rollback: Pick best valid checkpoint via scalarized J(e, w) ──
            valid_checkpoints = [cp for cp in checkpoints if cp["noise"] <= budget]
            
            if valid_checkpoints:
                def compute_J(cp):
                    norm_exec = cp["c_exec"] / max(1e-6, initial_exec)
                    norm_keys = cp["c_keys"] / n_budget
                    return (w_vec[0] * norm_exec) + (w_vec[1] * norm_keys)

                best = min(valid_checkpoints, key=compute_J)
                safe_expr = best["expression"]
                safe_exec = best["c_exec"]
                safe_keys = best["c_keys"]
                safe_noise = best["noise"]
                best_valid_step = best["step"]
                safety_activated = (best["step"] != agent_final["step"])
            else:
                safe_expr = test_expr
                safe_exec = initial_exec
                safe_keys = initial_keys
                safe_noise = initial_noise
                best_valid_step = 0
                safety_activated = True

            agent_cr = ((initial_exec - agent_final["c_exec"]) / initial_exec * 100) if initial_exec > 0 else 0
            safe_cr = ((initial_exec - safe_exec) / initial_exec * 100) if initial_exec > 0 else 0

            expr_name = expr_names[expr_idx] if expr_idx < len(expr_names) else f"expr_{expr_idx}"

            all_results.append({
                "Budget": budget,
                "Expression #": expr_idx + 1,
                "Expression Name": expr_name,
                "Is Feasible": is_feasible,
                "Initial Exec": initial_exec,
                "Initial Keys": initial_keys,
                "Initial Noise": round(initial_noise, 2),
                "Agent Final Exec": agent_final["c_exec"],
                "Agent Final Keys": agent_final["c_keys"],
                "Agent Final Noise": round(agent_final["noise"], 2),
                "Agent Cost Reduction (%)": round(agent_cr, 2),
                "Agent Violated": agent_final["noise"] > budget,
                "Safe Final Exec": safe_exec,
                "Safe Final Keys": safe_keys,
                "Safe Final Noise": round(safe_noise, 2),
                "Safe Cost Reduction (%)": round(safe_cr, 2),
                "Safe Violated": safe_noise > budget,
                "Safety Activated": safety_activated,
                "Best Valid Step": best_valid_step,
                "Total Steps": steps,
                "RL Time (ms)": round(rl_time_ms, 1),
                "Noise Margin": round(budget - safe_noise, 2),
                "_safe_expr": safe_expr,
            })

            tag = "FEASIBLE" if is_feasible else "INFEAS"
            safe_tag = "SAFE" if safe_noise <= budget else "VIOL"
            print(f"  [{tag}][{safe_tag}] {expr_name}: "
                  f"Exec {initial_exec}->{safe_exec} ({safe_cr:+.1f}%), "
                  f"Keys {initial_keys}->{safe_keys}, "
                  f"Noise {initial_noise:.0f}->{safe_noise:.0f}")

    # Write results
    excel_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
    df = pd.DataFrame(excel_results)

    if output_file is None:
        model_name = os.path.basename(model_filepath).replace(".zip", "")
        output_file = f"test_results_v2_{model_name}.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)

        feasible_rows = []
        for b in test_budgets:
            fdf = df[(df["Budget"] == b) & (df["Is Feasible"])]
            if len(fdf) == 0: continue
            feasible_rows.append({
                "Budget": b,
                "Feasible Expressions": len(fdf),
                "Agent Avg Cost Red (%)": round(fdf["Agent Cost Reduction (%)"].mean(), 2),
                "Agent Violation Rate (%)": round(fdf["Agent Violated"].mean() * 100, 1),
                "Safe Avg Cost Red (%)": round(fdf["Safe Cost Reduction (%)"].mean(), 2),
                "Safe Violation Rate (%)": round(fdf["Safe Violated"].mean() * 100, 1),
                "Safety Activations": int(fdf["Safety Activated"].sum()),
                "Avg Safe Noise Margin": round(fdf["Noise Margin"].mean(), 2),
            })
        if feasible_rows:
            pd.DataFrame(feasible_rows).to_excel(writer, sheet_name="feasible_summary", index=False)

    print(f"\nResults (v2) written to: {output_file}")

    if save_optimized:
        best_per_expr = {}
        for r in all_results:
            name = r["Expression Name"]
            if name not in best_per_expr or r["Safe Cost Reduction (%)"] > best_per_expr[name]["Safe Cost Reduction (%)"]:
                best_per_expr[name] = r
        with open(save_optimized, "w") as f:
            for name, r in best_per_expr.items():
                safe_expr = r.get("_safe_expr", "")
                f.write(f"{safe_expr}:{name}\n")
        print(f"RL-optimized expressions saved to: {save_optimized}")