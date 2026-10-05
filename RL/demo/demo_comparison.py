#!/usr/bin/env python3
"""
Demo 4: Unconstrained vs Constrained Agent Comparison

Runs the SAME expression with:
  1. Old unconstrained agent (Bilel & Raouf PPO) -- NO safety rollback
     Setup matches scripts/test_unconstrained_standalone.py exactly.
  2. New constrained agent (NATO-SC + Lag-PID + FiLM) -- WITH safety rollback
     Setup matches fhe_rl/test.py::test_agent_v2 exactly.

Expression: local idx 23 in load_expressions(benchmarks_69_augmented.txt)
  = Excel Expression #47 from test_aug_unconstrained_ppo.xlsx
  Initial noise: 283.8 bits, Initial cost: 73772
  UNC final noise in Excel: 459.0 (violates B=369 by +90 bits)
  CON safe noise in Excel: 41.8 (within B=369, CR=82.9%)
"""

import os
import sys
import time
import importlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RL_DIR.parent

sys.path.insert(0, str(RL_DIR))
sys.path.insert(0, str(RL_DIR / "pytrs"))
os.chdir(str(RL_DIR))

# Alias for cloudpickle deserialization of old model
sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")

from demo.demo_utils import (
    RESET, BOLD, DIM, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, GRAY,
    CHECK, CROSS, ARROW, STAR, DIAMOND, BULLET,
    print_banner, print_section, print_subsection, print_summary_box,
    print_kv, print_step, print_phase, print_exec_phase,
    print_waiting, print_done, noise_bar, generate_trajectory_plot,
    save_input_expression, save_rl_trajectory, save_safety_rollback,
    save_comparison_summary,
)

# ── Configuration ───────────────────────────────────────────────────────────

BENCHMARKS_FILE = "fhe_rl/datasets/benchmarks_69_augmented.txt"
LOCAL_EXPR_IDX = 23  # load_expressions() index matching Excel #47

BUDGET = 369

# Unconstrained agent (Bilel & Raouf, Oct 2025)
UNC_MODEL_PATH = "fhe_rl/trained_models/agent_dynamic_llm_data.zip"
UNC_EMB_PATH = "fhe_rl/trained_models/embeddings_ROT_15_32_5m_10742576.pth"
UNC_BUDGET_OPTIONS = [9_000_000]

# Constrained agent (NATO-SC + Lag-PID + FiLM, job 14529416)
CON_MODEL_PATH = "eval/best_model_model_14529416_nato_sc/best_model.zip"
CON_BUDGET_OPTIONS = [230, 236, 369, 9_000_000]

MAX_POSITIONS = 16


def load_target_expression():
    from fhe_rl.utils import load_expressions
    all_exprs = load_expressions(BENCHMARKS_FILE)
    print(f"  {GRAY}Loaded {len(all_exprs)} expressions from augmented benchmarks{RESET}")
    return all_exprs, all_exprs[LOCAL_EXPR_IDX]


def run_unconstrained(expression, all_expressions, rules_list, noise_estimator):
    """Run unconstrained agent EXACTLY as scripts/test_unconstrained_standalone.py does."""
    from fhe_rl.utils import load_embeddings
    from fhe_rl.env import fheEnv
    from stable_baselines3 import PPO
    from pytrs import parse_sexpr

    print_section("UNCONSTRAINED AGENT (Bilel & Raouf, 2024)")
    print(f"  {RED}{CROSS}{RESET} Safety rollback: {BOLD}DISABLED{RESET}")
    print(f"  {GRAY}Model: {UNC_MODEL_PATH}{RESET}")
    print(f"  {GRAY}Setup: budget_options=[9M], constraint_method=none{RESET}")
    print()

    print_waiting("Loading embeddings (TRAE)")
    embeddings, _ = load_embeddings(
        tokenizer_type="dynamic",
        checkpoint_path=UNC_EMB_PATH,
    )
    print_done()

    print_waiting("Loading model")
    model = PPO.load(UNC_MODEL_PATH)
    model_obs_keys = set(
        list(model.observation_space.spaces.keys())
        if hasattr(model.observation_space, "spaces") else []
    )
    print_done(f"obs keys: {model_obs_keys}")

    env = fheEnv(
        rules_list, [expression],
        max_positions=MAX_POSITIONS,
        embeddings_model=embeddings,
        budget_options=UNC_BUDGET_OPTIONS,
        constraint_method="none",
        verbose=False,
    )

    obs, info = env.reset(options={"budget": 9_000_000})
    initial_cost = env.initial_cost
    initial_expr = env.initial_expression
    initial_noise = float(noise_estimator.estimate(parse_sexpr(initial_expr)))

    print_kv("Initial cost:", str(initial_cost))
    print_kv("Initial noise:", f"{initial_noise:.1f} bits")
    print()

    trajectory = [{"step": 0, "cost": initial_cost, "noise": initial_noise, "expr": initial_expr, "rule": "initial"}]
    costs = [initial_cost]
    noises = [initial_noise]

    done = False
    step_num = 0
    max_steps = env.max_steps
    t_start = time.perf_counter()

    while not done:
        obs_for_model = {
            k: v for k, v in obs.items()
            if k in model_obs_keys or k == "action_mask" or k == "observation"
        }

        action, _ = model.predict(obs_for_model, deterministic=True)
        action = int(action) if hasattr(action, "__len__") and len(action) == 1 else int(action)
        old_cost = env.current_cost

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_num += 1

        rule_idx = action // MAX_POSITIONS
        pos_idx = action % MAX_POSITIONS
        rule_name = list(rules_list.keys())[rule_idx]

        new_cost = env.current_cost
        cp_expr = env.expression
        noise = float(noise_estimator.estimate(parse_sexpr(cp_expr)))

        if rule_name == "END":
            print(f"\n  {CYAN}--- Step {step_num} {'─' * 52}{RESET}")
            print(f"    {BOLD}Rule:{RESET} {MAGENTA}END{RESET}  {GRAY}(agent chose to stop){RESET}")
        else:
            print_step(step_num, max_steps, rule_name, pos_idx, old_cost, new_cost, noise, BUDGET)

        trajectory.append({"step": step_num, "cost": new_cost, "noise": noise, "expr": cp_expr, "rule": rule_name})
        costs.append(new_cost)
        noises.append(noise)

    rl_time = time.perf_counter() - t_start

    final = trajectory[-1]
    cost_reduction = ((initial_cost - final["cost"]) / initial_cost * 100) if initial_cost > 0 else 0
    violated = final["noise"] > BUDGET
    status_color = RED if violated else GREEN
    status_icon = CROSS if violated else CHECK
    status_text = (f"VIOLATED ({final['noise']:.1f} > {BUDGET})"
                   if violated else f"WITHIN ({final['noise']:.1f} <= {BUDGET})")

    print()
    print(f"  {BOLD}Result:{RESET}")
    print(f"    Cost reduction: {BOLD}{cost_reduction:.1f}%{RESET}")
    print(f"    Final noise:    {BOLD}{final['noise']:.1f} bits{RESET}  /  Budget: {BUDGET} bits")
    print(f"    Status:         {status_color}{BOLD}{status_icon} {status_text}{RESET}")
    print(f"    Time:           {rl_time:.1f}s  ({step_num} steps)")
    print()

    save_rl_trajectory("demo4", trajectory, BUDGET, rl_time, prefix="unconstrained_")

    return {
        "name": "Unconstrained (Bilel & Raouf)",
        "initial_cost": initial_cost,
        "initial_noise": initial_noise,
        "final_cost": final["cost"],
        "final_noise": final["noise"],
        "cost_reduction": cost_reduction,
        "violated": violated,
        "steps": step_num,
        "time": rl_time,
        "costs": costs,
        "noises": noises,
        "final_step": step_num,
    }


def run_constrained(expression, all_expressions, rules_list, noise_estimator):
    """Run constrained agent with safety rollback, matching test_agent_v2."""
    from fhe_rl.env import fheEnv
    from fhe_rl.policy import HierarchicalMaskablePolicy
    from fhe_rl.__main__ import load_embeddings_from_config
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from pytrs import parse_sexpr

    print_section("CONSTRAINED AGENT (NATO-SC + Lag-PID + FiLM)")
    print(f"  {GREEN}{CHECK}{RESET} Safety rollback: {BOLD}ENABLED{RESET}")
    print(f"  {GRAY}Model: {CON_MODEL_PATH}{RESET}")
    print(f"  {GRAY}Setup: budget_options={CON_BUDGET_OPTIONS}, constraint_method=nato_sc{RESET}")
    print()

    print_waiting("Loading embeddings (GNNAE)")
    emb_model, _ = load_embeddings_from_config()
    print_done()

    print_waiting("Loading model")
    env = DummyVecEnv([lambda: Monitor(fheEnv(
        rules_list, [expression],
        max_positions=MAX_POSITIONS,
        embeddings_model=emb_model,
        budget_options=CON_BUDGET_OPTIONS,
        constraint_method="nato_sc",
        verbose=False,
    ))])

    model = PPO(policy=HierarchicalMaskablePolicy, env=env)
    model = model.load(str(CON_MODEL_PATH))
    print_done()

    env.set_options({"budget": BUDGET})
    obs = env.reset()
    fhe_env = env.envs[0].env
    initial_cost = fhe_env.initial_cost
    initial_expr = fhe_env.initial_expression
    initial_noise = float(noise_estimator.estimate(parse_sexpr(initial_expr)))

    print_kv("Initial cost:", str(initial_cost))
    print_kv("Initial noise:", f"{initial_noise:.1f} bits")
    print()

    trajectory = [{"step": 0, "cost": initial_cost, "noise": initial_noise, "expr": initial_expr, "rule": "initial"}]
    costs = [initial_cost]
    noises = [initial_noise]

    done = False
    step_num = 0
    max_steps = fhe_env.max_steps
    t_start = time.perf_counter()

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        old_cost = fhe_env.current_cost

        obs, rewards, dones, infos = env.step(action)
        done = bool(dones[0])
        step_num += 1

        info = infos[0]
        rule_idx = int(action[0]) // MAX_POSITIONS
        pos_idx = int(action[0]) % MAX_POSITIONS
        rule_name = list(rules_list.keys())[rule_idx]

        if done:
            cp_expr = info.get("expression", fhe_env.expression)
            cp_cost = info.get("cost", fhe_env.current_cost)
        else:
            cp_expr = fhe_env.expression
            cp_cost = fhe_env.current_cost
        noise = float(noise_estimator.estimate(parse_sexpr(cp_expr)))

        if rule_name == "END":
            print(f"\n  {CYAN}--- Step {step_num} {'─' * 52}{RESET}")
            print(f"    {BOLD}Rule:{RESET} {MAGENTA}END{RESET}  {GRAY}(agent chose to stop){RESET}")
        else:
            print_step(step_num, max_steps, rule_name, pos_idx, old_cost, cp_cost, noise, BUDGET)

        trajectory.append({"step": step_num, "cost": cp_cost, "noise": noise, "expr": cp_expr, "rule": rule_name})
        costs.append(cp_cost)
        noises.append(noise)

    rl_time = time.perf_counter() - t_start

    # Safety rollback
    print_subsection("Safety Rollback")
    agent_final = trajectory[-1]
    valid_checkpoints = [cp for cp in trajectory if cp["noise"] <= BUDGET]
    violated_steps = [cp for cp in trajectory if cp["noise"] > BUDGET]

    if valid_checkpoints:
        best = min(valid_checkpoints, key=lambda c: c["cost"])
        safety_activated = (best["step"] != agent_final["step"])
        final_cost = best["cost"]
        final_noise = best["noise"]
        final_step = best["step"]

        print(f"  {GREEN}{CHECK}{RESET} {len(valid_checkpoints)}/{len(trajectory)} checkpoints within budget")
        if violated_steps:
            print(f"  {YELLOW}{STAR}{RESET} {len(violated_steps)} checkpoints exceeded budget")
        if safety_activated:
            print(f"  {YELLOW}{STAR}{RESET} Rolled back from step {len(trajectory)-1} to step {final_step}")
        else:
            print(f"  {GREEN}{CHECK}{RESET} Agent's final state already optimal within budget")
    else:
        final_cost = initial_cost
        final_noise = initial_noise
        final_step = 0
        print(f"  {YELLOW}{STAR}{RESET} No valid checkpoints -- returned initial expression")

    cost_reduction = ((initial_cost - final_cost) / initial_cost * 100) if initial_cost > 0 else 0
    violated = final_noise > BUDGET
    status_color = RED if violated else GREEN
    status_icon = CROSS if violated else CHECK
    status_text = (f"VIOLATED ({final_noise:.1f} > {BUDGET})"
                   if violated else f"WITHIN ({final_noise:.1f} <= {BUDGET})")

    print()
    print(f"  {BOLD}Result:{RESET}")
    print(f"    Cost reduction: {BOLD}{cost_reduction:.1f}%{RESET}")
    print(f"    Final noise:    {BOLD}{final_noise:.1f} bits{RESET}  /  Budget: {BUDGET} bits")
    print(f"    Status:         {status_color}{BOLD}{status_icon} {status_text}{RESET}")
    print(f"    Time:           {rl_time:.1f}s  ({step_num} steps)")
    print()

    save_rl_trajectory("demo4", trajectory, BUDGET, rl_time, prefix="constrained_")
    save_safety_rollback("demo4", trajectory, BUDGET, final_step,
                         final_step != trajectory[-1]["step"], prefix="constrained_")

    return {
        "name": "Constrained (NATO-SC)",
        "initial_cost": initial_cost,
        "initial_noise": initial_noise,
        "final_cost": final_cost,
        "final_noise": final_noise,
        "cost_reduction": cost_reduction,
        "violated": violated,
        "steps": step_num,
        "time": rl_time,
        "costs": costs,
        "noises": noises,
        "final_step": final_step,
    }


def generate_comparison_plot(unc_result, con_result, budget, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    agents = [
        (0, unc_result, "Unconstrained Agent (old)", "#E74C3C"),
        (1, con_result, "Constrained Agent (ours)", "#27AE60"),
    ]

    # Shared y-limits for noise row and cost row
    all_noises = unc_result["noises"] + con_result["noises"]
    all_costs = unc_result["costs"] + con_result["costs"]
    noise_lo = min(all_noises) - 20
    noise_hi = max(max(all_noises), budget) + 30
    cost_lo = min(all_costs) * 0.9
    cost_hi = max(all_costs) * 1.05

    for col, result, title_prefix, color_main in agents:
        steps = list(range(len(result["noises"])))
        noises_vals = result["noises"]
        costs_vals = result["costs"]
        fs = result.get("final_step", len(noises_vals) - 1)

        # ── Top row: Noise trajectory ────────────────────────────────
        ax_n = axes[0][col]
        ax_n.axhline(y=budget, color="#E67E22", linewidth=2, linestyle="--",
                     label=f"Budget = {budget} bits", zorder=1)
        above = [n if n > budget else budget for n in noises_vals]
        ax_n.fill_between(steps, budget, above, alpha=0.12, color="#E74C3C",
                          label="Violation zone")
        ax_n.plot(steps, noises_vals, color=color_main, linewidth=2, marker="o",
                  markersize=3, label="Noise (bits)", zorder=3)
        if fs < len(noises_vals):
            ax_n.scatter([fs], [noises_vals[fs]], color=color_main, s=120, zorder=5,
                         edgecolors="black", linewidth=2,
                         label=f"Selected (step {fs})")
        ax_n.set_ylim(noise_lo, noise_hi)
        ax_n.set_ylabel("Estimated Noise (bits)", fontsize=11)
        violated_str = "VIOLATES" if result["violated"] else "WITHIN BUDGET"
        ax_n.set_title(f"{title_prefix}\nNoise={result['final_noise']:.1f} → {violated_str}",
                       fontsize=12, fontweight="bold")
        ax_n.legend(loc="best", fontsize=8, framealpha=0.9)
        ax_n.grid(True, alpha=0.25, linestyle=":")

        # ── Bottom row: Cost trajectory ──────────────────────────────
        ax_c = axes[1][col]
        ax_c.plot(steps, costs_vals, color="#3498DB", linewidth=2, marker="s",
                  markersize=3, label="Cost", zorder=3)
        ax_c.axhline(y=result["initial_cost"], color="#95A5A6", linewidth=1,
                     linestyle=":", label="Initial cost", zorder=1)
        if fs < len(costs_vals):
            ax_c.scatter([fs], [costs_vals[fs]], color="#3498DB", s=120, zorder=5,
                         edgecolors="black", linewidth=2,
                         label=f"Selected (step {fs})")
        cr = result["cost_reduction"]
        ax_c.set_ylim(cost_lo, cost_hi)
        ax_c.set_xlabel("Optimization Step", fontsize=11)
        ax_c.set_ylabel("Cost", fontsize=11)
        ax_c.set_title(f"Cost Reduction = {cr:.1f}%", fontsize=12, fontweight="bold")
        ax_c.legend(loc="best", fontsize=8, framealpha=0.9)
        ax_c.grid(True, alpha=0.25, linestyle=":")

    fig.suptitle(f"Demo 4 — Unconstrained vs Constrained Agent  (Budget = {budget} bits)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {GREEN}{CHECK}{RESET} Comparison plot saved: {CYAN}{output_path}{RESET}")


def main():
    from pytrs import NoiseEstimator
    from fhe_rl.utils import create_rules

    print_banner(
        "CHEHAB -- Agent Comparison Demo",
        "Unconstrained (old) vs Constrained (new) on Same Expression"
    )

    print(f"  {WHITE}{BOLD}Budget:{RESET}     {YELLOW}{BOLD}B = {BUDGET} bits{RESET}")
    print()
    print(f"  {GRAY}This demo runs the same expression with two agents:{RESET}")
    print(f"  {RED}{BULLET}{RESET} {BOLD}Old unconstrained agent{RESET}: optimizes cost freely, ignores noise budget")
    print(f"     {GRAY}No safety rollback -- raw agent output shown{RESET}")
    print(f"  {GREEN}{BULLET}{RESET} {BOLD}New constrained agent{RESET}: NATO-SC + Lag-PID + FiLM + safety rollback")
    print(f"     {GRAY}Budget-aware optimization with safety guarantees{RESET}")
    print()

    all_exprs, expression = load_target_expression()
    print(f"  {GRAY}Target: local idx {LOCAL_EXPR_IDX} "
          f"({len(expression)} chars, {len(expression.split())} AST nodes){RESET}\n")

    print_waiting("Loading rewrite rules")
    rules_list = create_rules("rules.txt")
    rules_list["END"] = None
    print_done(f"{len(rules_list)} rules")

    noise_estimator = NoiseEstimator()

    # ── Run unconstrained agent (matches standalone test exactly) ────
    unc_result = run_unconstrained(expression, all_exprs, rules_list, noise_estimator)

    time.sleep(1)

    # ── Run constrained agent (matches test_agent_v2 with safety) ────
    con_result = run_constrained(expression, all_exprs, rules_list, noise_estimator)

    # ── Side-by-side comparison ──────────────────────────────────────
    print_section("COMPARISON SUMMARY")

    header = f"  {'':>25} {'UNCONSTRAINED':>18}   {'CONSTRAINED':>18}"
    sep    = f"  {'':>25} {'─' * 18}   {'─' * 18}"
    print(header)
    print(sep)
    print(f"  {'Initial cost:':>25} {str(unc_result['initial_cost']):>18}   {str(con_result['initial_cost']):>18}")
    print(f"  {'Final cost:':>25} {str(unc_result['final_cost']):>18}   {str(con_result['final_cost']):>18}")
    print(f"  {'Cost reduction:':>25} {unc_result['cost_reduction']:>17.1f}%   {con_result['cost_reduction']:>17.1f}%")
    print(sep)
    print(f"  {'Initial noise:':>25} {unc_result['initial_noise']:>15.1f} b   {con_result['initial_noise']:>15.1f} b")
    print(f"  {'Final noise:':>25} {unc_result['final_noise']:>15.1f} b   {con_result['final_noise']:>15.1f} b")
    print(f"  {'Budget:':>25} {BUDGET:>15} b   {BUDGET:>15} b")

    unc_status = f"{RED}{CROSS} VIOLATED{RESET}" if unc_result["violated"] else f"{GREEN}{CHECK} WITHIN{RESET}"
    con_status = f"{RED}{CROSS} VIOLATED{RESET}" if con_result["violated"] else f"{GREEN}{CHECK} WITHIN{RESET}"
    print(f"  {'Budget status:':>25}    {unc_status:>18}      {con_status}")
    print(sep)
    print(f"  {'Safety rollback:':>25} {'DISABLED':>18}   {'ENABLED':>18}")
    print(f"  {'Steps:':>25} {unc_result['steps']:>18}   {con_result['steps']:>18}")
    print(f"  {'Time:':>25} {unc_result['time']:>16.1f} s   {con_result['time']:>16.1f} s")
    print()

    if unc_result["violated"] and not con_result["violated"]:
        print(f"  {GREEN}{BOLD}{STAR} The constrained agent respects the noise budget{RESET}")
        print(f"  {GREEN}{BOLD}  while achieving {con_result['cost_reduction']:.1f}% cost reduction.{RESET}")
        print(f"  {RED}{BOLD}{CROSS} The unconstrained agent exceeds the budget by "
              f"{unc_result['final_noise'] - BUDGET:.1f} bits.{RESET}")
    elif not unc_result["violated"] and not con_result["violated"]:
        print(f"  {YELLOW}{STAR} Both agents stay within budget on this expression.{RESET}")
        print(f"  {GRAY}  Try a different expression or budget for a clearer contrast.{RESET}")
    print()

    save_input_expression("demo4", expression, f"idx {LOCAL_EXPR_IDX}", BUDGET)
    save_comparison_summary("demo4", unc_result, con_result, BUDGET,
                            "Unconstrained", "Constrained+Safety")

    # ── Generate comparison plot ─────────────────────────────────────
    plot_path = str(PROJECT_ROOT / "demo4_comparison.png")
    generate_comparison_plot(unc_result, con_result, BUDGET, plot_path)

    print_section("DEMO COMPLETE")
    print(f"  {GREEN}{BOLD}{CHECK} Comparison finished successfully.{RESET}")
    print(f"  {GRAY}Plot saved: demo4_comparison.png{RESET}")
    print()


if __name__ == "__main__":
    main()
