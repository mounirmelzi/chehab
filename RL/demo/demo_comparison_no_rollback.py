#!/usr/bin/env python3
"""
Demo 5: Constrained Agent Inherently Stays Within Budget (No Safety Rollback)

Both agents run WITHOUT safety rollback -- raw agent output only.
Shows that the constrained agent is inherently budget-aware: it never
exceeds the noise budget even without the safety net, while the
unconstrained agent violates freely.

Expression: local idx 60 in load_expressions(benchmarks_69_augmented.txt)
  = Excel Expression #52 from test_aug_unconstrained_ppo.xlsx
  Initial noise: 317.2 bits, Initial cost: 20273
  UNC final noise in Excel: 390.0 (violates B=369 by +21 bits, CR=29.3%)
  CON agent raw noise (local): 320.8 (within B=369, no rollback needed)
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
    save_input_expression, save_rl_trajectory, save_comparison_summary,
)

# ── Configuration ───────────────────────────────────────────────────────────

BENCHMARKS_FILE = "fhe_rl/datasets/benchmarks_69_augmented.txt"
LOCAL_EXPR_IDX = 60  # load_expressions() index matching Excel #52

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

    save_rl_trajectory("demo5", trajectory, BUDGET, rl_time, prefix="unconstrained_")

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
    """Run constrained agent WITHOUT safety rollback -- raw agent output only."""
    from fhe_rl.env import fheEnv
    from fhe_rl.policy import HierarchicalMaskablePolicy
    from fhe_rl.__main__ import load_embeddings_from_config
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from pytrs import parse_sexpr

    print_section("CONSTRAINED AGENT (NATO-SC + Lag-PID + FiLM)")
    print(f"  {RED}{CROSS}{RESET} Safety rollback: {BOLD}DISABLED{RESET}  {GRAY}(raw agent output){RESET}")
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

    # No safety rollback -- use raw agent final state
    final = trajectory[-1]
    final_cost = final["cost"]
    final_noise = final["noise"]
    final_step = step_num

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

    save_rl_trajectory("demo5", trajectory, BUDGET, rl_time, prefix="constrained_")

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

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    agents = [
        (0, unc_result, "Unconstrained Agent (old)", "#E74C3C"),
        (1, con_result, "Constrained Agent (ours)", "#27AE60"),
    ]

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
                         label=f"Final (step {fs})")
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
                         label=f"Final (step {fs})")
        cr = result["cost_reduction"]
        ax_c.set_ylim(cost_lo, cost_hi)
        ax_c.set_xlabel("Optimization Step", fontsize=11)
        ax_c.set_ylabel("Cost", fontsize=11)
        ax_c.set_title(f"Cost Reduction = {cr:.1f}%", fontsize=12, fontweight="bold")
        ax_c.legend(loc="best", fontsize=8, framealpha=0.9)
        ax_c.grid(True, alpha=0.25, linestyle=":")

    fig.suptitle(f"Demo 5 — Raw Agent Output (No Safety Rollback)  —  Budget = {budget} bits",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {GREEN}{CHECK}{RESET} Comparison plot saved: {CYAN}{output_path}{RESET}")


def main():
    from pytrs import NoiseEstimator
    from fhe_rl.utils import create_rules

    print_banner(
        "CHEHAB -- Inherent Budget Awareness Demo",
        "Both agents WITHOUT safety rollback -- raw output only"
    )

    print(f"  {WHITE}{BOLD}Budget:{RESET}     {YELLOW}{BOLD}B = {BUDGET} bits{RESET}")
    print()
    print(f"  {GRAY}This demo shows the constrained agent inherently respects the budget{RESET}")
    print(f"  {GRAY}even WITHOUT the safety rollback mechanism:{RESET}")
    print(f"  {RED}{BULLET}{RESET} {BOLD}Old unconstrained agent{RESET}: no budget awareness, no safety rollback")
    print(f"  {GREEN}{BULLET}{RESET} {BOLD}New constrained agent{RESET}: NATO-SC + Lag-PID + FiLM, {BOLD}no safety rollback{RESET}")
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
    print(f"  {'Safety rollback:':>25} {'DISABLED':>18}   {'DISABLED':>18}")
    print(f"  {'Steps:':>25} {unc_result['steps']:>18}   {con_result['steps']:>18}")
    print(f"  {'Time:':>25} {unc_result['time']:>16.1f} s   {con_result['time']:>16.1f} s")
    print()

    if unc_result["violated"] and not con_result["violated"]:
        print(f"  {GREEN}{BOLD}{STAR} The constrained agent INHERENTLY stays within budget{RESET}")
        print(f"  {GREEN}{BOLD}  even without safety rollback ({con_result['cost_reduction']:.1f}% CR).{RESET}")
        print(f"  {RED}{BOLD}{CROSS} The unconstrained agent exceeds the budget by "
              f"{unc_result['final_noise'] - BUDGET:.1f} bits.{RESET}")
        print(f"  {CYAN}{BULLET} The budget-aware training (NATO-SC + Lag-PID) teaches the agent{RESET}")
        print(f"  {CYAN}  to avoid high-noise rewrites -- no rollback needed.{RESET}")
    elif not unc_result["violated"] and not con_result["violated"]:
        print(f"  {YELLOW}{STAR} Both agents stay within budget on this expression.{RESET}")
        print(f"  {GRAY}  Try a different expression or budget for a clearer contrast.{RESET}")
    print()

    save_input_expression("demo5", expression, f"idx {LOCAL_EXPR_IDX}", BUDGET)
    save_comparison_summary("demo5", unc_result, con_result, BUDGET,
                            "Unconstrained (raw)", "Constrained (raw)")

    # ── Generate comparison plot ─────────────────────────────────────
    plot_path = str(PROJECT_ROOT / "demo5_no_rollback.png")
    generate_comparison_plot(unc_result, con_result, BUDGET, plot_path)

    print_section("DEMO COMPLETE")
    print(f"  {GREEN}{BOLD}{CHECK} Comparison finished successfully.{RESET}")
    print(f"  {GRAY}Plot saved: demo5_no_rollback.png{RESET}")
    print()


if __name__ == "__main__":
    main()
