#!/usr/bin/env python3
"""Find the best expression for Demo 4: Unconstrained vs Constrained comparison.

Tests benchmark expressions with both agents, writing results to a log file
to avoid conda run buffering issues.
"""

import os
import sys
import importlib
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(RL_DIR)

sys.path.insert(0, RL_DIR)
sys.path.insert(0, os.path.join(RL_DIR, "pytrs"))
os.chdir(RL_DIR)

LOG_FILE = os.path.join(PROJECT_ROOT, "demo4_search.log")
log_fh = open(LOG_FILE, "w")

def log(msg=""):
    log_fh.write(msg + "\n")
    log_fh.flush()

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from pytrs import NoiseEstimator
from fhe_rl.utils import create_rules
from fhe_rl.env import fheEnv
from fhe_rl.policy import HierarchicalMaskablePolicy

sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")

UNCONSTRAINED_MODEL = os.path.join(RL_DIR, "fhe_rl", "trained_models", "agent_dynamic_llm_data.zip")
CONSTRAINED_MODEL = os.path.join(RL_DIR, "eval", "best_model_model_14529416_nato_sc", "best_model.zip")
BENCHMARKS_FILE = os.path.join(RL_DIR, "fhe_rl", "datasets", "benchmarks.txt")

UNCONSTRAINED_BUDGET_OPTIONS = [240, 300, 1_000_000]
CONSTRAINED_BUDGET_OPTIONS = [230, 236, 369, 9_000_000]

# Focus on expressions with multiplications (likely to produce high noise)
# and test at budget 230 (tight) and 369 (relaxed)
TEST_BUDGETS = [230, 369]

# Indices of promising candidates: those with multiplications that are likely
# to produce noise above ~230 bits after optimization
CANDIDATE_INDICES = list(range(43))  # all benchmarks


def load_expressions_named(filepath):
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    result = []
    for line in lines:
        if ":" in line:
            expr, name = line.rsplit(":", 1)
            result.append((expr, name))
        else:
            result.append((line, f"expr_{len(result)}"))
    return result


def run_agent_on_expr(model, env, vec_env, noise_estimator, budget=9_000_000):
    vec_env.set_options({"budget": budget})
    obs = vec_env.reset()

    fhe_env = vec_env.envs[0].env
    initial_cost = fhe_env.initial_cost
    initial_expr = fhe_env.initial_expression
    initial_noise = float(noise_estimator.estimate(initial_expr))

    trajectory = [{"cost": initial_cost, "noise": initial_noise, "expr": initial_expr}]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)
        done = bool(dones[0])

        info = infos[0]
        cp_cost = info.get("cost", fhe_env.current_cost)
        cp_noise = float(info.get("noise", 0.0))
        cp_expr = info.get("expression", fhe_env.expression)
        trajectory.append({"cost": cp_cost, "noise": cp_noise, "expr": cp_expr})

    final = trajectory[-1]
    valid = [cp for cp in trajectory if cp["noise"] <= budget]
    if valid:
        best = min(valid, key=lambda c: c["cost"])
    else:
        best = trajectory[0]

    return {
        "initial_cost": initial_cost,
        "initial_noise": initial_noise,
        "agent_final_cost": final["cost"],
        "agent_final_noise": final["noise"],
        "rollback_cost": best["cost"],
        "rollback_noise": best["noise"],
        "steps": len(trajectory) - 1,
    }


def main():
    log("Loading shared components...")
    rules_list = create_rules("rules.txt")
    rules_list["END"] = None
    noise_estimator = NoiseEstimator()

    from fhe_rl.__main__ import load_embeddings_from_config
    embeddings_model, _ = load_embeddings_from_config()

    all_exprs = load_expressions_named(BENCHMARKS_FILE)
    log(f"Loaded {len(all_exprs)} benchmark expressions")

    # ── Unconstrained agent setup ──
    log("Setting up UNCONSTRAINED agent...")
    unc_env = fheEnv(
        rules_list=rules_list,
        expressions=[all_exprs[0][0]],
        max_positions=16,
        embeddings_model=embeddings_model,
        budget_options=UNCONSTRAINED_BUDGET_OPTIONS,
        constraint_method="none",
        verbose=False,
    )
    unc_vec = DummyVecEnv([lambda: Monitor(unc_env)])
    unc_model = PPO(policy=HierarchicalMaskablePolicy, env=unc_vec)
    unc_model = unc_model.load(UNCONSTRAINED_MODEL)
    log("  Loaded.")

    # ── Constrained agent setup ──
    log("Setting up CONSTRAINED agent (NATO-SC)...")
    con_env = fheEnv(
        rules_list=rules_list,
        expressions=[all_exprs[0][0]],
        max_positions=16,
        embeddings_model=embeddings_model,
        budget_options=CONSTRAINED_BUDGET_OPTIONS,
        constraint_method="nato_sc",
        verbose=False,
    )
    con_vec = DummyVecEnv([lambda: Monitor(con_env)])
    con_model = PPO(policy=HierarchicalMaskablePolicy, env=con_vec)
    con_model = con_model.load(CONSTRAINED_MODEL)
    log("  Loaded.")

    header = f"{'Idx':>3} {'Name':>25} {'Budget':>6} | {'UNC_noise':>10} {'UNC_CR%':>8} {'UNC>B':>5} | {'CON_noise':>10} {'CON_CR%':>8} {'CON>B':>5} | Result"
    log("=" * len(header))
    log(header)
    log("=" * len(header))

    candidates = []

    for idx in CANDIDATE_INDICES:
        if idx >= len(all_exprs):
            continue
        expr, name = all_exprs[idx]

        for budget in TEST_BUDGETS:
            t0 = time.time()

            # Run unconstrained (always at 9M, no constraint)
            unc_env.expressions = [expr]
            unc_env.current_index = 0
            unc_res = run_agent_on_expr(unc_model, unc_env, unc_vec, noise_estimator, budget=9_000_000)

            # Run constrained at specific budget
            con_env.expressions = [expr]
            con_env.current_index = 0
            con_res = run_agent_on_expr(con_model, con_env, con_vec, noise_estimator, budget=budget)

            elapsed = time.time() - t0

            unc_noise = unc_res["agent_final_noise"]
            unc_cr = ((unc_res["initial_cost"] - unc_res["agent_final_cost"]) / unc_res["initial_cost"] * 100) if unc_res["initial_cost"] > 0 else 0
            unc_violated = unc_noise > budget

            con_noise = con_res["rollback_noise"]
            con_cost = con_res["rollback_cost"]
            con_cr = ((con_res["initial_cost"] - con_cost) / con_res["initial_cost"] * 100) if con_res["initial_cost"] > 0 else 0
            con_violated = con_noise > budget

            good = unc_violated and not con_violated and con_cr > 10
            result = "*** GOOD ***" if good else ""

            log(f"{idx:>3} {name:>25} {budget:>6} | "
                f"{unc_noise:>10.1f} {unc_cr:>7.1f}% {'YES':>5} | "
                f"{con_noise:>10.1f} {con_cr:>7.1f}% {'YES' if con_violated else 'no':>5} | "
                f"{result}  ({elapsed:.1f}s)") if unc_violated else \
            log(f"{idx:>3} {name:>25} {budget:>6} | "
                f"{unc_noise:>10.1f} {unc_cr:>7.1f}% {'no':>5} | "
                f"{con_noise:>10.1f} {con_cr:>7.1f}% {'YES' if con_violated else 'no':>5} | "
                f"{result}  ({elapsed:.1f}s)")

            if good:
                candidates.append({
                    "idx": idx, "name": name, "budget": budget,
                    "unc_noise": unc_noise, "unc_cr": unc_cr,
                    "con_noise": con_noise, "con_cr": con_cr,
                    "initial_noise": unc_res["initial_noise"],
                })

    log("")
    log("=" * 80)
    log(f"Found {len(candidates)} good candidates:")
    for c in candidates:
        log(f"  {c['name']:>25} @ B={c['budget']}: "
            f"UNC noise={c['unc_noise']:.1f} (VIOLATES), "
            f"CON noise={c['con_noise']:.1f} CR={c['con_cr']:.1f}% (WITHIN)")

    if candidates:
        best = max(candidates, key=lambda c: c["unc_noise"] - c["con_noise"])
        log(f"\nBEST: {best['name']} @ B={best['budget']}")
        log(f"  Unconstrained: noise={best['unc_noise']:.1f} (exceeds B={best['budget']})")
        log(f"  Constrained:   noise={best['con_noise']:.1f}, CR={best['con_cr']:.1f}% (within budget)")
        log(f"  Noise gap:     {best['unc_noise'] - best['con_noise']:.1f} bits")
    else:
        log("\nNo good candidates found.")

    log_fh.close()


if __name__ == "__main__":
    main()
