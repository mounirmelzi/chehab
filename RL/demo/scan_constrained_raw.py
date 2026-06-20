#!/usr/bin/env python3
"""Scan: find expressions where constrained agent raw output stays within budget locally."""
import sys, os, importlib, time
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RL_DIR))
sys.path.insert(0, str(RL_DIR / "pytrs"))
os.chdir(str(RL_DIR))
sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")

from fhe_rl.utils import load_expressions, create_rules
from fhe_rl.env import fheEnv
from fhe_rl.policy import HierarchicalMaskablePolicy
from fhe_rl.__main__ import load_embeddings_from_config
from pytrs import NoiseEstimator, parse_sexpr
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

BENCHMARKS = "fhe_rl/datasets/benchmarks_69_augmented.txt"
CON_MODEL = "eval/best_model_model_14529416_nato_sc/best_model.zip"
UNC_MODEL = "fhe_rl/trained_models/agent_dynamic_llm_data.zip"
UNC_EMB = "fhe_rl/trained_models/embeddings_ROT_15_32_5m_10742576.pth"
BUDGET = 369
CON_BUDGETS = [230, 236, 369, 9_000_000]

# Candidate local indices (from match_expressions.py -- expressions where UNC violates)
CANDIDATES = [21, 23, 60, 62, 28, 63, 29, 64, 30, 65, 31, 66, 32, 67, 33, 68]

exprs = load_expressions(BENCHMARKS)
rules_list = create_rules("rules.txt")
rules_list["END"] = None
ne = NoiseEstimator()

emb_model, _ = load_embeddings_from_config()

print(f"Scanning {len(CANDIDATES)} candidates with constrained agent (raw, no rollback)")
print(f"Budget = {BUDGET}")
print(f"{'idx':>4} {'init_n':>8} {'CON_raw':>8} {'CON_within':>11} {'steps':>6}")
print("-" * 45)
sys.stdout.flush()

for idx in CANDIDATES:
    expr = exprs[idx]

    env = DummyVecEnv([lambda e=expr: Monitor(fheEnv(
        rules_list, [e], max_positions=16,
        embeddings_model=emb_model,
        budget_options=CON_BUDGETS,
        constraint_method="nato_sc",
        verbose=False,
    ))])

    model = PPO(policy=HierarchicalMaskablePolicy, env=env)
    model = model.load(CON_MODEL)

    env.set_options({"budget": BUDGET})
    obs = env.reset()
    fhe_env = env.envs[0].env
    init_noise = float(ne.estimate(parse_sexpr(fhe_env.initial_expression)))

    done = False
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        done = bool(dones[0])
        steps += 1

    if done:
        final_expr = infos[0].get("expression", fhe_env.expression)
    else:
        final_expr = fhe_env.expression
    final_noise = float(ne.estimate(parse_sexpr(final_expr)))

    within = "YES" if final_noise <= BUDGET else "no"
    marker = " <<<" if final_noise <= BUDGET else ""
    print(f"{idx:>4} {init_noise:>8.1f} {final_noise:>8.1f} {within:>11} {steps:>6}{marker}")
    sys.stdout.flush()

print("\nDone.")
