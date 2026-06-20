#!/usr/bin/env python3
"""Quick scan: find expressions where unconstrained agent violates budget locally."""
import sys, importlib, time
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RL_DIR))
sys.path.insert(0, str(RL_DIR / "pytrs"))
import os; os.chdir(str(RL_DIR))

from fhe_rl.utils import load_expressions, create_rules
from fhe_rl.env import fheEnv
from fhe_rl.policy import HierarchicalMaskablePolicy
from fhe_rl.__main__ import load_embeddings_from_config
from pytrs import NoiseEstimator, parse_sexpr
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

AUGMENTED = str(RL_DIR / "fhe_rl" / "datasets" / "benchmarks_69_augmented.txt")
UNC_MODEL = str(RL_DIR / "fhe_rl" / "trained_models" / "agent_dynamic_llm_data.zip")
BUDGET = 369

exprs = load_expressions(AUGMENTED)
rules_list = create_rules("rules.txt")
rules_list["END"] = None
ne = NoiseEstimator()
emb_model, _ = load_embeddings_from_config()

env = DummyVecEnv([lambda: Monitor(fheEnv(
    rules_list, exprs, max_positions=16,
    embeddings_model=emb_model,
    budget_options=[240, 300, 1_000_000],
    constraint_method="none",
    verbose=False,
))])

model = PPO(policy=HierarchicalMaskablePolicy, env=env)
sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
model = model.load(UNC_MODEL)
env.set_options({"budget": 9_000_000})

print(f"Testing unconstrained agent on {len(exprs)} expressions (budget={BUDGET})")
print(f"{'#':>4} {'InitNoise':>10} {'FinalNoise':>11} {'CR%':>8} {'Status':>10}")
print("-" * 50)
sys.stdout.flush()

candidates = []
for idx in range(len(exprs)):
    obs = env.reset()
    fhe_env = env.envs[0].env
    init_cost = fhe_env.initial_cost
    init_noise = float(ne.estimate(parse_sexpr(fhe_env.initial_expression)))

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        done = bool(dones[0])

    final_noise = float(infos[0]["noise"])
    final_cost = infos[0]["cost"]
    cr = ((init_cost - final_cost) / init_cost * 100) if init_cost > 0 else 0
    violated = final_noise > BUDGET
    status = "VIOLATED" if violated else "ok"

    if violated or idx >= 42:
        print(f"{idx+1:>4} {init_noise:>10.1f} {final_noise:>11.1f} {cr:>7.1f}% {status:>10}")
        sys.stdout.flush()
        if violated:
            candidates.append((idx, init_noise, final_noise, cr))

print(f"\nTotal violations at B={BUDGET}: {len(candidates)}")
for idx, in_n, fn, cr in candidates:
    print(f"  Expr #{idx+1}: init={in_n:.1f}, final={fn:.1f}, CR={cr:.1f}%")
sys.stdout.flush()
