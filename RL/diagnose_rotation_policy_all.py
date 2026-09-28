"""
Run from RL/ with venv active:  python diagnose_rotation_policy_all.py

Version batch : teste tous les fichiers expression_<benchmark>.txt trouvés
dans le dossier courant, et affiche un résumé compact par benchmark
(pas le détail step-by-step, pour rester lisible sur 12 benchmarks).
"""
import os
import sys
import glob
import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, os.getcwd())

from fhe_rl.utils import load_expressions_named, create_rules, load_embeddings_from_config
from fhe_rl.env import fheEnv
from fhe_rl.policy import HierarchicalMaskablePolicy
from fhe_rl.config import get_model_path

MODEL_PATH = get_model_path("agent_model")
PREF_TO_CHECK = [1.0, 0.0]   # celle utilisée réellement par run_morl_benchmarks_budget.py
BUDGET_TO_CHECK = 300
MAX_POSITIONS = 16
MAX_STEPS_PER_EPISODE = 75

print(f"Loading model: {MODEL_PATH}")
embeddings, _ = load_embeddings_from_config()

rules_list = create_rules("rules.txt", "rotations_rules.txt")
rules_list["END"] = None
rule_names = list(rules_list.keys())
rotate_rule_indices = [
    i for i, n in enumerate(rule_names)
    if "rotate" in n.lower() or n.lower().startswith("rot-")
]
print(f"{len(rule_names)} rules loaded. rotate indices: {rotate_rule_indices}\n")


def obs_to_batch(obs):
    return {k: torch.as_tensor(np.asarray(v, dtype=np.float32)).unsqueeze(0) for k, v in obs.items()}


def rotate_probability(policy, obs):
    obs_t = obs_to_batch(obs)
    with torch.no_grad():
        enc = policy.encoder(obs_t)
        mask = obs_t["action_mask"].bool().view(1, policy.rule_dim, policy.max_positions)
        rule_mask = mask.any(dim=2)
        rule_dist = policy._rule_dist(enc, rule_mask)
        rule_probs = rule_dist.probs.squeeze(0).numpy()
    return float(sum(rule_probs[r] for r in rotate_rule_indices if r < len(rule_probs)))


expr_files = sorted(glob.glob("expression_*.txt"))
if not expr_files:
    print("Aucun fichier expression_*.txt trouvé dans le dossier courant.")
    sys.exit(1)

results = []

for expr_file in expr_files:
    bench_name = expr_file.replace("expression_", "").replace(".txt", "")
    named_expressions = load_expressions_named(expr_file)
    expressions = [e for e, _ in named_expressions]

    env = fheEnv(
        rules_list,
        expressions,
        max_positions=MAX_POSITIONS,
        embeddings_model=embeddings,
        budget_options=[100, 200, 300],
        pref_list=[PREF_TO_CHECK],
        verbose=False,
    )
    model = PPO(policy=HierarchicalMaskablePolicy, env=env)
    model = model.load(MODEL_PATH)
    policy = model.policy
    policy.eval()

    any_mask_open = False
    max_rotate_prob = 0.0
    rotate_taken = 0
    steps_checked = 0
    rotate_step_names = []

    for expr_idx in range(len(expressions)):
        obs, info = env.reset(options={"budget": BUDGET_TO_CHECK})
        env.set_preference_vector(PREF_TO_CHECK)
        obs["preference_vector"] = np.array(PREF_TO_CHECK, dtype=np.float32)

        for step in range(MAX_STEPS_PER_EPISODE):
            mask = env.get_action_mask()
            mask_rotate_open = any(
                mask[r * MAX_POSITIONS: r * MAX_POSITIONS + MAX_POSITIONS].sum() > 0
                for r in rotate_rule_indices
            )
            rmass = rotate_probability(policy, obs)
            steps_checked += 1

            if mask_rotate_open:
                any_mask_open = True
                max_rotate_prob = max(max_rotate_prob, rmass)

            with torch.no_grad():
                action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            taken_rule_idx = action // MAX_POSITIONS

            if taken_rule_idx in rotate_rule_indices:
                rotate_taken += 1
                rotate_step_names.append(rule_names[taken_rule_idx])

            obs, reward, terminated, truncated, step_info = env.step(action)
            if terminated or truncated:
                break

    results.append({
        "bench": bench_name,
        "steps": steps_checked,
        "mask_open": any_mask_open,
        "max_prob": max_rotate_prob,
        "taken": rotate_taken,
        "rules_used": sorted(set(rotate_step_names)),
    })

print("\n" + "=" * 100)
print(f"{'Benchmark':<15} {'Steps':>6} {'MaskOpen':>9} {'MaxProb%':>9} {'Taken':>6}  RulesUtilisées")
print("=" * 100)
for r in results:
    print(f"{r['bench']:<15} {r['steps']:>6} {str(r['mask_open']):>9} "
          f"{r['max_prob']*100:>8.2f}% {r['taken']:>6}  {', '.join(r['rules_used']) or '-'}")
