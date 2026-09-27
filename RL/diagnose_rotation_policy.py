"""
Run from RL/ with venv active:  python diagnose_rotation_policy.py

Ajout : Affichage détaillé étape par étape AVEC l'évolution des expressions
mathématiques (formule initiale, formules intermédiaires, formule finale).
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

sys.path.insert(0, os.getcwd())

from fhe_rl.utils import load_expressions_named, create_rules, load_embeddings_from_config
from fhe_rl.env import fheEnv
from fhe_rl.policy import HierarchicalMaskablePolicy
from fhe_rl.config import get_model_path

MODEL_PATH = get_model_path("agent_model")
PREFS_TO_CHECK = [[0.8, 0.2], [1.0, 0.0]]
BUDGET_TO_CHECK = 300
MAX_POSITIONS = 16
MAX_STEPS_PER_EPISODE = 75

print(f"Loading model: {MODEL_PATH}")
embeddings, _ = load_embeddings_from_config()

if not os.path.exists("rotations_rules.txt"):
    print("WARNING: rotations_rules.txt not found in cwd.")

rules_list = create_rules("rules.txt", "rotations_rules.txt")
rules_list["END"] = None
rule_names = list(rules_list.keys())
rotate_rule_indices = [i for i, n in enumerate(rule_names) if "rotate" in n.lower() or n.lower().startswith("rot-")]
print(f"{len(rule_names)} rules loaded. rotate indices: {rotate_rule_indices} "
      f"-> {[rule_names[i] for i in rotate_rule_indices]}\n")

named_expressions = load_expressions_named("expression_dot_product.txt")
expressions = [e for e, _ in named_expressions]
names = [n for _, n in named_expressions]
env = fheEnv(
    rules_list,
    expressions,
    max_positions=MAX_POSITIONS,
    embeddings_model=embeddings,
    budget_options=[100, 200, 300],
    pref_list=PREFS_TO_CHECK,
    verbose=False,
)

model = PPO(policy=HierarchicalMaskablePolicy, env=env)
model = model.load(MODEL_PATH)
policy = model.policy
policy.eval()

print(f"policy.rule_dim = {policy.rule_dim}  (should equal {len(rule_names)})")
print(f"policy.max_positions = {policy.max_positions}  (should equal {MAX_POSITIONS})")
if policy.rule_dim != len(rule_names):
    print("*** MISMATCH: the loaded checkpoint's rule_dim does not match...")
print()


def obs_to_batch(obs):
    return {k: torch.as_tensor(np.asarray(v, dtype=np.float32)).unsqueeze(0) for k, v in obs.items()}


def rotate_probability(obs):
    obs_t = obs_to_batch(obs)
    with torch.no_grad():
        enc = policy.encoder(obs_t)
        mask = obs_t["action_mask"].bool().view(1, policy.rule_dim, policy.max_positions)
        rule_mask = mask.any(dim=2)
        rule_dist = policy._rule_dist(enc, rule_mask)
        rule_probs = rule_dist.probs.squeeze(0).numpy()
    return float(sum(rule_probs[r] for r in rotate_rule_indices if r < len(rule_probs))), rule_probs


def get_expr_string(env_instance, info_dict):
    """Fonction robuste pour extraire l'expression de l'environnement"""
    if "expression" in info_dict:
        return info_dict["expression"]
    if "expr" in info_dict:
        return info_dict["expr"]
    if hasattr(env_instance, "current_expr"):
        return str(env_instance.current_expr)
    if hasattr(env_instance, "expr"):
        return str(env_instance.expr)
    if hasattr(env_instance, "get_current_expression"):
        return str(env_instance.get_current_expression())
    return "[Expression introuvable - vérifiez l'attribut dans fheEnv]"


overall_max_rotate_prob = {tuple(p): 0.0 for p in PREFS_TO_CHECK}
overall_any_mask_open = {tuple(p): False for p in PREFS_TO_CHECK}
overall_rotate_taken = {tuple(p): 0 for p in PREFS_TO_CHECK}
overall_steps_checked = {tuple(p): 0 for p in PREFS_TO_CHECK}

import time
_t0 = time.time()

for expr_idx in range(len(expressions)):
    print(f"\n====================================================================")
    print(f"[{time.time()-_t0:7.1f}s] DEBUT EXPR {expr_idx+1}/{len(expressions)}")
    print(f"Nom du benchmark : {names[expr_idx]}")
    print(f"Texte du Dataset : {expressions[expr_idx]}")
    print(f"====================================================================")
    
    for pref in PREFS_TO_CHECK:
        print(f"\n--- Test avec Préférence (w_ops, w_keys) : {pref} ---")
        obs, info = env.reset(options={"budget": BUDGET_TO_CHECK})
        env.set_preference_vector(pref)
        obs["preference_vector"] = np.array(pref, dtype=np.float32)

        # Affichage de l'arbre parsé initialement
        initial_expr = get_expr_string(env, info)
        print(f"  [Init] Formule de départ : {initial_expr}")

        for step in range(MAX_STEPS_PER_EPISODE):
            mask = env.get_action_mask()
            mask_rotate_open = any(
                mask[r * MAX_POSITIONS: r * MAX_POSITIONS + MAX_POSITIONS].sum() > 0
                for r in rotate_rule_indices
            )

            rmass, _ = rotate_probability(obs)
            overall_steps_checked[tuple(pref)] += 1

            if mask_rotate_open:
                overall_any_mask_open[tuple(pref)] = True
                overall_max_rotate_prob[tuple(pref)] = max(overall_max_rotate_prob[tuple(pref)], rmass)

            with torch.no_grad():
                action, _ = model.predict(obs, deterministic=True)
            
            action = int(np.asarray(action).reshape(-1)[0])
            taken_rule_idx = action // MAX_POSITIONS
            taken_position = action % MAX_POSITIONS
            rule_name = rule_names[taken_rule_idx]

            if taken_rule_idx in rotate_rule_indices:
                overall_rotate_taken[tuple(pref)] += 1
                rotate_alert = "!!! ROTATION CHOISIE !!!"
            else:
                rotate_alert = ""

            print(f"  Step {step:02d} | Rotation permise : {'OUI' if mask_rotate_open else 'NON'}")
            if mask_rotate_open:
                print(f"           | Probabilité donnée aux rotations : {rmass*100:.2f}%")
            print(f"           | Action choisie : {rule_name} (position {taken_position}) {rotate_alert}")

            obs, reward, terminated, truncated, step_info = env.step(action)
            
            # --- EXTRACTION ET AFFICHAGE DE L'EXPRESSION INTERMEDIAIRE ---
            current_expr = get_expr_string(env, step_info)
            print(f"           | Formule actuelle : {current_expr}")
            
            if terminated or truncated:
                print(f"  -> Fin (Terminated: {terminated}, Truncated: {truncated}) | Récompense finale : {reward:.4f}")
                break

print("\n================ SUMMARY across all benchmark expressions ================\n")
for pref in PREFS_TO_CHECK:
    key = tuple(pref)
    print(f"preference {pref}  ({overall_steps_checked[key]} steps checked):")
    print(f"  Did ANY step ever have a legal rotate_* action in the mask? {overall_any_mask_open[key]}")
    print(f"  Max P(any rotate_*) the policy ever assigned at such a step: {overall_max_rotate_prob[key]:.8f}")
    print(f"  Number of times the policy actually TOOK a rotate_* action:  {overall_rotate_taken[key]}")
    print()