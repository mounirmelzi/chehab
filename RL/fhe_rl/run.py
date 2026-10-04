import os
import sys
import time
import importlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from pytrs import NoiseEstimator
from .utils import load_expressions, create_rules, parse_sexpr, calc_vec_sizes
from .config import get_env_class, get_policy_class
from .env import fheEnv


def run_agent(expressions_file: str, embeddings_model, model_filepath: str,
              output_file: str, noise_budget: int = 230, w_ops: float = 1.0, w_keys: float = 0.0,
              budget_options: list = None, constraint_method: str = None):

    pref = [w_ops, w_keys]

    # CORRECTION DE LA REGRESSION : Si constraint_method n'est pas spécifié,
    # on s'adapte par défaut au standard ou au modèle, évitant de forcer lagrangian_pid aveuglément.
    if constraint_method is None:
        constraint_method = "lagrangian_od_ov" if "imed" in model_filepath.lower() else "lagrangian_pid"

    if budget_options is None:
        budget_options = list(fheEnv.DEFAULT_BUDGET_OPTIONS)  # [230, 369, 9000]

    start_time = time.perf_counter()
    expressions = load_expressions(expressions_file)
    if not len(expressions):
        print("No valid expressions found in the file.")
        sys.exit(1)
        return
    print(expressions)

    # Resolve rule files using absolute paths relative to the fhe_rl package
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rules_path = os.path.join(base_dir, "rules.txt")
    rot_rules_path = os.path.join(base_dir, "rotations_rules.txt")

    if not os.path.exists(rot_rules_path):
        print(f"WARNING: {rot_rules_path} not found — "
              "rotation rules will be missing from the action space!")
    else:
        print(f"rotation rules loaded from: {rot_rules_path}")

    rules_list = create_rules(rules_path, rot_rules_path)
    rules_list["END"] = None
    max_positions = 16

    EnvCls = get_env_class()
    PolicyCls = get_policy_class()

    env = DummyVecEnv([
        lambda: Monitor(EnvCls(
            rules_list,
            expressions,
            max_positions=max_positions,
            embeddings_model=embeddings_model,
            budget_options=budget_options,
            constraint_method=constraint_method,
            pref_list=[pref] if hasattr(EnvCls, "pref_list") else None
        ))
    ])

    env.set_options({"budget": noise_budget})
    
    # Sécurité : n'applique set_preference_vector que si l'environnement le supporte
    try:
        env.env_method("set_preference_vector", pref)
    except AttributeError:
        pass

    model = PPO(
        policy=PolicyCls,
        env=env
    )

    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    model = model.load(model_filepath)

    obs = env.reset()
    wrapper = env.envs[0]
    fhe_env = wrapper.env

    assert fhe_env.budget == noise_budget, \
        f"budget not applied: env has {fhe_env.budget}, requested {noise_budget}"
    print(f"Noise budget (requested / used by env): {noise_budget} / {fhe_env.budget}")

    test_expr = fhe_env.initial_expression
    initial_cost = getattr(fhe_env, "initial_cost", 0)
    initial_ops = max(getattr(fhe_env, "initial_ops", initial_cost), 1e-9)
    initial_keys = max(getattr(fhe_env, "initial_keys", 1.0), 1.0)
    noise_estimator = NoiseEstimator()
    initial_noise = float(noise_estimator.estimate(parse_sexpr(test_expr)))

    def score(c_exec, c_keys):
        return w_ops * (c_exec / initial_ops) + w_keys * (c_keys / initial_keys)

    best_safe = None
    if initial_noise <= noise_budget:
        best_safe = {
            "expression": test_expr,
            "c_exec": initial_ops,
            "c_keys": initial_keys,
            "noise": initial_noise,
            "score": score(initial_ops, initial_keys),
        }
    else:
        print(f"WARNING: the INPUT expression already exceeds the budget "
              f"(noise {initial_noise:.1f} > {noise_budget}).")

    done = False
    steps = 0
    final_expr, final_exec, final_keys, final_noise = test_expr, initial_ops, initial_keys, initial_noise

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        done = bool(dones[0])
        steps += 1

        info = infos[0]
        expr = info.get("expression", getattr(fhe_env, "expression", test_expr))
        c_exec = info.get("c_exec", getattr(fhe_env, "curr_ops", initial_ops))
        c_keys = info.get("c_keys", getattr(fhe_env, "curr_keys", initial_keys))
        noise = float(info.get("noise", float("inf")))

        if noise <= noise_budget:
            sc = score(c_exec, c_keys)
            if best_safe is None or sc < best_safe["score"]:
                best_safe = {"expression": expr, "c_exec": c_exec, "c_keys": c_keys,
                             "noise": noise, "score": sc}

        if done:
            final_expr, final_exec, final_keys, final_noise = expr, c_exec, c_keys, noise

    agent_violated = final_noise > noise_budget
    if agent_violated:
        if best_safe is not None:
            print(f"WARNING: agent's final expression violates the budget "
                  f"(noise {final_noise:.1f} > {noise_budget}). Rolling back to the best safe "
                  f"expression (noise {best_safe['noise']:.1f}).")
            final_expr = best_safe["expression"]
            final_exec = best_safe["c_exec"]
            final_keys = best_safe["c_keys"]
            final_noise = best_safe["noise"]
        else:
            print(f"WARNING: no expression within budget {noise_budget} was found "
                  f"(final noise {final_noise:.1f}). Output may exceed the budget!")

    parsed = parse_sexpr(final_expr)
    vec_sizes = " ".join(str(x) for x in calc_vec_sizes(parsed))

    with open(output_file, "w") as f:
        f.write(final_expr + "\n" + vec_sizes)

    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    print(f"Optimization completed in {elapsed_seconds:.2f} seconds.")
    print(f"Final exec cost   : {final_exec}")
    print(f"Final keys cost   : {final_keys}")
    print(f"Final noise (est) : {final_noise:.2f} / budget {noise_budget} "
          f"[{'OK' if final_noise <= noise_budget else 'VIOLATED'}]")