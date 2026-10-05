from stable_baselines3 import PPO
import time
from .utils import load_expressions, create_rules, parse_sexpr, calc_vec_sizes
import sys
import importlib
from stable_baselines3.common.vec_env import DummyVecEnv
from .config import get_env_class, get_policy_class
from stable_baselines3.common.monitor import Monitor


def run_agent_mo(expressions_file: str, embeddings_model, model_filepath: str, output_file: str, w_ops: float, w_keys: float):
    start_time = time.perf_counter()
    expressions = load_expressions(expressions_file)
    if not len(expressions):
        print("No valid expressions found in the file.")
        sys.exit(1)
        return
    
    from pytrs.rules import _create_rules_mo
    rules_list = _create_rules_mo("rules.txt", "rotations_rules.txt")
    rules_list["END"] = None
    print(f"DEBUG: len(rules_list) = {len(rules_list)}")
    max_positions = 16
    
    EnvCls = get_env_class()
    PolicyCls = get_policy_class()
    
    env = DummyVecEnv([
        lambda: Monitor(EnvCls(rules_list, expressions, max_positions=max_positions, embeddings_model=embeddings_model))
    ])
    
    # LOCK the preference vector to the one provided by the user
    w = [w_ops, w_keys]
    env.env_method("set_preference_vector", w)

    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    import fhe_rl.policy
    # Monkey patch the policy class to ensure SB3 unpickles the correct MO class
    fhe_rl.policy.HierarchicalMaskablePolicy = PolicyCls
    
    model = PPO.load(
        model_filepath, 
        env=env, 
        custom_objects={
            "policy_class": PolicyCls,
        }
    )
    
    obs = env.reset()
    wrapper  = env.envs[0]
    fhe_env   = wrapper.env
    test_expr = fhe_env.initial_expression
    initial_cost = fhe_env.initial_cost
    done = False
    steps = 0
    
    while not done:
        last_expr = fhe_env.expression
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        done = bool(dones[0])
        steps += 1

    parsed = parse_sexpr(last_expr)
    vec_sizes=" ".join(str(x) for x in calc_vec_sizes(parsed))
    with open (output_file, "w") as f:
        f.write(last_expr+"\n"+vec_sizes)
    
    end_time = time.perf_counter()
    print(f"Elapsed inference time: {end_time - start_time} seconds")
