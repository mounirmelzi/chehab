import os
import sys
import time
import importlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .utils import load_expressions, create_rules, parse_sexpr, calc_vec_sizes
from .env import fheEnv
from .policy import HierarchicalMaskablePolicy


def run_agent(expressions_file: str, embeddings_model, model_filepath: str, 
              output_file: str, noise_budget: int = 300, w_ops: float = 1.0, w_keys: float = 0.0):

    pref = [w_ops, w_keys]

    start_time = time.perf_counter()
    expressions = load_expressions(expressions_file)
    if not len(expressions):
        print("No valid expressions found in the file.")
        sys.exit(1)
        return
    print(expressions)
    
    # FIX: Resolve rule files using absolute paths relative to the fhe_rl package
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
    
    env = DummyVecEnv([
        lambda: Monitor(fheEnv(
            rules_list, 
            expressions, 
            max_positions=max_positions, 
            embeddings_model=embeddings_model, 
            budget_options=[300, 1000, 9000], 
            pref_list=[pref]
        ))
    ])
    
    env.set_options({ "budget": noise_budget })
    env.env_method("set_preference_vector", pref)
    
    model = PPO(
        policy=HierarchicalMaskablePolicy,
        env=env
    )
    
    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    model = model.load(model_filepath)
    
    obs = env.reset()
    wrapper   = env.envs[0]
    fhe_env   = wrapper.env
    
    test_expr = fhe_env.initial_expression
    initial_cost = fhe_env.initial_cost
    done = False
    steps = 0
    last_expr = None
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        done = bool(dones[0])
        steps += 1
        
        if done:
            terminal_info = infos[0]
            final_expr = terminal_info.get("expression", fhe_env.expression)
            final_exec = terminal_info.get("c_exec", fhe_env.curr_ops)
            final_keys = terminal_info.get("c_keys", fhe_env.curr_keys)

    parsed = parse_sexpr(final_expr)
    vec_sizes = " ".join(str(x) for x in calc_vec_sizes(parsed))
    
    with open(output_file, "w") as f:
        f.write(final_expr + "\n" + vec_sizes)
        
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    print(f"Optimization completed in {elapsed_seconds:.2f} seconds.")
    print(f"Final exec cost   : {final_exec}")
    print(f"Final keys cost   : {final_keys}")