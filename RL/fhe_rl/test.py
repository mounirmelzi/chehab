from .logger import log_test_results
from stable_baselines3 import PPO
import sys, importlib
from .utils import load_expressions, load_embeddings, create_rules
from pytrs import parse_sexpr, NoiseEstimator
from .env import fheEnv
from .policy import HierarchicalMaskablePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import os
import torch


def test_agent(expressions_file: str, embeddings_model, model_filepath: str, noise_budget: int):
    expressions = load_expressions(expressions_file)
    rules_list = create_rules("rules.txt")
    rules_list["END"] = None
    results = []
    max_positions = 16

    env = DummyVecEnv(
        [
            lambda: Monitor(
                fheEnv(
                    rules_list,
                    expressions,
                    max_positions=max_positions,
                    embeddings_model=embeddings_model,
                )
            )
        ]
    )

    env.set_options({ "budget": noise_budget })

    model = PPO(policy=HierarchicalMaskablePolicy, env=env)
    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    model = model.load(model_filepath)
    noise_estimator = NoiseEstimator()

    for _ in range(len(expressions)):
        obs = env.reset()

        wrapper = env.envs[0]
        fhe_env = wrapper.env

        test_expr = fhe_env.initial_expression
        initial_cost = fhe_env.initial_cost
        initial_noise = noise_estimator.estimate(parse_sexpr(test_expr))

        done = False
        steps = 0
        last_expr = None

        while not done:
            last_expr = fhe_env.expression
            last_cost = fhe_env.current_cost

            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            done = bool(dones[0])
            steps += 1

        last_noise = noise_estimator.estimate(parse_sexpr(last_expr))

        results.append(
            {
                "Test Expression": test_expr,
                "Final Expression": last_expr,
                "Initial Cost": initial_cost,
                "Final Cost": last_cost,
                "Steps": steps,
                "Initial Noise Used": initial_noise,
                "Final Noise Used": last_noise,
            }
        )

    job_id = os.environ.get("SLURM_JOB_ID", "jobid")
    sheet_name = f"HierarchicalPPO_Test_{job_id}"
    log_test_results(results, sheet_name=sheet_name)
