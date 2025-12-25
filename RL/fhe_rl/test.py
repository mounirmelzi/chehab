from .logger import log_test_results
from stable_baselines3 import PPO
import sys, importlib
from .utils import load_expressions, create_rules
from pytrs import parse_sexpr, estimate_expression_noise
from .env import fheEnv
from .policy import HierarchicalMaskablePolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
from typing import List, Optional
from .config import (
    get_budget_strategy,
    BudgetStrategy,
    BUDGET_OPTIONS,
    INFINITE_BUDGET_VALUE,
)


def _format_budget(budget: float) -> str:
    return "infinite" if budget >= INFINITE_BUDGET_VALUE else f"{budget:.0f}"


def test_agent(
    expressions_file: str,
    embeddings_model,
    model_filepath: str,
    budgets: Optional[List[float]] = None,
    budget_strategy_override: str | None = None,
):
    expressions = load_expressions(expressions_file)
    rules_list = create_rules("rules.txt")
    rules_list["END"] = None
    results = []
    max_positions = 16

    if budget_strategy_override:
        try:
            strategy = BudgetStrategy(budget_strategy_override)
        except ValueError as exc:
            raise ValueError(f"Unsupported budget strategy override: {budget_strategy_override}") from exc
    else:
        strategy = get_budget_strategy()
    if budgets is None or len(budgets) == 0:
        budgets = BUDGET_OPTIONS if strategy == BudgetStrategy.ONE_HOT else [300.0]

    # Create environment factory
    def make_env():
        return Monitor(
                fheEnv(
                    rules_list,
                    expressions,
                    max_positions=max_positions,
                    embeddings_model=embeddings_model,
                )
    )
    
    env = DummyVecEnv([make_env])

    model = PPO(policy=HierarchicalMaskablePolicy, env=env)
    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    model = model.load(model_filepath)

    for budget in budgets:
        wrapper = env.envs[0]
        fhe_env = wrapper.env
        fhe_env.set_test_budget(budget)

        for _ in range(len(expressions)):
            obs = env.reset()
        test_expr = fhe_env.initial_expression
        initial_cost = fhe_env.initial_cost
        initial_noise_info = estimate_expression_noise(parse_sexpr(test_expr))

        done = False
        steps = 0
            last_expr = test_expr
            last_cost = initial_cost

        while not done:
            last_expr = fhe_env.expression
            last_cost = fhe_env.current_cost
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])
            steps += 1

        last_noise_info = estimate_expression_noise(parse_sexpr(last_expr))
        results.append(
            {
                    "Budget Strategy": strategy.value,
                    "Budget": _format_budget(budget),
                "Test Expression": test_expr,
                "Final Expression": last_expr,
                "Initial Cost": initial_cost,
                "Final Cost": last_cost,
                "Steps": steps,
                "Initial Noise Used": initial_noise_info["noise_used"],
                "Final Noise Used": last_noise_info["noise_used"],
            }
        )

        fhe_env.clear_test_budget()

    job_id = os.environ.get("SLURM_JOB_ID", "jobid")
    sheet_name = f"HierarchicalPPO_Test_{job_id}"
    log_test_results(results, sheet_name=sheet_name)
