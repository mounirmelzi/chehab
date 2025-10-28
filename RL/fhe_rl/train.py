import os
import math
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from .env    import fheEnv
from .policy import HierarchicalMaskablePolicy
from stable_baselines3 import PPO
from .utils  import load_expressions, create_rules, load_embeddings
from .logger import log_training_details
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize,DummyVecEnv
from .callbacks import linear_schedule, EntCoefScheduler
from stable_baselines3.common.callbacks import EvalCallback
from .config import get_rl_algorithm, RLAlgorithm
from .wrappers import LagrangianVecEnvWrapper, SuccessBonusKPenaltyWrapper
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent / "datasets"
RL_DIR = Path(__file__).resolve().parents[1]  # .../RL

def train_agent(expressions_file: str, embeddings_model, total_timesteps: int = 1_000_000, num_envs: int = 8):
    match get_rl_algorithm():
        case RLAlgorithm.PPO:
            train_ppo_agent(expressions_file, embeddings_model, total_timesteps, num_envs)
        case RLAlgorithm.LAGRANGIAN_PPO:
            train_lagrangian_ppo_agent(expressions_file, embeddings_model, total_timesteps, num_envs)
        case _:
            train_ppo_agent(expressions_file, embeddings_model, total_timesteps, num_envs)


def train_ppo_agent(expressions_file: str, embeddings_model, total_timesteps: int = 500_000, num_envs: int = 8):
    try:
        total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", str(total_timesteps)))
    except Exception:
        pass
    benchmarks = load_expressions(str(DATA_DIR / "benchmarks.txt"))
    expressions = load_expressions(expressions_file)
    max_positions = 16
    rules_list  = create_rules(str(RL_DIR / "rules.txt"))
    rules_list["END"] = None
    job_id = os.environ.get("SLURM_JOB_ID", "jobid")
    run_name = f"model_{job_id}"
    tensorboard_log_dir = f"./tensorboard/{run_name}"
    def make_env(): return Monitor(fheEnv(rules_list, expressions, max_positions=max_positions, embeddings_model=embeddings_model))
    env = SubprocVecEnv([make_env for _ in range(num_envs)], start_method='spawn')    
    val_env = DummyVecEnv([
    lambda: Monitor(fheEnv(rules_list, benchmarks, max_positions=max_positions,embeddings_model=embeddings_model))
    ])
    ent_schedule = linear_schedule(0.1)
    model_params = {
        "policy": HierarchicalMaskablePolicy,
        "env": env,
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "n_epochs": 15,
        "clip_range": 0.1,
        "clip_range_vf": 0.2,
        "ent_coef": 0.1,
        "verbose": 1,
        "tensorboard_log": tensorboard_log_dir,
        "policy_kwargs": {
            "ent_coef": 0.1,
            "rule_dim":      len(rules_list),
            "max_positions": max_positions,
            "rule_hidden_dims":   [128, 64],
            "pos_hidden_dims":    [64, 64],
            "value_hidden_dims":    [256, 128, 64],
        }
    }
    print(f"[START] Algo=PPO total_timesteps={total_timesteps} n_steps={model_params['n_steps']} num_envs={num_envs} lr={model_params['learning_rate']}", flush=True)
    model = PPO(**model_params)
    log_training_details(
        model_params,
        job_id,
        num_data=len(expressions),
        num_actions=len(rules_list),
        total_timesteps=total_timesteps,
        output_model_name=run_name,
        notes="2 level hierarchical PPO max steps 75 and 8 envs"
    )
    num_benchmarks = len(benchmarks)
    eval_callback = EvalCallback(
        val_env, 
        best_model_save_path=f"./eval/best_model_{run_name}", 
        log_path=tensorboard_log_dir, 
        eval_freq=10000,
        n_eval_episodes=num_benchmarks,
        deterministic=True, 
        render=False, 
        verbose=1
    )
    model.learn(
        total_timesteps=total_timesteps, 
        log_interval=1, 
        progress_bar=True, 
        callback=[eval_callback,EntCoefScheduler(ent_schedule)]
    )
    model.save(run_name)


def train_lagrangian_ppo_agent(expressions_file: str, embeddings_model, total_timesteps: int = 500_000, num_envs: int = 8):
    try:
        total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", str(total_timesteps)))
    except Exception:
        pass
    benchmarks = load_expressions(str(DATA_DIR / "benchmarks.txt")) 
    expressions = load_expressions(expressions_file)
    max_positions = 16
    RULES_PATH = RL_DIR / "rules.txt"
    rules_list  = create_rules(str(RULES_PATH))
    rules_list["END"] = None
    job_id = os.environ.get("SLURM_JOB_ID", "jobid")
    run_name = f"model_{job_id}"
    tensorboard_log_dir = f"./tensorboard/{run_name}"
    def make_env(): return Monitor(fheEnv(rules_list, expressions, max_positions=max_positions, embeddings_model=embeddings_model))

    env = SubprocVecEnv([
        make_env for _ in range(num_envs)
    ], start_method='spawn')
    constraint_wrapper = os.getenv("CONSTRAINT_WRAPPER", "terminal").lower()
    if constraint_wrapper == "success_k":
        env = SuccessBonusKPenaltyWrapper(env)
    else:
        env = LagrangianVecEnvWrapper(env)

    val_env = DummyVecEnv([
        lambda: Monitor(fheEnv(rules_list, benchmarks, max_positions=max_positions,embeddings_model=embeddings_model))
    ])
    if constraint_wrapper == "success_k":
        val_env = SuccessBonusKPenaltyWrapper(val_env)
    else:
        val_env = LagrangianVecEnvWrapper(val_env)

    ent_schedule = linear_schedule(0.1)
    model_params = {
        "policy": HierarchicalMaskablePolicy,
        "env": env,
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "n_epochs": 15,
        "clip_range": 0.1,
        "clip_range_vf": 0.2,
        "ent_coef": 0.1,
        "verbose": 1,
        "tensorboard_log": tensorboard_log_dir,
        "policy_kwargs": {
            "ent_coef": 0.1,
            "rule_dim":      len(rules_list),
            "max_positions": max_positions,
            "rule_hidden_dims":   [128, 64],
            "pos_hidden_dims":    [64, 64],
            "value_hidden_dims":    [256, 128, 64],
        }
    }
    model = PPO(**model_params)
    log_training_details(
        model_params,
        job_id,
        num_data=len(expressions),
        num_actions=len(rules_list),
        total_timesteps=total_timesteps,
        output_model_name=run_name,
        notes="2 level hierarchical PPO max steps 75 and 8 envs"
    )
    num_benchmarks = len(benchmarks)
    eval_callback = EvalCallback(
        val_env, 
        best_model_save_path=f"./eval/best_model_{run_name}", 
        log_path=tensorboard_log_dir, 
        eval_freq=10000,
        n_eval_episodes=num_benchmarks,
        deterministic=True, 
        render=False, 
        verbose=1
    )

    # Lagrangian PPO training ============== [Start] ==============

    tensorboard_writer = SummaryWriter(tensorboard_log_dir)

    lambda_penalty = 0.1
    env.set_lambda_penalty(lambda_penalty)
    val_env.set_lambda_penalty(lambda_penalty)
    # If success_k, allow setting K via env
    if constraint_wrapper == "success_k":
        try:
            k_consecutive = int(os.getenv("K_CONSECUTIVE", "3"))
        except Exception:
            k_consecutive = 3
        if hasattr(env, "set_k"):
            env.set_k(k_consecutive)
        if hasattr(val_env, "set_k"):
            val_env.set_k(k_consecutive)

    # Constraint curriculum: budget schedule from BUDGET_START -> BUDGET_FINAL
    try:
        budget_start = float(os.getenv("BUDGET_START", "200"))
    except Exception:
        budget_start = 200.0
    try:
        budget_final = float(os.getenv("BUDGET_FINAL", str(budget_start)))
    except Exception:
        budget_final = budget_start
    env.set_budget(budget_start)
    val_env.set_budget(budget_start)


    try:
        denom_factor = int(os.getenv("LAGRANGE_DENOM_FACTOR", "1"))
        denom_factor = max(1, denom_factor)
    except Exception:
        denom_factor = 1
    
    lagrange_iterations = max(1, total_timesteps // (2048 * num_envs * denom_factor))

    
    # fix lambda (disable updates) if was specified
    fix_lambda_env = os.getenv("FIX_LAMBDA")
    freeze_lambda = False
    if fix_lambda_env is not None:
        try:
            lambda_penalty = float(fix_lambda_env)
            freeze_lambda = True
        except Exception:
            freeze_lambda = False
    env.set_lambda_penalty(lambda_penalty)
    val_env.set_lambda_penalty(lambda_penalty)

    print(f"[START] Algo=LAGRANGIAN_PPO total_timesteps={total_timesteps} n_steps={model_params['n_steps']} num_envs={num_envs} denom_factor={denom_factor} lagrange_iterations={lagrange_iterations} lambda_fixed={freeze_lambda} lambda_init={lambda_penalty} wrapper={constraint_wrapper} budget_start={budget_start} budget_final={budget_final}", flush=True)



    

    for iteration in range(lagrange_iterations):  # outer Lagrange loop
        # linearly decay budget across outer iterations
        if lagrange_iterations > 1:
            frac = iteration / (lagrange_iterations - 1)
            current_budget = budget_start + (budget_final - budget_start) * frac
        else:
            current_budget = budget_final
        env.set_budget(current_budget)
        val_env.set_budget(current_budget)
        model.learn(
            total_timesteps=total_timesteps // lagrange_iterations, 
            reset_num_timesteps=False,
            log_interval=1, 
            progress_bar=True, 
            callback=[eval_callback, EntCoefScheduler(ent_schedule)]
        )

        # Evaluate average noise across validation env (sum of per-step noise, preserved for continuity)
        total_noise = 0
        num_episodes = len(benchmarks)
        for _ in range(num_episodes):
            obs = val_env.reset()
            done, ep_noise = False, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = val_env.step(action)
                ep_noise += info[0].get("noise", 0.0)
            total_noise += ep_noise
        avg_noise = total_noise / num_episodes

        # Lagrange update (unless frozen)
        if not freeze_lambda:
            if avg_noise > noise_threshold:
                lambda_penalty += 0.01 * (avg_noise - noise_threshold)
            else:
                lambda_penalty = max(0, lambda_penalty - 0.01)

        # Apply current lambda to envs
        env.set_lambda_penalty(lambda_penalty)
        val_env.set_lambda_penalty(lambda_penalty)

        # Logs
        print(f"[Iter {iteration}] budget={current_budget:.1f} Avg noise: {avg_noise:.2f}, λ: {lambda_penalty:.3f}")
        tensorboard_writer.add_scalar("Lagrange/lambda_penalty", lambda_penalty, iteration)
        tensorboard_writer.add_scalar("Lagrange/avg_noise", avg_noise, iteration)
        tensorboard_writer.add_scalar("Lagrange/budget", current_budget, iteration)
        model.save(f"{run_name}_iter_{iteration}_real")

    # Lagrangian PPO training ============== [End] ==============

    model.save(run_name)