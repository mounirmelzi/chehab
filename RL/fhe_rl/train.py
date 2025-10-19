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
from .wrappers import LagrangianVecEnvWrapper
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
    env = LagrangianVecEnvWrapper(env) # Use the Lagrangian wrapper

    val_env = DummyVecEnv([
        lambda: Monitor(fheEnv(rules_list, benchmarks, max_positions=max_positions,embeddings_model=embeddings_model))
    ])
    val_env = LagrangianVecEnvWrapper(val_env) # Use the Lagrangian wrapper

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

    noise_threshold = 100.0


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

    print(f"[START] Algo=LAGRANGIAN_PPO total_timesteps={total_timesteps} n_steps={model_params['n_steps']} num_envs={num_envs} denom_factor={denom_factor} lagrange_iterations={lagrange_iterations} lambda_fixed={freeze_lambda} lambda_init={lambda_penalty}", flush=True)



    

    for iteration in range(lagrange_iterations):  # outer Lagrange loop
        model.learn(
            total_timesteps=total_timesteps // lagrange_iterations, 
            reset_num_timesteps=False,
            log_interval=1, 
            progress_bar=True, 
            callback=[eval_callback, EntCoefScheduler(ent_schedule)]
        )

        # Evaluate average noise across validation env
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
        print(f"[Iter {iteration}] Avg noise: {avg_noise:.2f}, λ: {lambda_penalty:.3f}")
        tensorboard_writer.add_scalar("Lagrange/lambda_penalty", lambda_penalty, iteration)
        tensorboard_writer.add_scalar("Lagrange/avg_noise", avg_noise, iteration)
        model.save(f"{run_name}_iter_{iteration}_real")

    # Lagrangian PPO training ============== [End] ==============

    model.save(run_name)

