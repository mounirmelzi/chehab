import os
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


def train_agent(expressions_file: str, embeddings_model, total_timesteps: int = 2_000_000, num_envs: int = 8):
    match get_rl_algorithm():
        case RLAlgorithm.PPO:
            train_ppo_agent(expressions_file, embeddings_model, total_timesteps, num_envs)
        case RLAlgorithm.LAGRANGIAN_PPO:
            train_lagrangian_ppo_agent(expressions_file, embeddings_model, total_timesteps, num_envs)
        case _:
            train_ppo_agent(expressions_file, embeddings_model, total_timesteps, num_envs)


def train_ppo_agent(expressions_file: str, embeddings_model, total_timesteps: int, num_envs: int = 8):
    benchmarks = load_expressions("./fhe_rl/datasets/benchmarks.txt") 
    expressions = load_expressions(expressions_file, benchmarks)
    max_positions = 16
    rules_list  = create_rules("rules.txt")
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


def train_lagrangian_ppo_agent(expressions_file: str, embeddings_model, total_timesteps: int, num_envs: int = 8):
    benchmarks = load_expressions("./fhe_rl/datasets/benchmarks.txt") 
    expressions = load_expressions(expressions_file, benchmarks)
    max_positions = 16
    rules_list  = create_rules("rules.txt")
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


    lagrange_delay = 0
    denom_factor = 4
    n_steps = 2048


    model_params = {
        "policy": HierarchicalMaskablePolicy,
        "env": env,
        "seed": 42,
        "learning_rate": 1e-4,
        "n_steps": n_steps,
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
        notes=f"Lagrangian PPO [ON_DONE+ON_VIOLATION] Delayed({lagrange_delay}): denom_factor={denom_factor}"
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

    lagrange_iterations = total_timesteps // ((n_steps * num_envs) * denom_factor)
    lagrange_iterations = max(1, lagrange_iterations)

    for lagrange_iteration in range(lagrange_iterations):  # outer Lagrange loop
        model.learn(
            total_timesteps=total_timesteps // lagrange_iterations,
            reset_num_timesteps=False,
            log_interval=1,
            progress_bar=True,
            callback=[eval_callback, EntCoefScheduler(ent_schedule)]
        )

        # Evaluate average noise across validation env
        total_noise = 0
        for _ in range(num_benchmarks):
            obs = val_env.reset()
            done, ep_noise = False, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = val_env.step(action)
                ep_noise += info[0].get("noise", 0.0)
            total_noise += ep_noise

            if lagrange_iteration >= lagrange_delay: # Delay lambda penalty updates
                val_env.update_lambda_penalty(noise=ep_noise, budget=val_env.unwrapped.reset_infos[0]["budget"])
                env.update_lambda_penalty(noise=ep_noise, budget=val_env.unwrapped.reset_infos[0]["budget"])

        lambda_penalty = val_env.lambda_penalty
        avg_noise = total_noise / num_benchmarks

        # Logs
        print(f"[Step {model.num_timesteps}] Avg noise: {avg_noise:.2f}, λ: {lambda_penalty:.3f}")
        tensorboard_writer.add_scalar("Lagrange/lambda_penalty", lambda_penalty, model.num_timesteps)
        tensorboard_writer.add_scalar("Lagrange/avg_noise", avg_noise, model.num_timesteps)

    tensorboard_writer.close()

    # Lagrangian PPO training ============== [End] ==============
