import os
import random
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from .env import fheEnv
from .policy import HierarchicalMaskablePolicy
from .policy_film_b import HierarchicalMaskablePolicyFiLMB
from .utils import load_expressions, create_rules
from .logger import log_training_details
from .callbacks import linear_schedule, EntCoefScheduler, ParetoEvalCallback
from .pareto import generate_pref_list

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_agent(
    expressions_file: str,
    embeddings_model,
    total_timesteps: int = 2_000_000,
    num_envs: int = 8,
    seed: int = 42,
    budget_options: list = None,
    constraint_method: str = "lagrangian_pid",
    budget_encoding: str = "film",
    ent_coef: float = 0.01,
    algo: str = "ppo",
    lambda_env: float = 0.0,
    lambda_kl: float = 0.0,
    n_cycle: int = 1,
    n_budget: int = 5,
    policy_variant: str = "film_a"
):
    set_random_seed(seed)
    
    # MORL setup
    N = 11
    pref_list = generate_pref_list(N)
    
    # Common setup
    benchmarks = load_expressions("./fhe_rl/datasets/benchmarks.txt")
    expressions = load_expressions(expressions_file)
    max_positions = 16
    rules_list = create_rules("rules.txt", "rotations_rules.txt")
    rules_list["END"] = None
    
    job_id = os.environ.get("SLURM_JOB_ID", "jobid")
    run_name = f"model_{job_id}_{constraint_method}_{policy_variant}"
    tensorboard_log_dir = f"./tensorboard/{run_name}"
    checkpoint_dir = f"./checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Checkpoint resumption
    checkpoint_path = None
    steps_done = 0
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("rl_model_") and f.endswith("_steps.zip")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("_")[2]))
            checkpoint_path = os.path.join(checkpoint_dir, latest)
            steps_done = int(latest.split("_")[2])
            print(f"Found checkpoint: {checkpoint_path}")
            print(f"Resuming from step {steps_done}")

    # Environment creation
    def make_env(rank, exprs):
        def _init():
            # Seed worker-local Python, NumPy, and PyTorch RNGs
            np.random.seed(seed + rank)
            torch.manual_seed(seed + rank)
            env = fheEnv(
                rules_list, exprs, max_positions=max_positions,
                embeddings_model=embeddings_model,
                budget_options=budget_options,
                constraint_method=constraint_method,
                pref_list=pref_list,
                lambda_env=lambda_env, lambda_kl=lambda_kl,
                n_cycle=n_cycle, n_budget=n_budget, env_idx=rank,
                verbose=False
            )
            return Monitor(env)
        return _init  
    env = SubprocVecEnv([make_env(i, expressions) for i in range(num_envs)], start_method='spawn')
    pid_wrapper = None
    if constraint_method in ("lagrangian_pid", "lagrangian_pid_nato"):
        from .algos.lagrangian_pid import PIDLagrangianWrapper
        pid_wrapper = PIDLagrangianWrapper(env)
        env = pid_wrapper
        # Restore PID state if resuming from checkpoint
        if checkpoint_path:
            pid_state_file = checkpoint_path.replace(".zip", "_pid.json")
            if os.path.exists(pid_state_file):
                pid_wrapper.load_state(pid_state_file)
    val_env = DummyVecEnv([make_env(0, benchmarks)])
    val_env.seed(seed)
   

    # PPO model params
    ent_schedule = linear_schedule(ent_coef)
    ent_callback = EntCoefScheduler(ent_schedule, verbose=1)
    
    policy_cls = HierarchicalMaskablePolicyFiLMB if policy_variant == "film_b" else HierarchicalMaskablePolicy

    model_params = {
        "policy": policy_cls,
        "env": env,
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "n_epochs": 15,
        "clip_range": 0.1,
        "clip_range_vf": 0.2,
        "ent_coef": ent_coef,
        "verbose": 1,
        "tensorboard_log": tensorboard_log_dir,
        "seed": seed,
        "policy_kwargs": {
            "ent_coef": ent_coef,
            "budget_encoding": budget_encoding,
            "rule_dim": len(rules_list),
            "max_positions": max_positions,
            "rule_hidden_dims": [64, 32],
            "pos_hidden_dims": [32, 32],
            "value_hidden_dims": [128, 64, 32],
            "seed": seed,
        }
    }

    # Algorithm selection
    if algo == "focops":
        from .algos.focops import FOCOPS
        model = FOCOPS(**model_params, cost_limit=0.0, nu_lr=0.01, nu_max=10.0)
    else:
        if checkpoint_path:
            print(f"Loading model from checkpoint: {checkpoint_path}")
            model = PPO.load(checkpoint_path, env=env, tensorboard_log=tensorboard_log_dir)
        else:
            model = PPO(**model_params)

    log_training_details(
        model_params,
        job_id,
        num_data=len(expressions),
        num_actions=len(rules_list),
        total_timesteps=total_timesteps,
        output_model_name=run_name,
        notes=f"Algo: {algo} | Method: {constraint_method} | budget_encoding={budget_encoding} | num_envs={num_envs} | budgets={budget_options} | n_cycle={n_cycle} | n_budget={n_budget}",
    )

    # Callbacks
    class PIDCheckpointCallback(CheckpointCallback):
        """CheckpointCallback that also saves PID controller state if present."""
        def __init__(self, *args, pid_wrapper=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._pid_wrapper = pid_wrapper

        def _on_step(self) -> bool:
            res = super()._on_step()
            if self._pid_wrapper is not None and self.n_calls % self.save_freq == 0:
                pid_path = self._checkpoint_path(extension="json").replace("rl_model_", "rl_model_").replace(".json", "_pid.json")
                try:
                    self._pid_wrapper.save_state(pid_path)
                    if self.verbose >= 2:
                        print(f"Saved PID state to {pid_path}")
                except Exception as e:
                    print(f"Warning: could not save PID state to {pid_path}: {e}")
            return res

    checkpoint_callback = PIDCheckpointCallback(
        save_freq=5000,
        save_path=checkpoint_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
        pid_wrapper=pid_wrapper,
    )

    pareto_eval_cb = ParetoEvalCallback(
        val_env, 
        pref_list=pref_list,
        best_model_save_path=f"./eval/best_model_{run_name}", 
        log_path=tensorboard_log_dir, 
        eval_freq=50000,
        n_eval_episodes=len(benchmarks),
        deterministic=True, 
        verbose=1
    )

    # Training execution
    if checkpoint_path:
        remaining_steps = total_timesteps - steps_done
        print(f"Resuming training: {remaining_steps} steps remaining out of {total_timesteps} total")
    else:
        remaining_steps = total_timesteps
        print(f"Starting fresh training: {total_timesteps} total steps")

    model.learn(
        total_timesteps=remaining_steps, 
        log_interval=1, 
        progress_bar=True, 
        callback=[ent_callback, pareto_eval_cb, checkpoint_callback],
        reset_num_timesteps=(checkpoint_path is None)
    )
    
    model.save(run_name)
    print(f"\nModel saved as: {run_name}")