import os
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from .utils  import load_expressions, create_rules
from .logger import log_training_details
from .callbacks import CurriculumCallback
from .config import get_env_class, get_policy_class, get_wrapper_class
from stable_baselines3.common.callbacks import EvalCallback
from .wrappers import (
    LagrangianVecEnvWrapper,
    LagrangianPerStepWrapper,
    LagrangianAlwaysDoneWrapper,
)
from torch.utils.tensorboard import SummaryWriter


# Wrapper class for each Lagrangian variant (used when wrapper_class = "auto";
# constraint methods not listed here train without a wrapper)
WRAPPER_MAP = {
    "lagrangian_od_ov":        LagrangianVecEnvWrapper,
    "lagrangian_perstep":      LagrangianPerStepWrapper,
    "lagrangian_always_done":  LagrangianAlwaysDoneWrapper,
}


def train_agent(
    expressions_file: str,
    embeddings_model,
    total_timesteps: int = 2_000_000,
    num_envs: int = 8,
    budget_options: list = None,
    denom_factor: int = 4,
    constraint_method: str = "lagrangian_od_ov",
    budget_encoding: str = "raw",
    ent_coef: float = 0.01,
    curriculum: bool = False,
    algo: str = "ppo",
):
    """Unified training entry point for all constraint methods.

    Parameters
    ----------
    constraint_method : str
        One of: none, lagrangian_od_ov, lagrangian_perstep,
        lagrangian_always_done, margin_barrier, noise_masking
    budget_encoding : str
        How to encode budget in the policy network:
        raw = concat one-hot, embed = learned embedding, film = FiLM conditioning
    ent_coef : float
        Entropy coefficient for exploration (constant throughout training).
    curriculum : bool
        If True, gradually widen the set of active budgets during training.
    """

    # ── Common setup ─────────────────────────────────────────────────────────
    benchmarks = load_expressions("./fhe_rl/datasets/benchmarks.txt")
    expressions = load_expressions(expressions_file, benchmarks)
    max_positions = 16
    rules_list = create_rules("rules.txt")
    rules_list["END"] = None
    job_id = os.environ.get("SLURM_JOB_ID", "jobid")
    run_name = f"model_{job_id}_{constraint_method}"
    tensorboard_log_dir = f"./tensorboard/{run_name}"

    EnvCls = get_env_class()
    PolicyCls = get_policy_class()

    # ── Environment creation ─────────────────────────────────────────────────
    def make_env():
        return Monitor(
            EnvCls(
                rules_list, expressions,
                max_positions=max_positions,
                embeddings_model=embeddings_model,
                budget_options=budget_options,
                constraint_method=constraint_method,
            )
        )

    env = SubprocVecEnv([make_env for _ in range(num_envs)], start_method='spawn')
    val_env = DummyVecEnv([
        lambda: Monitor(
            EnvCls(
                rules_list, benchmarks,
                max_positions=max_positions,
                embeddings_model=embeddings_model,
                budget_options=budget_options,
                constraint_method=constraint_method,
            )
        )
    ])

    # ── Wrap env if a wrapper is configured ────────────────────────────────────
    # "auto" picks the Lagrangian wrapper matching the constraint method;
    # None disables wrapping; a custom class from COMPONENT_CONFIG is used as-is.
    wrapper_setting = get_wrapper_class()
    if wrapper_setting == "auto":
        WrapperCls = WRAPPER_MAP.get(constraint_method)
    else:
        WrapperCls = wrapper_setting
    if WrapperCls is not None:
        env = WrapperCls(env)
        val_env = WrapperCls(val_env)

    # The Lagrangian outer loop (train -> eval noise -> update lambda) runs only
    # when the wrapper exposes lambda updates. Custom wrappers without
    # update_lambda_penalty get the plain model.learn() loop.
    use_lagrange = hasattr(env, "update_lambda_penalty")

    # ── PPO model ────────────────────────────────────────────────────────────
    n_steps = 2048

    print("=" * 80)
    print(f"Training config:")
    print(f"  algo            = {algo}")
    print(f"  method          = {constraint_method}")
    print(f"  budget_encoding = {budget_encoding}")
    print(f"  budget_options  = {budget_options}")
    print(f"  ent_coef        = {ent_coef}")
    print(f"  curriculum      = {curriculum}")
    print(f"  total_timesteps = {total_timesteps:,}")
    print(f"  num_envs        = {num_envs}")
    print(f"  denom_factor    = {denom_factor}")
    print(f"  use_lagrange    = {use_lagrange}")
    print(f"  run_name        = {run_name}")
    print("=" * 80)

    model_params = {
        "policy": PolicyCls,
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
        "ent_coef": ent_coef,
        "verbose": 1,
        "tensorboard_log": tensorboard_log_dir,
        "policy_kwargs": {
            "ent_coef": ent_coef,
            "budget_encoding": budget_encoding,
            "rule_dim":      len(rules_list),
            "max_positions": max_positions,
            "rule_hidden_dims":   [128, 64],
            "pos_hidden_dims":    [64, 64],
            "value_hidden_dims":  [256, 128, 64],
        }
    }
    # ── Algorithm selection ──────────────────────────────────────────────
    if algo == "focops":
        from .algos.focops import FOCOPS
        model = FOCOPS(**model_params, cost_limit=0.0, nu_lr=0.01, nu_max=10.0)
    elif algo == "lagrangian_pid":
        from .algos.lagrangian_pid import PIDLagrangianWrapper
        env = PIDLagrangianWrapper(env)
        model_params["env"] = env
        model = PPO(**model_params)
    else:
        model = PPO(**model_params)

    log_training_details(
        model_params,
        job_id,
        num_data=len(expressions),
        num_actions=len(rules_list),
        total_timesteps=total_timesteps,
        output_model_name=run_name,
        notes=f"Algo: {algo} | Method: {constraint_method} | budget_encoding={budget_encoding} | denom_factor={denom_factor} | num_envs={num_envs} | budgets={budget_options}",
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
        verbose=1,
    )

    # ── Build callback list ──────────────────────────────────────────────────
    callbacks = [eval_callback]
    if curriculum and budget_options and len(budget_options) >= 3:
        sorted_budgets = sorted(budget_options)
        n = len(sorted_budgets)
        phase_0 = sorted_budgets[:2]
        phase_1 = sorted_budgets[:max(3, n - 1)]
        phase_2 = sorted_budgets
        callbacks.append(CurriculumCallback(
            budget_phases=[phase_0, phase_1, phase_2],
            phase_boundaries=[0.33, 0.66],
        ))
        print(f"Curriculum phases: {phase_0} → {phase_1} → {phase_2}")

    # ── Training loop ────────────────────────────────────────────────────────
    if use_lagrange:
        _train_lagrangian_loop(
            model, env, val_env,
            total_timesteps=total_timesteps,
            n_steps=n_steps,
            num_envs=num_envs,
            denom_factor=denom_factor,
            num_benchmarks=num_benchmarks,
            eval_callback=eval_callback,
            callbacks=callbacks,
            tensorboard_log_dir=tensorboard_log_dir,
            run_name=run_name,
        )
    else:
        # Simple training (none, margin_barrier, noise_masking): single model.learn()
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=1,
            progress_bar=True,
            callback=callbacks,
        )

    # Always save the final model
    model.save(run_name)
    print(f"\nModel saved as: {run_name}")


def _train_lagrangian_loop(
    model, env, val_env, *,
    total_timesteps, n_steps, num_envs, denom_factor,
    num_benchmarks, eval_callback, callbacks,
    tensorboard_log_dir, run_name,
):
    """Lagrangian PPO outer loop: train, evaluate, update lambda."""

    tensorboard_writer = SummaryWriter(tensorboard_log_dir)

    lagrange_iterations = total_timesteps // ((n_steps * num_envs) * denom_factor)
    lagrange_iterations = max(1, lagrange_iterations)
    timesteps_per_iter = total_timesteps // lagrange_iterations

    print(f"Lagrangian loop: {lagrange_iterations} iterations x {timesteps_per_iter:,} timesteps each")

    for lagrange_iteration in range(lagrange_iterations):
        model.learn(
            total_timesteps=timesteps_per_iter,
            reset_num_timesteps=False,
            log_interval=1,
            progress_bar=True,
            callback=callbacks,
        )

        # ── Evaluate average noise on validation benchmarks ──────────────
        total_noise = 0
        for _ in range(num_benchmarks):
            obs = val_env.reset()
            done, ep_noise = False, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = val_env.step(action)
                ep_noise = info[0].get("noise", 0.0)
            total_noise += ep_noise

            # Update lambda on both train and val wrappers
            val_env.update_lambda_penalty(
                noise=ep_noise,
                budget=val_env.unwrapped.reset_infos[0]["budget"],
            )
            env.update_lambda_penalty(
                noise=ep_noise,
                budget=val_env.unwrapped.reset_infos[0]["budget"],
            )

        lambda_penalty = val_env.lambda_penalty
        avg_noise = total_noise / num_benchmarks

        print(f"[Step {model.num_timesteps}] Avg noise: {avg_noise:.2f}, "
              f"\u03bb: {lambda_penalty:.3f}")
        tensorboard_writer.add_scalar(
            "Lagrange/lambda_penalty", lambda_penalty, model.num_timesteps,
        )
        tensorboard_writer.add_scalar(
            "Lagrange/avg_noise", avg_noise, model.num_timesteps,
        )

    tensorboard_writer.close()
