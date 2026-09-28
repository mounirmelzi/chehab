"""PID-controlled Lagrangian wrapper for constrained PPO.

Replaces the crude +/-0.01 lambda update in the existing
BaseLagrangianWrapper with a PID controller for smoother convergence.

Usage in train.py:
    from .algos.lagrangian_pid import PIDLagrangianWrapper
    env = PIDLagrangianWrapper(env)
    model = PPO(...)
    model.learn(...)
"""

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class PIDLagrangianWrapper(VecEnvWrapper):
    """VecEnv wrapper that adds a Lagrangian constraint penalty to rewards.

    The penalty lambda is updated via a PID controller on the normalised
    constraint violation  error = (noise - budget) / budget.

    At episode termination, rewards are modified:
        reward -= lambda * max(0, noise - budget)

    PID gains (Kp, Ki, Kd) are tuned for the FHE noise problem where
    typical violations are O(10-100) bits and budgets are O(200-400).
    """

    def __init__(
        self,
        venv,
        Kp: float = 0.005,
        Ki: float = 0.001,
        Kd: float = 0.001,
        lambda_init: float = 0.1,
        lambda_max: float = 20.0,
    ):
        super().__init__(venv)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.lambda_penalty = lambda_init
        self.lambda_max = lambda_max
        self._integral = 0.0
        self._prev_error = 0.0

    def update_lambda_penalty(self, noise, budget):
        """PID update for the dual variable."""
        error = (noise - budget) / max(budget, 1)
        self._integral = float(np.clip(self._integral + error, -10.0, 10.0))
        derivative = error - self._prev_error
        delta = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        self.lambda_penalty = float(np.clip(
            self.lambda_penalty + delta, 0.0, self.lambda_max,
        ))
        self._prev_error = error

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_idx, info in enumerate(infos):
            if not dones[env_idx]:
                continue
            noise = info.get("noise", 0.0)
            budget = info.get("budget", 1)
            delta = (noise - budget) / max(budget, 1)  # normalized, matches PID controller units

            if delta > 0:
                rewards[env_idx] -= self.lambda_penalty * delta
            self.update_lambda_penalty(noise, budget)
        return obs, rewards, dones, infos

    def save_state(self, filepath: str):
        """Save PID controller state alongside a model checkpoint."""
        import json
        state = {
            "lambda_penalty": float(self.lambda_penalty),
            "_integral": float(self._integral),
            "_prev_error": float(self._prev_error),
            "Kp": float(self.Kp),
            "Ki": float(self.Ki),
            "Kd": float(self.Kd),
            "lambda_max": float(self.lambda_max),
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load PID controller state from file."""
        import json, os
        if not os.path.exists(filepath):
            print(f"[PIDLagrangianWrapper] No PID state file found at {filepath}, keeping defaults.")
            return
        with open(filepath, "r") as f:
            state = json.load(f)
        self.lambda_penalty = float(state.get("lambda_penalty", self.lambda_penalty))
        self._integral = float(state.get("_integral", self._integral))
        self._prev_error = float(state.get("_prev_error", self._prev_error))
        print(f"[PIDLagrangianWrapper] Loaded PID state from {filepath}: "
              f"lambda={self.lambda_penalty:.4f}, integral={self._integral:.4f}, prev_error={self._prev_error:.4f}")
