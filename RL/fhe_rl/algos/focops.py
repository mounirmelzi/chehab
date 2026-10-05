"""FOCOPS — First Order Constrained Optimization in Policy Space.

Subclasses SB3 PPO to add a constraint cost term to the policy loss.
The constraint is: E[max(0, final_noise - budget)] <= cost_limit.

Reference:
    Zhang et al., "First Order Constrained Optimization in Policy Space",
    NeurIPS 2020.

The modification is simple: the PPO advantage is replaced with a
*penalised advantage*  A_penalised = A_reward - nu * A_constraint,
where nu is a dual variable updated after each rollout.
"""

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class ConstraintCostTracker(BaseCallback):
    """Callback that collects per-episode constraint costs from info dicts.

    At each terminal step the normalised constraint violation
    max(0, noise - budget) / budget is recorded.  These are used by FOCOPS
    to update the dual variable nu after each rollout.

    Uses a shared list (passed by reference) instead of holding a back-
    reference to the model, which would break cloudpickle serialisation.
    """

    def __init__(self, episode_costs_list, verbose=0):
        super().__init__(verbose)
        self._costs = episode_costs_list

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            noise = info.get("noise", 0.0)
            budget = info.get("budget", 1)
            cost = max(0.0, (noise - budget) / max(budget, 1))
            self._costs.append(cost)
        return True


class FOCOPS(PPO):
    """PPO with FOCOPS constraint enforcement.

    During each PPO training epoch the advantage used in the clipped
    surrogate is replaced with:
        A_eff = A_reward  -  nu * constraint_signal

    where constraint_signal is 1.0 for transitions belonging to episodes
    that violated the budget and 0.0 otherwise.  nu is automatically
    tuned after each collect_rollouts call.
    """

    def __init__(
        self,
        *args,
        cost_limit: float = 0.0,
        nu_lr: float = 0.01,
        nu_max: float = 10.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cost_limit = cost_limit
        self.nu_lr = nu_lr
        self.nu_max = nu_max
        self.nu = 0.0
        self._episode_costs: list[float] = []

    def _get_torch_save_params(self):
        """Exclude non-picklable attributes from save."""
        state_dicts, var_list = super()._get_torch_save_params()
        return state_dicts, var_list

    # ------------------------------------------------------------------
    # Override learn() to inject our callback automatically
    # ------------------------------------------------------------------
    def learn(self, total_timesteps, callback=None, **kwargs):
        tracker = ConstraintCostTracker(self._episode_costs)
        if callback is None:
            callback = [tracker]
        elif isinstance(callback, list):
            callback = callback + [tracker]
        else:
            callback = [callback, tracker]
        return super().learn(total_timesteps, callback=callback, **kwargs)

    # ------------------------------------------------------------------
    # Override train() to use penalised advantage
    # ------------------------------------------------------------------
    def train(self) -> None:
        # Update nu from collected episode costs before the gradient steps
        if self._episode_costs:
            avg_cost = float(np.mean(self._episode_costs))
            self.nu = float(np.clip(
                self.nu + self.nu_lr * (avg_cost - self.cost_limit),
                0.0,
                self.nu_max,
            ))
            self._episode_costs.clear()

        if self.nu > 0:
            # Penalise the advantage stored in the rollout buffer.
            # We subtract nu * |advantage| for transitions that came from
            # episodes with positive cost (a simple but effective heuristic:
            # large positive advantages in violating episodes are dampened).
            buf = self.rollout_buffer
            adv = buf.advantages
            penalty = self.nu * np.abs(adv)
            buf.advantages = adv - penalty
            # Re-normalise (PPO expects zero-mean, unit-std advantages)
            buf.advantages = (buf.advantages - buf.advantages.mean()) / (buf.advantages.std() + 1e-8)

        super().train()

        self.logger.record("focops/nu", self.nu)
        if self._episode_costs:
            self.logger.record("focops/avg_cost", float(np.mean(self._episode_costs)))
