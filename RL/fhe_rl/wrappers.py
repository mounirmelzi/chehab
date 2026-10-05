from stable_baselines3.common.vec_env import VecEnvWrapper


class BaseLagrangianWrapper(VecEnvWrapper):
    """Base class for Lagrangian wrappers with shared lambda update logic.

    All Lagrangian variants share the same lambda (dual variable) update rule:
      - If noise > budget: increase lambda proportionally to excess
      - Otherwise: slowly decrease lambda toward 0
    The subclasses differ only in WHEN the penalty is applied (step_wait).
    """

    def __init__(self, venv, lambda_penalty=0.1):
        super().__init__(venv)
        self.lambda_penalty = lambda_penalty

    def update_lambda_penalty(self, noise, budget):
        if noise > budget:
            self.lambda_penalty += 0.01 * (noise - budget)
        else:
            self.lambda_penalty = max(0, self.lambda_penalty - 0.01)

    def reset(self):
        return self.venv.reset()


class LagrangianVecEnvWrapper(BaseLagrangianWrapper):
    """ON_DONE + ON_VIOLATION: penalty only at episode end AND only when budget is violated.

    This is the current stable method. The agent receives the Lagrangian penalty
    only at the terminal step if the final expression's noise exceeds the budget.
    """

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            delta = info["noise"] - info["budget"]
            ON_DONE, ON_VIOLATION = dones[env_index], delta > 0
            if ON_DONE and ON_VIOLATION:
                rewards[env_index] = rewards[env_index] - (self.lambda_penalty * delta)
        return obs, rewards, dones, infos


class LagrangianPerStepWrapper(BaseLagrangianWrapper):
    """Per-step violation penalty: penalize EVERY step when noise > budget.

    Provides dense feedback. The agent gets a penalty at every timestep where
    the current expression's estimated noise exceeds the budget. This gives
    stronger gradient signal but may compete with the cost-reduction reward.
    """

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            delta = info["noise"] - info["budget"]
            if delta > 0:
                rewards[env_index] = rewards[env_index] - (self.lambda_penalty * delta)
        return obs, rewards, dones, infos


class LagrangianAlwaysDoneWrapper(BaseLagrangianWrapper):
    """Always penalize at terminal: symmetric signal at episode end.

    At every episode termination, applies lambda * (noise - budget):
      - If noise > budget (violation):  penalty is subtracted (negative delta -> subtract positive)
      - If noise < budget (compliant):  bonus is added (positive delta -> add reward)
    This gives the agent a symmetric signal: reward for staying under budget,
    penalty for going over.
    """

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            if dones[env_index]:
                delta = info["noise"] - info["budget"]
                rewards[env_index] = rewards[env_index] - (self.lambda_penalty * delta)
        return obs, rewards, dones, infos
