from stable_baselines3.common.vec_env import VecEnvWrapper


class LagrangianVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.lambdas = {}

    def _get_lambda(self, budget):
        if budget not in self.lambdas:
            self.lambdas[budget] = 0.1
        return self.lambdas[budget]

    def _set_lambda(self, budget, lambda_penalty):
        self.lambdas[budget] = lambda_penalty

    def update_lambda_penalty(self, noise, budget):
        lambda_penalty = self._get_lambda(budget)
        if noise > budget:
            lambda_penalty += 0.01 * (noise - budget)
        else:
            lambda_penalty = max(0, lambda_penalty - 0.01)
        self._set_lambda(budget, lambda_penalty)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            noise, budget = info.get("noise"), info.get("budget")
            delta = noise - budget
            ON_DONE, ON_VIOLATION = dones[env_index], delta > 0
            if ON_DONE and ON_VIOLATION:
                rewards[env_index] = rewards[env_index] - (self._get_lambda(budget) * delta)
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()
