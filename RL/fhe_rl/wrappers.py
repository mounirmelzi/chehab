from stable_baselines3.common.vec_env import VecEnvWrapper


class LagrangianVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, lambda_penalty=0.1):
        super().__init__(venv)
        self.set_lambda_penalty(lambda_penalty)
        self._budget = 200.0

    def set_lambda_penalty(self, lambda_penalty):
        self.lambda_penalty = lambda_penalty
    
    def set_budget(self, budget):
        try:
            self._budget = float(budget)
        except Exception:
            pass

    def reset(self, *args, **kwargs):
        return self.venv.reset(*args, **kwargs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # expose lambda and budget for logging
        for info in infos:
            info["lambda_penalty"] = self.lambda_penalty
            info["budget"] = self._budget

        # apply terminal-only penalty using the final expression's noise
        for i, done in enumerate(dones):
            if bool(done):
                final_noise = float(infos[i].get("noise", 0.0))
                excess = max(0.0, final_noise - self._budget)
                penalties = self.lambda_penalty * (excess / self._budget)
                rewards[i] -= penalties
        return obs, rewards, dones, infos