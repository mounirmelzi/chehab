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


class SuccessBonusKPenaltyWrapper(VecEnvWrapper):
    """Terminal success bonus; per-step penalty only after k consecutive above-budget noisy expressions.

    - success bonus: add a small positive bonus at episode end if final noise <= budget
    - k-consecutive penalty: if for k consecutive steps the per-step expression noise > budget,
      apply a small per-step penalty proportional to (noise - budget)/budget.
    """

    def __init__(self, venv, lambda_penalty=0.1, k_consecutive: int = 3):
        super().__init__(venv)
        self._budget = 200.0
        self.lambda_penalty = lambda_penalty
        self._k = max(1, int(k_consecutive))
        self._consec_above = None
        self._success_bonus = 1.0

    def set_lambda_penalty(self, lambda_penalty):
        self.lambda_penalty = lambda_penalty

    def set_budget(self, budget):
        try:
            self._budget = float(budget)
        except Exception:
            pass

    def set_k(self, k_consecutive: int):
        try:
            self._k = max(1, int(k_consecutive))
        except Exception:
            pass

    def reset(self, *args, **kwargs):
        obs = self.venv.reset(*args, **kwargs)
        try:
            num_envs = len(self.venv.remotes) if hasattr(self.venv, "remotes") else self.num_envs
        except Exception:
            num_envs = self.num_envs
        if self._consec_above is None or len(self._consec_above) != num_envs:
            self._consec_above = [0 for _ in range(num_envs)]
        else:
            for i in range(len(self._consec_above)):
                self._consec_above[i] = 0
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # annotate
        for info in infos:
            info["lambda_penalty"] = self.lambda_penalty
            info["budget"] = self._budget

        if self._consec_above is None:
            self._consec_above = [0 for _ in range(len(rewards))]

        # k-consecutive above-budget per-step penalty
        for i, info in enumerate(infos):
            noise = float(info.get("noise", 0.0))
            if noise > self._budget:
                self._consec_above[i] += 1
            else:
                self._consec_above[i] = 0
            if self._consec_above[i] >= self._k:
                over = max(0.0, noise - self._budget)
                step_pen = self.lambda_penalty * (over / self._budget)
                rewards[i] -= step_pen

        # terminal success bonus and cleanup
        for i, done in enumerate(dones):
            if bool(done):
                final_noise = float(infos[i].get("noise", 0.0))
                if final_noise <= self._budget:
                    rewards[i] += self._success_bonus
                self._consec_above[i] = 0
        return obs, rewards, dones, infos