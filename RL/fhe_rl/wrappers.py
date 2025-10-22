from stable_baselines3.common.vec_env import VecEnvWrapper


class LagrangianVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, lambda_penalty=0.1):
        super().__init__(venv)
        self.set_lambda_penalty(lambda_penalty)
        # Tracking cumulative noise per env index for excess-only penalty
        self._budget = 200.0
        self._margin = 10.0
        self._cum_noise = None  # lazily sized on first reset

    def set_lambda_penalty(self, lambda_penalty):
        self.lambda_penalty = lambda_penalty

    def reset(self, *args, **kwargs):
        # delegate; SB3 VecEnv.reset returns obs (keep pass-through)
        obs = self.venv.reset(*args, **kwargs)
        # initialize/reset cumulative noise tracker
        try:
            num_envs = len(self.venv.remotes) if hasattr(self.venv, "remotes") else self.num_envs
        except Exception:
            num_envs = self.num_envs
        if self._cum_noise is None or len(self._cum_noise) != num_envs:
            self._cum_noise = [0.0 for _ in range(num_envs)]
        else:
            for i in range(len(self._cum_noise)):
                self._cum_noise[i] = 0.0
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            noise = info.get("noise", 0.0)
            # accumulate noise and penalize only excess beyond (budget - margin)
            if self._cum_noise is None:
                self._cum_noise = [0.0 for _ in range(len(rewards))]
            self._cum_noise[env_index] += float(noise)
            excess = max(0.0, self._cum_noise[env_index] - (self._budget - self._margin))
            # normalize by budget to keep scales comparable
            penalties = self.lambda_penalty * (excess / self._budget)
            rewards[env_index] = rewards[env_index] - penalties
            info["lambda_penalty"] = self.lambda_penalty
            info["cum_noise"] = self._cum_noise[env_index]
            info["excess_noise"] = excess
        # reset cum_noise when episode ends
        for i, done in enumerate(dones):
            if bool(done):
                self._cum_noise[i] = 0.0
        return obs, rewards, dones, infos
