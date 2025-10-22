from stable_baselines3.common.vec_env import VecEnvWrapper


class LagrangianVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, lambda_penalty=0.1):
        super().__init__(venv)
        self.set_lambda_penalty(lambda_penalty)
        # Tracking cumulative noise for living bonus gating
        self._budget = 200.0
        self._margin = 10.0
        self._cum_noise = None
        self._living_bonus_base = 0.5  # tuned small so it won't overpower penalties

    def set_lambda_penalty(self, lambda_penalty):
        self.lambda_penalty = lambda_penalty

    def reset(self, *args, **kwargs):
        # delegate; SB3 VecEnv.reset returns obs (keep pass-through)
        obs = self.venv.reset(*args, **kwargs)
        # init/reset cumulative noise tracker
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
            noise = float(info.get("noise", 0.0))
            # accumulate noise for gating
            if self._cum_noise is None:
                self._cum_noise = [0.0 for _ in range(len(rewards))]
            self._cum_noise[env_index] += noise
            # per-step penalty
            rewards[env_index] -= (self.lambda_penalty * noise)
            # living bonus while safely under budget - margin
            headroom = max(0.0, (self._budget - self._margin) - self._cum_noise[env_index])
            if headroom > 0.0:
                scale = headroom / (self._budget - self._margin)
                rewards[env_index] += self._living_bonus_base * scale
            info["lambda_penalty"] = self.lambda_penalty
            info["cum_noise"] = self._cum_noise[env_index]
        # reset cum_noise per finished env
        for i, done in enumerate(dones):
            if bool(done):
                self._cum_noise[i] = 0.0
        return obs, rewards, dones, infos
