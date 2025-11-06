from stable_baselines3.common.vec_env import VecEnvWrapper


class LagrangianVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, lambda_penalty=0.1, noise_threshold=300.0):
        super().__init__(venv)
        self.set_lambda_penalty(lambda_penalty)
        self.set_noise_threshold(noise_threshold)

    def set_lambda_penalty(self, lambda_penalty):
        self.lambda_penalty = lambda_penalty

    def set_noise_threshold(self, noise_threshold):
        self.noise_threshold = noise_threshold

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            noise = info.get("noise", 0.0)
            delta = noise - self.noise_threshold
            ON_DONE, ON_VIOLATION = dones[env_index], delta > 0
            if ON_DONE and ON_VIOLATION:
                rewards[env_index] = rewards[env_index] - (self.lambda_penalty * delta)
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()


class LagrangianPerStepViolationWrapper(VecEnvWrapper):
    """Ablation 1: Penalize on ALL actions when noise > threshold (remove ON_DONE check).
    
    - Penalizes on every step when noise > threshold (not just at episode end)
    - Still only penalizes when there's a violation (noise > threshold)
    - Expected: More frequent penalties, agent learns to avoid violations earlier
    """
    def __init__(self, venv, lambda_penalty=0.1, noise_threshold=300.0):
        super().__init__(venv)
        self.set_lambda_penalty(lambda_penalty)
        self.set_noise_threshold(noise_threshold)

    def set_lambda_penalty(self, lambda_penalty):
        self.lambda_penalty = lambda_penalty

    def set_noise_threshold(self, noise_threshold):
        self.noise_threshold = noise_threshold

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            noise = info.get("noise", 0.0)
            delta = noise - self.noise_threshold
            ON_VIOLATION = delta > 0
            # Penalize on ALL steps when violation occurs (removed ON_DONE check)
            if ON_VIOLATION:
                rewards[env_index] = rewards[env_index] - (self.lambda_penalty * delta)
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()


class LagrangianAlwaysOnDoneWrapper(VecEnvWrapper):
    """Ablation 2: Always penalize at episode end (remove ON_VIOLATION check).
    
    - Penalizes only at episode end (keeps ON_DONE check)
    - Always applies penalty, not just when noise > threshold (removed ON_VIOLATION check)
    - Penalty = lambda * (noise - threshold)
    - If noise < threshold: negative penalty (reward bonus)
    - If noise > threshold: positive penalty (punishment)
    - Expected: Agent gets feedback even when within budget, learns to stay further below threshold
    """
    def __init__(self, venv, lambda_penalty=0.1, noise_threshold=300.0):
        super().__init__(venv)
        self.set_lambda_penalty(lambda_penalty)
        self.set_noise_threshold(noise_threshold)

    def set_lambda_penalty(self, lambda_penalty):
        self.lambda_penalty = lambda_penalty

    def set_noise_threshold(self, noise_threshold):
        self.noise_threshold = noise_threshold

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            noise = info.get("noise", 0.0)
            delta = noise - self.noise_threshold
            ON_DONE = dones[env_index]
            # Always penalize at episode end (removed ON_VIOLATION check)
            # delta can be negative (bonus) or positive (penalty)
            if ON_DONE:
                rewards[env_index] = rewards[env_index] - (self.lambda_penalty * delta)
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()


class LagrangianAlwaysPerStepWrapper(VecEnvWrapper):
    """Ablation 3: Always penalize on ALL steps (remove both ON_DONE and ON_VIOLATION checks).
    
    - Penalizes on every step (removed ON_DONE check)
    - Always applies penalty, not just when noise > threshold (removed ON_VIOLATION check)
    - Penalty = lambda * (noise - threshold) on every step
    - If noise < threshold: negative penalty (reward bonus) on every step
    - If noise > threshold: positive penalty (punishment) on every step
    - Expected: Constant feedback to agent about distance from threshold, learns to stay near threshold
    """
    def __init__(self, venv, lambda_penalty=0.1, noise_threshold=300.0):
        super().__init__(venv)
        self.set_lambda_penalty(lambda_penalty)
        self.set_noise_threshold(noise_threshold)

    def set_lambda_penalty(self, lambda_penalty):
        self.lambda_penalty = lambda_penalty

    def set_noise_threshold(self, noise_threshold):
        self.noise_threshold = noise_threshold

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            noise = info.get("noise", 0.0)
            delta = noise - self.noise_threshold
            # Always penalize on ALL steps (removed both ON_DONE and ON_VIOLATION checks)
            # delta can be negative (bonus) or positive (penalty)
            rewards[env_index] = rewards[env_index] - (self.lambda_penalty * delta)
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()
