from stable_baselines3.common.vec_env import VecEnvWrapper


class LagrangianVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, lambda_penalty=0.1, noise_threshold=200.0):
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
