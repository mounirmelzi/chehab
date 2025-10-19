from stable_baselines3.common.vec_env import VecEnvWrapper


class LagrangianVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, lambda_penalty=0.1):
        super().__init__(venv)
        self.set_lambda_penalty(lambda_penalty)

    def set_lambda_penalty(self, lambda_penalty):
        self.lambda_penalty = lambda_penalty

    def reset(self, *args, **kwargs):
        # delegate; SB3 VecEnv.reset returns obs (keep pass-through)
        return self.venv.reset(*args, **kwargs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            noise = info.get("noise", 0.0)
            rewards[env_index] = rewards[env_index] - (self.lambda_penalty * noise)
            info["lambda_penalty"] = self.lambda_penalty
        return obs, rewards, dones, infos
