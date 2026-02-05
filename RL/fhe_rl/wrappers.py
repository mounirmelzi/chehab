from stable_baselines3.common.vec_env import VecEnvWrapper


class LagrangianVecEnvWrapper(VecEnvWrapper):
    def __init__(self, venv, lambda_penalty=0.1):
        super().__init__(venv)
        self.lambda_penalty = lambda_penalty

    def update_lambda_penalty(self, noise, budget):
        if noise > budget:
            self.lambda_penalty += 0.01 * (noise - budget)
        else:
            self.lambda_penalty = max(0, self.lambda_penalty - 0.01)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            delta = info["noise"] - info["budget"]
            ON_DONE, ON_VIOLATION = dones[env_index], delta > 0
            if ON_DONE and ON_VIOLATION:
                rewards[env_index] = rewards[env_index] - (self.lambda_penalty * delta)
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()
