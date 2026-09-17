from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import torch
import os


class EntCoefScheduler(BaseCallback):
    def __init__(self, schedule, verbose: int = 0):
        super().__init__(verbose)
        self.schedule = schedule

    def _on_training_start(self) -> None:
        p = self.model._current_progress_remaining
        self.model.ent_coef = float(self.schedule(p))

    def _on_rollout_end(self) -> None:
        p = self.model._current_progress_remaining
        self.model.ent_coef = float(self.schedule(p))

    def _on_step(self) -> bool:
        return True

class ParetoEvalCallback(BaseCallback):
    def __init__(self, eval_env, pref_list, 
                 best_model_save_path=None, 
                 log_path=None, 
                 eval_freq=512, 
                 n_eval_episodes=5, 
                 deterministic=True, 
                 verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.pref_list = pref_list
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        
        # Track best performance (Average across the Pareto Front)
        self.best_mean_reward = -np.inf

        # Ensure directories exist
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\nStep {self.num_timesteps}: Starting Pareto Evaluation ({len(self.pref_list)} points)")
            
            all_means = []
            all_lengths = []
            
            for w in self.pref_list:
                # Force eval env to this specific goal
                self.eval_env.env_method("set_preference_vector", w)
                
                # Run evaluation
                episode_rewards, episode_lengths = evaluate_policy(
                    self.model, 
                    self.eval_env, 
                    n_eval_episodes=self.n_eval_episodes, 
                    deterministic=self.deterministic,
                    return_episode_rewards=True 
                )
                
                mean_r = np.mean(episode_rewards)
                all_means.append(mean_r)
                all_lengths.append(np.mean(episode_lengths))

            # 1. Calculate the Global Score
            current_mean_reward = np.mean(all_means)
            current_mean_length = np.mean(all_lengths)

            # 2. Log to Tensorboard/Logger (standard EvalCallback style)
            self.logger.record("eval/pareto_avg_reward", current_mean_reward)
            self.logger.record("eval/pareto_avg_ep_length", current_mean_length)
            
            # Optional: Log specific weight performance for deeper insight
            for i, w in enumerate(self.pref_list):
                self.logger.record(f"eval/reward_w_{w[1]}", all_means[i])
            
            self.eval_env.env_method("unlock_preferences")

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={current_mean_reward:.2f} +/- {np.std(all_means):.2f}")
                print(f"Episode length: {current_mean_length:.2f} +/- {np.std(all_lengths):.2f}")

            # 3. Check if this is the "Best" model found so far
            if current_mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                
                self.best_mean_reward = current_mean_reward

            # Trigger potential logging for SB3 monitoring
            self.logger.dump(step=self.num_timesteps)

        return True


class CurriculumCallback(BaseCallback):
    """Gradually widens the set of budgets sampled during training.

    Parameters
    ----------
    budget_phases : list[list[int]]
        Budget lists for each phase.  Phase 0 is used from the start.
        Example: [[60,80], [60,80,100,200], [60,80,100,200,1000000]]
    phase_boundaries : list[float]
        Training progress thresholds (0→1) at which to advance.
        Example: [0.33, 0.66] means phase 1 starts at 33% and phase 2 at 66%.
    """

    def __init__(self, budget_phases, phase_boundaries, verbose=0):
        super().__init__(verbose)
        self.budget_phases = budget_phases
        self.phase_boundaries = phase_boundaries
        self._current_phase = 0

    def _on_training_start(self) -> None:
        self._update_envs(self.budget_phases[0])
        print(f"[Curriculum] Phase 0: budgets = {self.budget_phases[0]}")

    def _on_step(self) -> bool:
        if self._current_phase >= len(self.phase_boundaries):
            return True
        progress = 1.0 - self.model._current_progress_remaining
        if progress >= self.phase_boundaries[self._current_phase]:
            self._current_phase += 1
            new_budgets = self.budget_phases[self._current_phase]
            self._update_envs(new_budgets)
            print(f"[Curriculum] Phase {self._current_phase}: budgets = {new_budgets}")
        return True

    def _update_envs(self, budgets):
        env = self.model.get_env()
        while hasattr(env, 'venv'):
            env = env.venv
        env.env_method('set_active_budgets', budgets)


def linear_schedule(start: float, end: float = 0.0):
    def sched(progress_remaining):
        return (start - end) * progress_remaining + end
    return sched