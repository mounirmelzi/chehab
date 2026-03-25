from stable_baselines3.common.callbacks import BaseCallback


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