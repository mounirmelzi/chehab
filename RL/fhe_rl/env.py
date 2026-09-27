import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pytrs import parse_sexpr, calculate_cost, NoiseEstimator, expr_to_str
import torch


RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"

# Budget above this threshold is considered "unconstrained" (no utilization bonus)
UNCONSTRAINED_BUDGET_THRESHOLD = 100_000


class fheEnv(gym.Env):
    DEFAULT_BUDGET_OPTIONS = [240, 300, 1_000_000]
    
    def __init__(self, rules_list, expressions, max_positions=2, embeddings_model=None, 
                 budget_options=None, constraint_method="lagrangian_pid", verbose=True,
                 pref_list=[[1.0, 0.0], [0.0, 1.0]], lambda_env=0.0, lambda_kl=0.0, 
                 n_cycle=1, n_budget=5, env_idx=0):
        super().__init__()
        
        # General parameters
        self.rules = rules_list
        self.expressions = expressions
        self.noise_estimator = NoiseEstimator()
        self.max_positions = max_positions
        self.embeddings_model = embeddings_model
        self.constraint_method = constraint_method
        self.verbose = verbose
        self.max_steps = 75
        self.max_expression_size = 10000
        self.embedding_dim = 256
        
        # Budget parameters (Melzi)
        self.budget_options = budget_options if budget_options is not None else self.DEFAULT_BUDGET_OPTIONS
        self.budget_dim = len(self.budget_options)
        self.active_budgets = list(self.budget_options)
        
        # MORL & Preference parameters (Imed)
        self.n_cycle = n_cycle
        self.episode_count = 0
        self.pref_list = [np.array(p, dtype=np.float32) for p in pref_list]
        self.current_pref_idx = env_idx % len(self.pref_list)
        self.current_w = self.pref_list[self.current_pref_idx]
        self.n_budget = n_budget
        self.lambda_kl = lambda_kl
        self.lambda_env = lambda_env
        self.pref_locked = False
        
        # State tracking
        self.initial_cost = 0
        self.current_cost = 0
        self.initial_ops = 0
        self.initial_keys = 0
        self.curr_ops = 0
        self.curr_keys = 0
        
        self.action_space = spaces.Discrete(len(self.rules.keys()) * self.max_positions)

        # Build combined observation space
        obs_dict = {
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32),
            "budget_one_hot_encoding": spaces.Box(low=0, high=1, shape=(self.budget_dim,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(len(self.rules.keys())*self.max_positions,), dtype=np.float32),
            "preference_vector": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        }
        if self.constraint_method == "margin_barrier":
            obs_dict["budget_margin"] = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            
        # Always expose noise_ratio to the policy (needed for nato_sc and lagrangian_pid)
        obs_dict["noise_ratio"] = spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Dict(obs_dict)
        # self.reset() called by SubprocVecEnv wrapper usually, omitted here to prevent double-reset

    def set_preference_vector(self, w):
        """Set the preference vector for multi-objective optimization"""
        self.current_w = np.array(w, dtype=np.float32)
        self.pref_locked = True

    def unlock_preferences(self):
        """Method to restore automatic cycling behavior"""
        self.pref_locked = False

    def get_split_costs(self, expr_str):
        parsed = parse_sexpr(expr_str)
        ops = calculate_cost(parsed, w_keys=0.0)
        keys = calculate_cost(parsed, w_keys=1.0) - ops
        return ops, keys    

    def _sample_preference(self) -> tuple[np.ndarray, str]:
        """
        Implements biased preference sampling:
            episode_count mod (n_cycle + 1) < n_cycle  ->  speed-focus [1, 0]
            otherwise                                  ->  random from Ω
        """
        if self.n_cycle > 0 and (self.episode_count % (self.n_cycle + 1)) < self.n_cycle:
            # Speed-focus episode
            w         = self.pref_list[0]          # [1.0, 0.0] by convention
            mode_name = "FIXED SPEED FOCUS"
        else:
            # Random exploration episode
            idx       = self.np_random.integers(0, len(self.pref_list))
            w         = self.pref_list[idx]
            mode_name = "RANDOM SEARCH"
 
        self.episode_count += 1
        return w, mode_name

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not hasattr(self, "current_index"):
            self.current_index = 0
            
        self.expression = self.expressions[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.expressions)
        self.initial_expression = self.expression
        self.steps = 0

        # Base costs
        self.initial_cost = self.current_cost = self.get_cost(self.expression)
        self.initial_ops, self.initial_keys = self.get_split_costs(self.expression)
        self.curr_ops, self.curr_keys = self.initial_ops, self.initial_keys

        # Budget sampling
        budget = np.random.choice(self.active_budgets)
        if isinstance(options, dict):
            budget = options.get("budget", budget)
        self.set_noise_budget(budget)
        noise = self.noise_estimator.estimate(self.expression)

        # Preference sampling
        if self.pref_locked:
            mode_name = "LOCKED (EVALUATION)"
        else:
            self.current_w, mode_name = self._sample_preference()

        if self.verbose:
            print(f"\n{BLUE}{'='*40} EPISODE START {'='*40}{RESET}")
            print(f"{BOLD}{MAGENTA}Optimization Mode{RESET}  : {CYAN}{mode_name}{RESET}") 
            print(f"{BOLD}{MAGENTA}Noise Budget     {RESET}  : {YELLOW}{self.budget}{RESET}") 
                    
        obs = self._get_obs(noise)

        return obs, {
            "expression": self.expression,
            "budget": self.budget,
            "noise": noise,
            "cost": self.current_cost,
            "c_exec": self.curr_ops,
            "c_keys": self.curr_keys
        }

    def _get_obs(self, noise: float):
        embedding = self._embed_expression(self.expression)
        if embedding is None:
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            
        w = np.atleast_1d(self.current_w).astype(np.float32)
        
        obs = {
            "observation": embedding,
            "budget_one_hot_encoding": self.budget_one_hot_encoding,
            "action_mask": self.get_action_mask(),
            "preference_vector": w
        }
        
        if self.constraint_method == "margin_barrier":
            margin = np.clip((self.budget - noise) / max(self.budget, 1), -1.0, 1.0)
            obs["budget_margin"] = np.array([margin], dtype=np.float32)
            
        obs["noise_ratio"] = np.array([noise / max(self.budget, 1)], dtype=np.float32)
        return obs

    def step(self, action: int):
        # Prevent wasting steps on 100% key-reduction since scalar code has 0 keys
        if self.current_w[0] == 0.0 and self.current_w[1] == 1.0:
            action = list(self.rules.keys()).index("END") * self.max_positions
            
        self.steps += 1
        rule_idx = action // self.max_positions
        pos_idx = action % self.max_positions
        rule_name = list(self.rules.keys())[rule_idx]
        terminated = False
        truncated = False
        reward = 0

        if self.verbose:
            print(f"\n{CYAN}{'-'*100}{RESET}")
            print(f"{BOLD}{MAGENTA}Old expression{RESET}: {YELLOW}{self.expression}{RESET}")
            print(f"{BOLD}{MAGENTA}Old exec cost      {RESET}: {RED}{self.curr_ops}{RESET}")
            print(f"{BOLD}{MAGENTA}Old keys cost      {RESET}: {RED}{self.curr_keys}{RESET}")

        if rule_name == "END":
            terminated = True
            truncated = False
            reward = self.calculate_final_reward()
            noise = self.noise_estimator.estimate(self.expression)
        else:
            parsed = parse_sexpr(self.expression)
            rule_obj = self.rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(parsed)
            k, _ = matches[pos_idx]
            new_expr_tree = rule_obj.apply_rule(parsed, path=k)
            temp = expr_to_str(new_expr_tree)
            self.expression = temp
            
            new_cost = self.get_cost(self.expression)
            new_ops, new_keys = self.get_split_costs(self.expression)
            noise = self.noise_estimator.estimate(self.expression)

            reward = self.calculate_intermediate_reward(new_ops, new_keys)

            self.curr_ops = new_ops
            self.curr_keys = new_keys
            self.current_cost = new_cost              
            
            if (self.steps >= self.max_steps):
                terminated = True
                reward = self.calculate_final_reward()

        info = {
            "expression": self.expression,
            "budget": self.budget,
            "noise": noise,
            "cost": self.current_cost,
            "c_exec": self.curr_ops,
            "c_keys": self.curr_keys
        }

        if self.verbose:
            reward_color = GREEN if reward >= 0 else RED
            print(f"{BOLD}{MAGENTA}New expression{RESET}: {YELLOW}{self.expression}{RESET}")
            print(f"{BOLD}{MAGENTA}New exec cost      {RESET}: {GREEN}{self.curr_ops}{RESET}")
            print(f"{BOLD}{MAGENTA}New keys cost      {RESET}: {GREEN}{self.curr_keys}{RESET}")
            print(f"{BOLD}{MAGENTA}Reward        {RESET}: {reward_color}{reward}{RESET}")
            print(f"{BOLD}{MAGENTA}Rule name     {RESET}: {CYAN}{rule_name}{RESET}")
            print(f"{BOLD}{MAGENTA}At position   {RESET}: {BLUE}{pos_idx}{RESET}")
            print(f"{BOLD}{MAGENTA}Budget         {RESET}: {YELLOW}{info['budget']}{RESET}")
            print(f"{BOLD}{MAGENTA}Noise         {RESET}: {YELLOW}{info['noise']}{RESET}")
            print(f"{CYAN}{'-'*100}{RESET}")

        obs = self._get_obs(noise)
        
        # If embedder fails, early terminate
        if self._embed_expression(self.expression) is None:
            terminated = True
            truncated = True
            reward = self.calculate_final_reward()

        # ── Margin barrier: override terminal reward with hard penalty + utilization ──
        if self.constraint_method == "margin_barrier" and (terminated or truncated):
            if noise > self.budget:
                reward = -100.0
            else:
                cost_reward = self.calculate_final_reward()
                if self.budget <= UNCONSTRAINED_BUDGET_THRESHOLD:
                    utilization = noise / max(self.budget, 1)
                    reward = cost_reward + utilization * 5.0
                else:
                    reward = cost_reward

        # ── NATO-SC: quadratic terminal penalty, allows intermediate violations ──
        if self.constraint_method == "nato_sc" and (terminated or truncated):
            cost_reward = self.calculate_final_reward()
            if noise > self.budget:
                violation_ratio = (noise - self.budget) / max(self.budget, 1)
                reward = cost_reward - 50.0 * (violation_ratio ** 2)
            else:
                if self.budget <= UNCONSTRAINED_BUDGET_THRESHOLD:
                    utilization = noise / max(self.budget, 1)
                    reward = cost_reward + utilization * 5.0
                else:
                    reward = cost_reward

        if terminated or truncated:
            info["episode"] = {
                "r": reward,
                "l": self.steps,
                "t": None
            }

        return obs, reward, terminated, truncated, info
    
    def _valid_end_action(self,expr: str) -> bool:
        expr_tree = parse_sexpr(expr)
        action_mask = self.get_action_mask()
        isValid = True
        for i, rule_name in enumerate(self.rules.keys()):
            if rule_name == "END":
                continue
            rule_obj = self.rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(expr_tree)
            if len(matches) > 0:
                for i,match in enumerate( matches):
                    if i >= self.max_positions:
                        break
                    k, _ = match
                    new_expr_tree = rule_obj.apply_rule(expr_tree, path=k)
                    temp = expr_to_str(new_expr_tree)
                    if calculate_cost(new_expr_tree) < self.current_cost:
                        isValid = False
                        break
            if not isValid:
                break
        return isValid
    
    # ── MORL Reward Logic ──
    def _reward_vector(self, delta_ops_old, delta_ops_new,
                       delta_keys_old, delta_keys_new) -> np.ndarray:
        """Compute the 2-D reward vector [r_ops, r_keys].

        FIX: r_keys was previously normalized by a fixed constant
        (self.n_budget, default 5) instead of by delta_keys_old the
        way r_ops is normalized by delta_ops_old. Since key-cost is
        typically on a much larger absolute scale than op-count,
        dividing by a small fixed constant made r_keys' magnitude
        dwarf r_ops' magnitude on ANY step that touched a rotation
        key, regardless of w_keys. That made the composed reward
        effectively ignore w_ops/w_keys and treat every rotation as
        catastrophic, so the trained policy converged to never
        emitting rotate actions at any preference setting, and at
        any checkpoint (this is baked into training, not something
        a later/earlier checkpoint would fix).

        Now both terms are relative improvements on the same
        [-1, 1]-ish scale, so the preference weights in
        _compose_reward actually control the tradeoff as intended.
        """
        r_ops  = (delta_ops_old  - delta_ops_new)  / delta_ops_old  if delta_ops_old  != 0 else 0.0
        r_keys = (delta_keys_old - delta_keys_new) / delta_keys_old if delta_keys_old != 0 else 0.0
        return np.array([r_ops, r_keys], dtype=np.float32)

    def _kl_bonus(self, r_vec: np.ndarray) -> float:
        """KL divergence of the preference-weighted improvement distribution"""
        r_pos       = np.maximum(r_vec, 0.0) + 1e-6
        r_tilde     = r_pos / np.sum(r_pos)
        uniform     = np.full_like(r_tilde, 1.0 / len(r_tilde))
        return float(np.sum(r_tilde * np.log((r_tilde + 1e-6) / uniform)))

    def _pareto_envelope_bonus(self, r_vec: np.ndarray) -> float:
        """max_{w ∈ Ω} w · r⃗  — reward under the best-fitting preference."""
        return float(max(np.dot(p, r_vec) for p in self.pref_list))

    def _compose_reward(self, r_vec: np.ndarray) -> float:
        """Total reward = linear term + optional bonuses."""
        reward = float(np.dot(self.current_w, r_vec))
        if self.lambda_env != 0.0:
            reward += self.lambda_env * self._pareto_envelope_bonus(r_vec)
        if self.lambda_kl != 0.0:
            reward += self.lambda_kl * self._kl_bonus(r_vec)
        return reward
    
    def calculate_intermediate_reward(self, new_ops: float, new_keys: float) -> float:
        r_vec = self._reward_vector(self.curr_ops, new_ops, self.curr_keys, new_keys)
        return self._compose_reward(r_vec)
 
    def calculate_final_reward(self) -> float:
        r_vec = self._reward_vector(self.initial_ops, self.curr_ops,
                                    self.initial_keys, self.curr_keys)
        return self._compose_reward(r_vec) * 100
    
    def get_cost(self, expr: str) -> float:
        return calculate_cost(parse_sexpr(expr))
    
    def _embed_expression(self, expr: str) -> np.ndarray:
        if hasattr(self.embeddings_model, "get_embedding"):
            return self.embeddings_model.get_embedding(expr)
        return None

    def set_noise_budget(self, budget: int | None):
        if budget is None:
            self.budget = None
            self.budget_one_hot_encoding = None
            return
        self.budget = budget
        self.budget_one_hot_encoding = np.zeros(self.budget_dim, dtype=np.float32)
        if budget in self.budget_options:
            budget_idx = self.budget_options.index(budget)
        else:
            # Test budget not in training set — use nearest training budget for encoding
            budget_idx = min(range(len(self.budget_options)),
                             key=lambda i: abs(self.budget_options[i] - budget))
        self.budget_one_hot_encoding[budget_idx] = 1.0

    def set_active_budgets(self, budgets):
        """Update which budgets are sampled during reset."""
        self.active_budgets = list(budgets)
    
    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.rules.keys()) * self.max_positions, dtype=np.float32)
        parsed = parse_sexpr(self.expression)
        use_noise_mask = (
            self.constraint_method == "noise_masking"
            and self.budget is not None
            and self.budget < UNCONSTRAINED_BUDGET_THRESHOLD
        )
        for rule_idx, rule_name in enumerate(self.rules.keys()):
            if rule_name == "END":
                mask[rule_idx * self.max_positions] = 1.0
                continue
            rule_obj = self.rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(parsed)
            valid_positions = min(len(matches), self.max_positions)
            if valid_positions > 0:
                start = rule_idx * self.max_positions
                if use_noise_mask:
                    for pos_idx in range(valid_positions):
                        k, _ = matches[pos_idx]
                        try:
                            new_expr_tree = rule_obj.apply_rule(parsed, path=k)
                            noise_est = self.noise_estimator.estimate(new_expr_tree)
                            if noise_est <= self.budget:
                                mask[start + pos_idx] = 1.0
                        except Exception:
                            pass
                else:
                    mask[start:start + valid_positions] = 1.0
        return mask