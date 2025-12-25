import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pytrs import parse_sexpr, calculate_cost, estimate_expression_noise, Expr, Const, Var, Op,expr_to_str
import torch
from .config import (
    get_tokenizer_type,
    get_budget_strategy,
    BudgetStrategy,
    BUDGET_OPTIONS,
    MIN_DYNAMIC_BUDGET,
    MAX_DYNAMIC_BUDGET,
    INFINITE_BUDGET_THRESHOLD,
    INFINITE_BUDGET_VALUE,
    DEFAULT_INFINITE_PROB,
)

if get_tokenizer_type() == "bpe":
    from .TRAE_bpe import get_expression_cls_embedding
else:
    from .TRAE import get_expression_cls_embedding



RESET   = "\033[0m"

BOLD    = "\033[1m"
DIM     = "\033[2m"

RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"


class fheEnv(gym.Env):
    def __init__(self, rules_list, expressions, max_positions=2,embeddings_model=None):
        
        super().__init__()
        self.rules = rules_list
        self.expressions = expressions
        self.max_positions = max_positions
        self.embeddings_model = embeddings_model
        self.max_steps =    75
        self.max_expression_size = 10000
        self.initial_cost = 0
        self.embedding_dim = 256
        self.noise_threshold = INFINITE_BUDGET_VALUE  # Updated per-episode
        self.initial_vectorization_potential = 0
        self.vectorizations_applied = 0
        self.vectorization_helper = 0
        self.action_space = spaces.Discrete(len(self.rules.keys()) * self.max_positions)
        self.budget_strategy = get_budget_strategy()
        self.budget_options = BUDGET_OPTIONS
        self.dynamic_budget_range = (MIN_DYNAMIC_BUDGET, MAX_DYNAMIC_BUDGET)
        self.dynamic_infinite_prob = DEFAULT_INFINITE_PROB
        self.current_budget = None
        self.test_budget = None
        self.budget_feature_dim = 0
        self.one_hot_dim = len(self.budget_options)

        obs_dim = self.embedding_dim
        if self.budget_strategy == BudgetStrategy.ONE_HOT:
            self.budget_feature_dim = self.one_hot_dim
            obs_dim += self.budget_feature_dim
        elif self.budget_strategy in (BudgetStrategy.BUDGET_THRESHOLD, BudgetStrategy.REMAINING_BUDGET):
            self.budget_feature_dim = 2  # [is_infinite, normalized_value]
            obs_dim += self.budget_feature_dim

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            "action_mask": spaces.Box(0, 1, (len(self.rules.keys()) * self.max_positions,), np.float32)
        })
        self.reset()

    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not hasattr(self, "current_index"):
            self.current_index = 0
        self.expression = self.expressions[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.expressions)
        self.initial_expression = self.expression
        self.steps = 0
        self.initial_cost = self.current_cost = self.get_cost(self.expression)
        self.current_budget = self._select_episode_budget()
        self._apply_budget_to_threshold()
        return {
            "observation": self._embed_expression(self.expression),
            "action_mask": self.get_action_mask()
        }, {}

    
    def step(self, action: int):
        self.steps += 1
        rule_idx = action // self.max_positions
        pos_idx = action % self.max_positions
        rule_name = list(self.rules.keys())[rule_idx]
        terminated = False
        truncated = False
        reward = 0

        print(f"\n{CYAN}{'-'*100}{RESET}")
        print(f"{BOLD}{MAGENTA}Old expression{RESET}: {YELLOW}{self.expression}{RESET}")
        print(f"{BOLD}{MAGENTA}Old cost      {RESET}: {RED}{self.current_cost}{RESET}")

        if rule_name == "END":
            terminated = True
            truncated = False
            reward = self.calculate_final_reward()
        else:
            parsed = parse_sexpr(self.expression)
            rule_obj = self.rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(parsed)
            k, _ = matches[pos_idx]
            new_expr_tree = rule_obj.apply_rule(parsed, path=k)
            temp = expr_to_str(new_expr_tree)
            self.expression = temp
            new_cost = self.get_cost(self.expression)
            reward = self.calculate_intermediate_reward(new_cost)
            self.current_cost = new_cost               
            if (self.steps >= self.max_steps):
                terminated = True
                reward = self.calculate_final_reward()

        info = {"expression": self.expression}

        noise = estimate_expression_noise(self.expression)
        noise = noise["noise_used"]
        info["noise"] = noise
        info["budget"] = self.current_budget
        info["budget_is_infinite"] = self._is_infinite_budget(self.current_budget)
        info["noise_threshold"] = self.noise_threshold

        reward_color = GREEN if reward >= 0 else RED

        print(f"{BOLD}{MAGENTA}New expression{RESET}: {YELLOW}{self.expression}{RESET}")
        print(f"{BOLD}{MAGENTA}New cost      {RESET}: {RED}{self.current_cost}{RESET}")
        print(f"{BOLD}{MAGENTA}Reward        {RESET}: {reward_color}{reward}{RESET}")
        print(f"{BOLD}{MAGENTA}Rule name     {RESET}: {CYAN}{rule_name}{RESET}")
        print(f"{BOLD}{MAGENTA}At position   {RESET}: {BLUE}{pos_idx}{RESET}")
        print(f"{BOLD}{MAGENTA}Noise         {RESET}: {YELLOW}{noise}{RESET}")
        print(f"{CYAN}{'-'*100}{RESET}")

        embedding = self._embed_expression(self.expression)
        if embedding is None:
            terminated = True
            truncated = True
            reward = self.calculate_final_reward()
        else:
            terminated = terminated or (self.steps >= self.max_steps)
        if terminated or truncated:
            info["episode"] = {
                "r": reward,
                "l": self.steps,
                "t": None
            }

        return {
            "observation": embedding,
            "action_mask": self.get_action_mask()
        }, reward, terminated, truncated, info
    
    def _valid_end_action(self,expr: str) -> bool:
        expr_tree = parse_sexpr(expr)
        vectorization_potenial = self.vectorisation_potential(expr)
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
                    if self.vectorisation_potential(temp) > vectorization_potenial:
                        isValid = False
                        break
            if not isValid:
                break
        return isValid
    
    def calculate_final_reward(self) -> float:
        if self.initial_cost == 0:
            return 0.0
        return (self.initial_cost - self.current_cost) / self.initial_cost * 100
    def calculate_intermediate_reward(self,new_cost) -> float:
        if self.current_cost == 0:
            return 0.0
        return ( ( self.current_cost - new_cost) / self.current_cost )
    
    def get_cost(self, expr: str) -> float:
        return calculate_cost(parse_sexpr(expr))
    
    def set_noise_threshold(self, noise_threshold: float):
        """Set the noise budget threshold for constraint calculations."""
        self.noise_threshold = float(noise_threshold)

    def set_test_budget(self, budget: float | None):
        """Force the environment to use a specific budget during reset (testing)."""
        if budget is None:
            self.test_budget = None
            return
        self.test_budget = float(budget)

    def clear_test_budget(self):
        self.test_budget = None

    # ────────────────────────────── Budget helpers ──────────────────────────────
    def _select_episode_budget(self) -> float:
        if self.test_budget is not None:
            return self.test_budget

        strategy = self.budget_strategy
        if strategy == BudgetStrategy.ONE_HOT:
            return float(np.random.choice(self.budget_options))
        if strategy in (BudgetStrategy.BUDGET_THRESHOLD, BudgetStrategy.REMAINING_BUDGET):
            return self._sample_dynamic_budget()

        # No special strategy: use current noise threshold
        return self.noise_threshold

    def _sample_dynamic_budget(self) -> float:
        if np.random.rand() < self.dynamic_infinite_prob:
            return INFINITE_BUDGET_VALUE
        low, high = self.dynamic_budget_range
        return float(np.random.uniform(low, high))

    def _apply_budget_to_threshold(self):
        if self.current_budget is None:
            return
        if self._is_infinite_budget(self.current_budget):
            self.noise_threshold = INFINITE_BUDGET_VALUE
        else:
            self.noise_threshold = float(self.current_budget)

    def _effective_budget_value(self) -> float:
        if self.current_budget is None:
            return self.noise_threshold
        if self._is_infinite_budget(self.current_budget):
            return INFINITE_BUDGET_VALUE
        return float(self.current_budget)

    @staticmethod
    def _is_infinite_budget(budget: float | None) -> bool:
        if budget is None:
            return False
        return budget > INFINITE_BUDGET_THRESHOLD

    def _normalized_budget_features(self, value: float | None, clamp_low_to_zero: bool) -> np.ndarray:
        if value is None:
            return np.zeros(2, dtype=np.float32)

        if self._is_infinite_budget(value):
            return np.array([1.0, 0.0], dtype=np.float32)

        low, high = self.dynamic_budget_range
        if clamp_low_to_zero:
            normalized = (value - low) / (high - low)
            normalized = np.clip(normalized, 0.0, 1.0)
        else:
            clipped = np.clip(value, low, high)
            normalized = (clipped - low) / (high - low)

        return np.array([0.0, normalized], dtype=np.float32)
    
    def _embed_expression(self, expr: str) -> np.ndarray:
        expr_tree = parse_sexpr(expr)
        with torch.no_grad():
            emb = get_expression_cls_embedding(expr_tree, self.embeddings_model)
        if emb is None:
            return None
        emb_np = emb.squeeze(0).cpu().numpy().astype(np.float32)

        if self.budget_strategy == BudgetStrategy.ONE_HOT:
            one_hot = np.zeros(self.one_hot_dim, dtype=np.float32)
            if self.current_budget in self.budget_options:
                idx = self.budget_options.index(self.current_budget)
                one_hot[idx] = 1.0
            return np.concatenate([emb_np, one_hot], axis=0)

        if self.budget_strategy == BudgetStrategy.BUDGET_THRESHOLD:
            features = self._normalized_budget_features(
                self.current_budget,
                clamp_low_to_zero=False
            )
            return np.concatenate([emb_np, features], axis=0)

        if self.budget_strategy == BudgetStrategy.REMAINING_BUDGET:
            noise_info = estimate_expression_noise(expr)
            current_noise = noise_info.get("noise_used", 0.0)
            remaining = max(self._effective_budget_value() - current_noise, 0.0)
            features = self._normalized_budget_features(
                remaining,
                clamp_low_to_zero=True
            )
            return np.concatenate([emb_np, features], axis=0)

        return emb_np
    
    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.rules.keys()) * self.max_positions, dtype=np.float32)
        parsed = parse_sexpr(self.expression)
        for rule_idx, rule_name in enumerate(self.rules.keys()):
            if rule_name == "END":
                mask[rule_idx * self.max_positions] = 1.0
                continue
            rule_obj = self.rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(parsed)
            valid_positions = min(len(matches), self.max_positions)
            if valid_positions > 0:
                start = rule_idx * self.max_positions
                mask[start:start + valid_positions] = 1.0
        return mask
    

