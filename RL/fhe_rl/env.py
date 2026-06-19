import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pytrs import parse_sexpr, calculate_cost, NoiseEstimator, Expr, Const, Var, Op, expr_to_str
import torch
from .config import get_tokenizer_type

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

# Budget above this threshold is considered "unconstrained" (no utilization bonus)
UNCONSTRAINED_BUDGET_THRESHOLD = 100_000


class fheEnv(gym.Env):
    DEFAULT_BUDGET_OPTIONS = [240, 300, 1_000_000]
    
    def __init__(self, rules_list, expressions, max_positions=2, embeddings_model=None, budget_options=None, constraint_method="lagrangian_od_ov", verbose=True):
        super().__init__()
        self.rules = rules_list
        self.expressions = expressions
        self.noise_estimator = NoiseEstimator()
        self.max_positions = max_positions
        self.embeddings_model = embeddings_model
        self.constraint_method = constraint_method
        self.verbose = verbose
        self.max_steps = 75
        self.max_expression_size = 10000
        self.initial_cost = 0
        self.embedding_dim = 256
        self.budget_options = budget_options if budget_options is not None else self.DEFAULT_BUDGET_OPTIONS
        self.budget_dim = len(self.budget_options)
        self.active_budgets = list(self.budget_options)
        self.initial_vectorization_potential = 0
        self.vectorizations_applied = 0
        self.vectorization_helper = 0
        self.action_space = spaces.Discrete(len(self.rules.keys()) * self.max_positions)

        # Build observation space - add margin dimension for margin_barrier method
        obs_dict = {
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32),
            "budget_one_hot_encoding": spaces.Box(low=0, high=1, shape=(self.budget_dim,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(len(self.rules.keys())*self.max_positions,), dtype=np.float32),
        }
        if self.constraint_method == "margin_barrier":
            obs_dict["budget_margin"] = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if self.constraint_method == "nato_sc":
            obs_dict["noise_ratio"] = spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_dict)
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

        budget = np.random.choice(self.active_budgets)
        if isinstance(options, dict):
            budget = options.get("budget", budget)
        self.set_noise_budget(budget)

        noise = self.noise_estimator.estimate(self.expression)

        obs = {
            "observation": self._embed_expression(self.expression),
            "budget_one_hot_encoding": self.budget_one_hot_encoding,
            "action_mask": self.get_action_mask(),
        }
        if self.constraint_method == "margin_barrier":
            margin = np.clip((self.budget - noise) / max(self.budget, 1), -1.0, 1.0)
            obs["budget_margin"] = np.array([margin], dtype=np.float32)
        if self.constraint_method == "nato_sc":
            obs["noise_ratio"] = np.array([noise / max(self.budget, 1)], dtype=np.float32)

        return obs, {
            "expression": self.expression,
            "budget": self.budget,
            "noise": noise,
        }


    def step(self, action: int):
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

        info = {
            "expression": self.expression,
            "budget": self.budget,
            "noise": self.noise_estimator.estimate(self.expression),
            "cost": self.current_cost,
        }


        if self.verbose:
            reward_color = GREEN if reward >= 0 else RED
            print(f"{BOLD}{MAGENTA}New expression{RESET}: {YELLOW}{self.expression}{RESET}")
            print(f"{BOLD}{MAGENTA}New cost      {RESET}: {RED}{self.current_cost}{RESET}")
            print(f"{BOLD}{MAGENTA}Reward        {RESET}: {reward_color}{reward}{RESET}")
            print(f"{BOLD}{MAGENTA}Rule name     {RESET}: {CYAN}{rule_name}{RESET}")
            print(f"{BOLD}{MAGENTA}At position   {RESET}: {BLUE}{pos_idx}{RESET}")
            print(f"{BOLD}{MAGENTA}Budget         {RESET}: {YELLOW}{info['budget']}{RESET}")
            print(f"{BOLD}{MAGENTA}Noise         {RESET}: {YELLOW}{info['noise']}{RESET}")
            print(f"{CYAN}{'-'*100}{RESET}")

        embedding = self._embed_expression(self.expression)
        if embedding is None:
            terminated = True
            truncated = True
            reward = self.calculate_final_reward()
        else:
            terminated = terminated or (self.steps >= self.max_steps)

        # ── Margin barrier: override terminal reward with hard penalty + utilization ──
        if self.constraint_method == "margin_barrier" and (terminated or truncated):
            noise = info["noise"]
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
            noise = info["noise"]
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

        obs = {
            "observation": embedding,
            "budget_one_hot_encoding": self.budget_one_hot_encoding,
            "action_mask": self.get_action_mask()
        }
        if self.constraint_method == "margin_barrier":
            margin = np.clip((self.budget - info["noise"]) / max(self.budget, 1), -1.0, 1.0)
            obs["budget_margin"] = np.array([margin], dtype=np.float32)
        if self.constraint_method == "nato_sc":
            obs["noise_ratio"] = np.array([info["noise"] / max(self.budget, 1)], dtype=np.float32)

        return obs, reward, terminated, truncated, info
    
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
    
    def _embed_expression(self, expr: str) -> np.ndarray:
        expr_tree = parse_sexpr(expr)
        with torch.no_grad():
            emb = get_expression_cls_embedding(expr_tree, self.embeddings_model)
        if emb is None:
            return None
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

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
        """Update which budgets are sampled during reset (for curriculum training).
        The one-hot encoding still uses self.budget_options for consistent dims."""
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
