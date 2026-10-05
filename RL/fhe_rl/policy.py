import torch
import torch.nn as nn
from torch.distributions import Categorical
from .utils import mlp
import gymnasium as gym
from typing import Any, Dict, Tuple, Union
import numpy as np 


class CustomFeaturesExtractor(nn.Module):
    """Encodes observation + budget + preferences into a flat feature vector.

    budget_encoding modes:
      - "raw":   concat raw one-hot
      - "embed": learned dense embedding from one-hot
      - "film":  FiLM conditioning — budget modulates expression features via
                 gamma * expr_emb + beta
    """

    BUDGET_EMBED_DIM = 32  # Dense budget embedding dimension

    def __init__(self, observation_space, budget_encoding="raw"):
        super().__init__()
        self._embed_dim = observation_space["observation"].shape[0]        # 256
        self._has_budget = "budget_one_hot_encoding" in observation_space.spaces
        self._budget_dim = observation_space["budget_one_hot_encoding"].shape[0] if self._has_budget else 0
        self._has_margin = "budget_margin" in observation_space.spaces
        self._margin_dim = observation_space["budget_margin"].shape[0] if self._has_margin else 0
        self._has_noise_ratio = "noise_ratio" in observation_space.spaces
        self._noise_ratio_dim = observation_space["noise_ratio"].shape[0] if self._has_noise_ratio else 0
        
        # Added preference vector support for MORL (Imed's addition)
        self._has_pref = "preference_vector" in observation_space.spaces
        self._pref_dim = observation_space["preference_vector"].shape[0] if self._has_pref else 0

        self._budget_encoding = budget_encoding if self._has_budget else "none"

        if budget_encoding == "embed":
            self.budget_encoder = nn.Sequential(
                nn.Linear(self._budget_dim, self.BUDGET_EMBED_DIM),
                nn.ReLU(),
            )

        elif budget_encoding == "film":
            self.film_gamma = nn.Sequential(
                nn.Linear(self._budget_dim, self.BUDGET_EMBED_DIM),
                nn.ReLU(),
                nn.Linear(self.BUDGET_EMBED_DIM, self._embed_dim),
            )
            self.film_beta = nn.Sequential(
                nn.Linear(self._budget_dim, self.BUDGET_EMBED_DIM),
                nn.ReLU(),
                nn.Linear(self.BUDGET_EMBED_DIM, self._embed_dim),
            )
            nn.init.zeros_(self.film_gamma[-1].weight)
            nn.init.ones_(self.film_gamma[-1].bias)
            nn.init.zeros_(self.film_beta[-1].weight)
            nn.init.zeros_(self.film_beta[-1].bias)

            self.budget_encoder = nn.Sequential(
                nn.Linear(self._budget_dim, self.BUDGET_EMBED_DIM),
                nn.ReLU(),
            )

    def forward(self, obs_dict):
        expr_emb = obs_dict["observation"]

        if self._budget_encoding == "none":
            parts = [expr_emb]

        elif self._budget_encoding == "raw":
            parts = [expr_emb, obs_dict["budget_one_hot_encoding"]]

        elif self._budget_encoding == "embed":
            budget_emb = self.budget_encoder(obs_dict["budget_one_hot_encoding"])
            parts = [expr_emb, budget_emb]

        elif self._budget_encoding == "film":
            budget_oh = obs_dict["budget_one_hot_encoding"]
            gamma = self.film_gamma(budget_oh)      # (B, 256)
            beta  = self.film_beta(budget_oh)        # (B, 256)
            modulated = gamma * expr_emb + beta      # budget gates expression features
            budget_emb = self.budget_encoder(budget_oh)
            parts = [modulated, budget_emb]

        else:
            raise ValueError(f"Unknown budget_encoding: {self._budget_encoding}")

        if self._has_margin:
            parts.append(obs_dict["budget_margin"])
        if self._has_noise_ratio:
            parts.append(obs_dict["noise_ratio"])
        if self._has_pref:
            parts.append(obs_dict["preference_vector"])
            
        return torch.cat(parts, dim=1)

    @property
    def features_dim(self):
        if self._budget_encoding == "none":
            base = self._embed_dim
        elif self._budget_encoding == "raw":
            base = self._embed_dim + self._budget_dim
        else:  # "embed" or "film"
            base = self._embed_dim + self.BUDGET_EMBED_DIM
        return base + self._margin_dim + self._noise_ratio_dim + self._pref_dim


class HierarchicalMaskablePolicy(nn.Module):
    """Rule‑then‑position actor‑critic with action masks."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, lr_schedule: Union[float, Any], **kwargs):
        super().__init__()

        seed = kwargs.pop("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)       

        self.rule_dim: int      = kwargs.pop("rule_dim", 5)
        self.max_positions: int = kwargs.pop("max_positions", 32)
        lr: float               = kwargs.pop("lr", 3e-4)
        budget_encoding: str    = kwargs.pop("budget_encoding", "raw")
        
        rule_hidden_dims        = kwargs.pop("rule_hidden_dims", [128, 128])
        pos_hidden_dims         = kwargs.pop("pos_hidden_dims", [128, 128])
        value_hidden_dims       = kwargs.pop("value_hidden_dims", [256, 128, 64])

        self.encoder = CustomFeaturesExtractor(observation_space, budget_encoding=budget_encoding)
        feat_dim = self.encoder.features_dim
        
        self.rule_head = mlp(feat_dim, rule_hidden_dims, self.rule_dim, layernorm=True)
        self.pos_head  = mlp(feat_dim + self.rule_dim, pos_hidden_dims, self.max_positions, layernorm=True)
        self.value_net = mlp(feat_dim, value_hidden_dims, 1, layernorm=True)

        actor_params  = list(self.encoder.parameters()) + list(self.rule_head.parameters()) + list(self.pos_head.parameters())
        critic_params = self.value_net.parameters()
        self.optimizer = torch.optim.Adam([
            {"params": actor_params,  "lr": lr},
            {"params": critic_params, "lr": lr},
        ])
        
        self.ent_coef = kwargs.pop("ent_coef", 0.01)

    # ───────── Helper distributions ──────────
    def _rule_dist(self, enc: torch.Tensor, rule_mask: torch.Tensor) -> Categorical:
        logits = torch.where(rule_mask, self.rule_head(enc), torch.finfo(torch.float32).min)
        return Categorical(logits=logits)

    def _pos_dist(self, enc: torch.Tensor, one_hot_rule: torch.Tensor, pos_mask: torch.Tensor) -> Categorical:
        logits = self.pos_head(torch.cat([enc, one_hot_rule], dim=1))
        logits = torch.where(pos_mask, logits, torch.finfo(torch.float32).min)
        return Categorical(logits=logits)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        """SB3 internal helper – returns (actions, value, log_prob) where log_prob is **combined** rule+pos."""
        actions, value, rule_logp, pos_logp, _, _ = self.forward_separate(obs, deterministic)
        return actions, value, rule_logp + pos_logp

    # ───────── Forward (separate log‑probs) ──────────
    def forward_separate(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        enc = self.encoder(obs)
        mask = obs["action_mask"].bool()
        B = mask.size(0)
        mask = mask.view(B, self.rule_dim, self.max_positions)
        rule_mask = mask.any(dim=2)
        rule_dist = self._rule_dist(enc, rule_mask)
        rule_action = rule_dist.mode if deterministic else rule_dist.sample()
        rule_logp = rule_dist.log_prob(rule_action)

        one_hot = torch.nn.functional.one_hot(rule_action, self.rule_dim).float()
        pos_mask = mask[torch.arange(B, device=enc.device), rule_action]
        pos_dist = self._pos_dist(enc, one_hot, pos_mask)
        pos_action = pos_dist.mode if deterministic else pos_dist.sample()
        pos_logp = pos_dist.log_prob(pos_action)

        flat_action = rule_action * self.max_positions + pos_action
        value = self.value_net(enc).squeeze(-1)
        entropy_rule = rule_dist.entropy()
        entropy_pos  = pos_dist.entropy()
        return flat_action, value, rule_logp, pos_logp, entropy_rule, entropy_pos

    # ───────── Evaluate for PPO update ──────────
    def evaluate_actions_separate(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor):
        enc = self.encoder(obs)
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=enc.device)
        
        if actions.device != enc.device:
            actions = actions.to(enc.device)
        
        if actions.dim() == 2:
            actions = actions.squeeze(-1)
        
        B = actions.shape[0]
        mask = obs["action_mask"].bool().view(B, self.rule_dim, self.max_positions)
        rule_actions = (actions // self.max_positions).long()
        pos_actions  = (actions % self.max_positions).long()
        rule_mask = mask.any(dim=2)
        rule_dist = self._rule_dist(enc, rule_mask)
        rule_logp = rule_dist.log_prob(rule_actions)
        one_hot   = torch.nn.functional.one_hot(rule_actions, self.rule_dim).float()
        pos_mask  = mask[torch.arange(B, device=enc.device), rule_actions]
        pos_dist  = self._pos_dist(enc, one_hot, pos_mask)
        pos_logp  = pos_dist.log_prob(pos_actions)
        entropy_rule = rule_dist.entropy()
        entropy_pos  = pos_dist.entropy()
        value = self.value_net(enc).squeeze(-1)
        return value, rule_logp, pos_logp, entropy_rule, entropy_pos

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor):
        """Wrapper for SB3 PPO which expects combined log‑prob & entropy."""
        value, rule_lp, pos_lp, ent_r, ent_p = self.evaluate_actions_separate(obs, actions)
        return value, rule_lp + pos_lp, ent_r + ent_p

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.value_net(self.encoder(obs)).squeeze(-1)

    def set_training_mode(self, mode: bool):
        self.train(mode)

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        device = next(self.parameters()).device
        
        if isinstance(observation, dict):
            obs = {k: torch.as_tensor(v, device=device, dtype=torch.float32) 
                for k, v in observation.items()}
            if obs[next(iter(obs))].dim() == 1:
                obs = {k: v.unsqueeze(0) for k, v in obs.items()}
        else:
            obs = torch.as_tensor(observation, device=device, dtype=torch.float32)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            actions, _, _ = self.forward(obs, deterministic=deterministic)
        
        actions = actions.cpu().numpy()
        
        if len(actions.shape) == 0:  
            actions = np.array([actions])  
            
        return actions, state