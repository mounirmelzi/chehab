import torch, torch.nn as nn
from torch.distributions import Categorical
from .utils import mlp
import gymnasium as gym
from typing import Any, Dict, Tuple, Union
import numpy as np


class CustomFeaturesExtractor(nn.Module):
    BUDGET_EMBED_DIM = 32

    def __init__(self, observation_space):
        super().__init__()
        self._embed_dim = observation_space["observation"].shape[0]
        self._budget_dim = observation_space["budget_one_hot_encoding"].shape[0]

        self.film_gamma = nn.Sequential(
            nn.Linear(self._budget_dim, self.BUDGET_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(self.BUDGET_EMBED_DIM, self._embed_dim),
        )
        nn.init.zeros_(self.film_gamma[-1].weight)
        nn.init.ones_(self.film_gamma[-1].bias)

        self.film_beta = nn.Sequential(
            nn.Linear(self._budget_dim, self.BUDGET_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(self.BUDGET_EMBED_DIM, self._embed_dim),
        )
        nn.init.zeros_(self.film_beta[-1].weight)
        nn.init.zeros_(self.film_beta[-1].bias)

        self.budget_encoder = nn.Sequential(
            nn.Linear(self._budget_dim, self.BUDGET_EMBED_DIM),
            nn.ReLU(),
        )

    def forward(self, obs_dict):
        obs, budget = obs_dict["observation"], obs_dict["budget_one_hot_encoding"]
        gamma, beta = self.film_gamma(budget), self.film_beta(budget)
        return torch.cat((gamma * obs + beta, self.budget_encoder(budget)), dim=1)

    @property
    def features_dim(self):
        return self._embed_dim + self.BUDGET_EMBED_DIM


class HierarchicalMaskablePolicy(nn.Module):
    """Rule‑then‑position actor‑critic with action masks."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space,  lr_schedule: Union[float, Any], **kwargs):
        super().__init__()
        self.rule_dim: int       = kwargs.pop("rule_dim", 5)
        self.max_positions: int  = kwargs.pop("max_positions", 32)
        lr: float               = kwargs.pop("lr", 3e-4)
        rule_hidden_dims        = kwargs.pop("rule_hidden_dims", [128, 128])
        pos_hidden_dims         = kwargs.pop("pos_hidden_dims", [128, 128])
        value_hidden_dims       = kwargs.pop("value_hidden_dims", [256, 128, 64])

        self.encoder = CustomFeaturesExtractor(observation_space)
        self.rule_head = mlp(self.encoder.features_dim, rule_hidden_dims, self.rule_dim, layernorm=True, dropout=0.1)
        self.pos_head  = mlp(self.encoder.features_dim + self.rule_dim, pos_hidden_dims, self.max_positions, layernorm=True, dropout=0.1)
        self.value_net = mlp(self.encoder.features_dim, value_hidden_dims, 1, layernorm=True, dropout=0.1)

        actor_params  = list(self.encoder.parameters()) + list(self.rule_head.parameters()) + list(self.pos_head.parameters())
        critic_params = self.value_net.parameters()
        self.optimizer = torch.optim.Adam([
            {"params": actor_params,  "lr": lr},
            {"params": critic_params, "lr": lr},
        ])
        # Entropy coefficient (tuned separately)
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
        
        # Ensure actions is a tensor
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=enc.device)
        
        # Ensure actions is on the same device as the encoder
        if actions.device != enc.device:
            actions = actions.to(enc.device)
        
        # Handle both 1D and 2D action tensors
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
    # Convenience wrappers for SB3 compatibility
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
        
        # Convert to tensors and ensure float type
        if isinstance(observation, dict):
            obs = {k: torch.as_tensor(v, device=device, dtype=torch.float32) 
                for k, v in observation.items()}
            # Add batch dim if needed
            if obs[next(iter(obs))].dim() == 1:
                obs = {k: v.unsqueeze(0) for k, v in obs.items()}
        else:
            obs = torch.as_tensor(observation, device=device, dtype=torch.float32)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            actions, _, _ = self.forward(obs, deterministic=deterministic)
        
        # Convert to numpy but keep the batch dimension for vec envs
        actions = actions.cpu().numpy()
        
        # Don't squeeze if we're dealing with vec envs - return shape should be (n_envs,)
        # even if n_envs=1
        if len(actions.shape) == 0:  # If it's a scalar
            actions = np.array([actions])  # Make it 1D array with one element
        
        return actions, state
