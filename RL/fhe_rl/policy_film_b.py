"""Design B feature extractor/policy for the FiLM A/B comparison.

Design A (current `policy.py`, `CustomFeaturesExtractor`, budget_encoding="film"):
    gamma, beta = film(budget_one_hot)         # (B, 256), conditioned on budget ONLY
    modulated   = gamma * expr_emb + beta
    features    = concat([modulated, budget_emb, margin?, noise_ratio?, preference_vector])
    -> preference never passes through FiLM; it's grafted on at the very end.

Design B (this file):
    cond        = concat([budget_one_hot, preference_vector])   # (B, budget_dim + 2)
    gamma, beta = film(cond)                    # (B, 256), conditioned on BOTH
    modulated   = gamma * expr_emb + beta
    features    = concat([modulated, budget_emb, margin?, noise_ratio?])
    -> preference is fused into the expression representation itself, before the
       policy heads, so the network can learn budget×preference interactions
       (e.g. "tight budget + memory-focused" should suppress different features
       than "tight budget + speed-focused").

Both share identical rule/pos/value heads (HierarchicalMaskablePolicy unchanged) —
only the encoder differs. This isolates the comparison to exactly the one design
decision in question, matching CHEHAB_unification_plan Phase 2.

Usage (see __main__.py / train.py wiring):
    python -m fhe_rl train --policy_variant film_b ...   # Design B
    python -m fhe_rl train --policy_variant film_a ...   # Design A (default, unchanged)
"""

import torch
import torch.nn as nn
import gymnasium as gym
from typing import Any, Union

from .policy import HierarchicalMaskablePolicy


class CustomFeaturesExtractorFiLMB(nn.Module):
    """Same interface as policy.CustomFeaturesExtractor, but FiLM is conditioned
    on concat[budget_one_hot, preference_vector] instead of budget_one_hot alone.

    Only meaningful with budget_encoding == "film" AND a preference_vector present
    in the observation space; falls back to identical behavior to Design A's other
    modes ("raw", "embed", "none") since those don't touch FiLM at all.
    """

    BUDGET_EMBED_DIM = 32

    def __init__(self, observation_space, budget_encoding="film"):
        super().__init__()
        self._embed_dim = observation_space["observation"].shape[0]
        self._has_budget = "budget_one_hot_encoding" in observation_space.spaces
        self._budget_dim = observation_space["budget_one_hot_encoding"].shape[0] if self._has_budget else 0
        self._has_margin = "budget_margin" in observation_space.spaces
        self._margin_dim = observation_space["budget_margin"].shape[0] if self._has_margin else 0
        self._has_noise_ratio = "noise_ratio" in observation_space.spaces
        self._noise_ratio_dim = observation_space["noise_ratio"].shape[0] if self._has_noise_ratio else 0
        self._has_pref = "preference_vector" in observation_space.spaces
        self._pref_dim = observation_space["preference_vector"].shape[0] if self._has_pref else 0

        self._budget_encoding = budget_encoding if self._has_budget else "none"

        # Only "film" differs from Design A; other modes are identical, kept for
        # CLI compatibility if someone runs --budget_encoding raw/embed/none with
        # --policy_variant film_b by mistake (degrades gracefully to Design A's
        # non-FiLM behavior rather than erroring).
        if self._budget_encoding == "embed":
            self.budget_encoder = nn.Sequential(
                nn.Linear(self._budget_dim, self.BUDGET_EMBED_DIM),
                nn.ReLU(),
            )

        elif self._budget_encoding == "film":
            # KEY DIFFERENCE: FiLM input dim includes preference_vector.
            self._film_in_dim = self._budget_dim + self._pref_dim
            self.film_gamma = nn.Sequential(
                nn.Linear(self._film_in_dim, self.BUDGET_EMBED_DIM),
                nn.ReLU(),
                nn.Linear(self.BUDGET_EMBED_DIM, self._embed_dim),
            )
            self.film_beta = nn.Sequential(
                nn.Linear(self._film_in_dim, self.BUDGET_EMBED_DIM),
                nn.ReLU(),
                nn.Linear(self.BUDGET_EMBED_DIM, self._embed_dim),
            )
            # Same near-identity init as Design A: at step 0, before any
            # preference-conditioning is learned, this behaves like a no-op FiLM
            # exactly like Design A does for budget alone. Preserves the same
            # "don't break what already works" property Design A relies on.
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
            if self._has_pref:
                cond = torch.cat([budget_oh, obs_dict["preference_vector"]], dim=1)
            else:
                cond = budget_oh
            gamma = self.film_gamma(cond)
            beta = self.film_beta(cond)
            modulated = gamma * expr_emb + beta
            budget_emb = self.budget_encoder(budget_oh)
            parts = [modulated, budget_emb]

        else:
            raise ValueError(f"Unknown budget_encoding: {self._budget_encoding}")

        if self._has_margin:
            parts.append(obs_dict["budget_margin"])
        if self._has_noise_ratio:
            parts.append(obs_dict["noise_ratio"])

        # KEY DIFFERENCE vs Design A: preference_vector is NOT appended here.
        # It has already been fused into `modulated` via FiLM above.

        return torch.cat(parts, dim=1)

    @property
    def features_dim(self):
        if self._budget_encoding == "none":
            base = self._embed_dim
        elif self._budget_encoding == "raw":
            base = self._embed_dim + self._budget_dim
        else:  # "embed" or "film"
            base = self._embed_dim + self.BUDGET_EMBED_DIM
        # NOTE: no + self._pref_dim here, unlike Design A's features_dim —
        # preference is consumed by FiLM, not concatenated downstream.
        return base + self._margin_dim + self._noise_ratio_dim


class HierarchicalMaskablePolicyFiLMB(HierarchicalMaskablePolicy):
    """Identical to HierarchicalMaskablePolicy except it builds its encoder from
    CustomFeaturesExtractorFiLMB instead of CustomFeaturesExtractor.

    All heads (rule_head, pos_head, value_net), the optimizer, and every method
    are inherited unchanged from the parent -- only self.encoder differs, so any
    observed performance gap is attributable to the FiLM design decision alone.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 lr_schedule: Union[float, Any], **kwargs):
        # Run the parent __init__ fully (builds encoder=A, heads, optimizer, etc.)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Then replace just the encoder with Design B and rebuild the optimizer's
        # param groups so they reference the new encoder's parameters instead of
        # the discarded Design-A encoder (otherwise gradients would silently not
        # flow into film_gamma_b/film_beta_b).
        budget_encoding = kwargs.get("budget_encoding", "raw")
        lr = kwargs.get("lr", 3e-4)

        self.encoder = CustomFeaturesExtractorFiLMB(observation_space, budget_encoding=budget_encoding)
        # features_dim differs (no +pref_dim) -> heads must be rebuilt too, since
        # rule_head/pos_head/value_net input dims were sized off Design A's dim.
        from .utils import mlp
        feat_dim = self.encoder.features_dim
        rule_hidden_dims = kwargs.get("rule_hidden_dims", [128, 128])
        pos_hidden_dims = kwargs.get("pos_hidden_dims", [128, 128])
        value_hidden_dims = kwargs.get("value_hidden_dims", [256, 128, 64])

        self.rule_head = mlp(feat_dim, rule_hidden_dims, self.rule_dim, layernorm=True)
        self.pos_head = mlp(feat_dim + self.rule_dim, pos_hidden_dims, self.max_positions, layernorm=True)
        self.value_net = mlp(feat_dim, value_hidden_dims, 1, layernorm=True)

        actor_params = list(self.encoder.parameters()) + list(self.rule_head.parameters()) + list(self.pos_head.parameters())
        critic_params = self.value_net.parameters()
        self.optimizer = torch.optim.Adam([
            {"params": actor_params, "lr": lr},
            {"params": critic_params, "lr": lr},
        ])