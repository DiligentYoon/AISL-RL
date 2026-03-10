import math
import torch
import torch.nn as nn

from typing import Optional
from torch.distributions import Normal
from lib.utils.Running_mean_std import RunningMeanStd
from lib.model.model import Model
from lib.model.MLP import SharedActor

class SingleTokenCrossAttention(Model):
    def __init__(self, token_dim: int, attn_dim: int):
        super().__init__()
        self.token_dim = token_dim
        self.attn_dim = attn_dim

        self.q_proj = nn.Linear(token_dim, attn_dim)
        self.k_proj = nn.Linear(token_dim, attn_dim)
        self.v_proj = nn.Linear(token_dim, attn_dim)

        self.out_proj = nn.Linear(attn_dim, token_dim)

    def forward(self, query_token: torch.Tensor, key_token: torch.Tensor, value_token: torch.Tensor):
        """
        query_token: [B, D]
        key_token  : [B, D]
        value_token: [B, D]

        returns:
            cooperative feature h: [B, D]
            attention weight alpha: [B, 1]
        """
        q = self.q_proj(query_token)   # [B, A]
        k = self.k_proj(key_token)     # [B, A]
        v = self.v_proj(value_token)   # [B, A]

        # Single-token scaled dot-product attention
        score = torch.sum(q * k, dim=-1, keepdim=True) / math.sqrt(self.attn_dim)  # [B, 1]
        alpha = torch.sigmoid(score)  # [B, 1]

        attended = alpha * v          # [B, A]
        h = self.out_proj(attended)   # [B, D]

        return h, alpha


class AttentionActor(SharedActor):
    def __init__(self,
                 possible_agents: list[str],
                 num_observations: dict[str, int],
                 num_actions: dict[str, int],
                 encoder_hidden_dim: int,
                 attn_hidden_dim: int,
                 RMA_hidden_dim: int,
                 min_log_std: float,
                 max_log_std: float,
                 squash: bool,
                 device: torch.device):
        super().__init__(possible_agents=possible_agents,
                         num_observations=num_observations,
                         num_actions=num_actions,
                         encoder_hidden_dim=encoder_hidden_dim,
                         RMA_hidden_dim=RMA_hidden_dim,
                         min_log_std=min_log_std,
                         max_log_std=max_log_std,
                         squash=squash,
                         device=device)

        # Optional RMA projection to token dimension
        if self.is_rma:
            self.rma_proj = nn.ModuleDict()
            self.rma_proj["arm"] = nn.Sequential(
                nn.Linear(RMA_hidden_dim, encoder_hidden_dim),
                nn.ELU()
            )
            self.rma_proj["leg"] = nn.Sequential(
                nn.Linear(RMA_hidden_dim, encoder_hidden_dim),
                nn.ELU()
            )

        # Agent-wise cross attention
        self.shared_backbone = nn.ModuleDict()
        self.shared_backbone["arm"] = SingleTokenCrossAttention(
            token_dim=encoder_hidden_dim,
            attn_dim=attn_hidden_dim
        )
        self.shared_backbone["leg"] = SingleTokenCrossAttention(
            token_dim=encoder_hidden_dim,
            attn_dim=attn_hidden_dim
        )

        # Final policy heads
        # x_arm = [z_arm, h_arm], x_leg = [z_leg, h_leg]
        self.head = nn.ModuleDict()
        self.head["arm"] = nn.Sequential(
            nn.Linear(2 * encoder_hidden_dim, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions["arm"])
        )
        self.head["leg"] = nn.Sequential(
            nn.Linear(2 * encoder_hidden_dim, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions["leg"])
        )

        self.init_weights()
        self.init_biases(val=0)

    def forward(self,
                observations: torch.Tensor | dict[str, torch.Tensor],
                shared_infos: Optional[torch.Tensor],
                taken_actions: torch.Tensor | dict[str, torch.Tensor] | None,
                deterministic: bool = False,
                update_rms: bool = False):

        eps = 1e-6

        # 1. Standardize local observations
        obs_arm = self.actor_standardizer["arm"].standardize(observations["arm"], update=update_rms)
        obs_leg = self.actor_standardizer["leg"].standardize(observations["leg"], update=update_rms)

        # 2. Local tokenization
        z_arm = self.encoder["arm"](obs_arm)   # [B, D]
        z_leg = self.encoder["leg"](obs_leg)   # [B, D]

        # Optional: inject RMA info into local tokens before attention
        if self.is_rma and shared_infos is not None:
            z_arm = z_arm + self.rma_proj["arm"](shared_infos)
            z_leg = z_leg + self.rma_proj["leg"](shared_infos)

        # 3. Agent-wise single-token cross-attention
        h_arm, _ = self.shared_backbone["arm"](query_token=z_arm,
                                               key_token=z_leg,
                                               value_token=z_leg)
        h_leg, _ = self.shared_backbone["leg"](query_token=z_leg,
                                               key_token=z_arm,
                                               value_token=z_arm)

        # 4. Final input
        x_arm = torch.cat([z_arm, h_arm], dim=-1)
        x_leg = torch.cat([z_leg, h_leg], dim=-1)

        # 5. Mean actions
        mean_action = {}
        mean_action["arm"] = self.head["arm"](x_arm)
        mean_action["leg"] = self.head["leg"](x_leg)

        # 6. Log std
        log_std = {}
        log_std["arm"] = torch.clamp(
            self.log_std_parameter["arm"], self.min_log_std, self.max_log_std
        )
        log_std["leg"] = torch.clamp(
            self.log_std_parameter["leg"], self.min_log_std, self.max_log_std
        )

        # 7. Sample / log-prob / entropy
        actions = {}
        log_probs = {}
        entropies = {}

        for uid in self.possible_agents:
            action_distribution = Normal(mean_action[uid], log_std[uid].exp())

            if deterministic:
                raw_actions = mean_action[uid]
            else:
                raw_actions = action_distribution.rsample()

            if self.squash:
                action = torch.tanh(raw_actions)
                if taken_actions is not None:
                    taken_actions[uid] = torch.clip(taken_actions[uid], -1.0 + eps, 1.0 - eps)
                    raw_taken_actions = torch.atanh(taken_actions[uid])
                    log_prob = (
                        action_distribution.log_prob(raw_taken_actions)
                        - torch.log(1 - taken_actions[uid].pow(2) + eps)
                    )
                else:
                    log_prob = (
                        action_distribution.log_prob(raw_actions)
                        - torch.log(1 - action.pow(2) + eps)
                    )
            else:
                action = raw_actions
                if taken_actions is not None:
                    log_prob = action_distribution.log_prob(taken_actions[uid])
                else:
                    log_prob = action_distribution.log_prob(action)

            log_prob = log_prob.sum(dim=-1)
            entropy = action_distribution.entropy().mean()

            actions[uid] = action
            log_probs[uid] = log_prob
            entropies[uid] = entropy

        return actions, log_probs, entropies