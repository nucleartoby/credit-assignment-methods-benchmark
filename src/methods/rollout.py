from dataclasses import dataclass

import torch


@dataclass
class Rollout:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor  # V(s_t)
    next_values: torch.Tensor  # V(s_{t+1}); 0 after a terminal state
    terminated: torch.Tensor
    dones: torch.Tensor
