from dataclasses import dataclass

import torch


@dataclass
class Rollout:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor  # V(s_t)
    next_values: torch.Tensor  # V(s_{t+1}) 0 after a terminal state
    terminated: torch.Tensor
    dones: torch.Tensor
    oracle_credit: torch.Tensor | None = None  # ground truth only the oracle method may read it


def episode_discounted_sum(x: torch.Tensor, dones: torch.Tensor, discount: float) -> torch.Tensor:
    ends = torch.nonzero(dones).squeeze(-1)
    starts = torch.cat([ends.new_zeros(1), ends[:-1] + 1])
    lengths = ends - starts + 1
    episode = torch.repeat_interleave(torch.arange(len(ends), device=x.device), lengths)
    step = torch.arange(len(x), device=x.device) - starts[episode]
    grid = x.new_zeros(int(lengths.max()), len(ends))
    grid[step, episode] = x
    for t in reversed(range(grid.shape[0] - 1)):
        grid[t] += discount * grid[t + 1]
    return grid[step, episode]
