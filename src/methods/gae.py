import torch

from .rollout import Rollout, episode_discounted_sum


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, terminated: torch.Tensor,
    dones: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95, ) -> tuple[torch.Tensor, torch.Tensor]:
    delta = rewards + gamma * next_values * (1.0 - terminated) - values
    advantages = episode_discounted_sum(delta, dones, gamma * gae_lambda)
    return advantages, advantages + values


def gae(rollout: Rollout, cfg) -> torch.Tensor:
    return compute_gae(rollout.rewards, rollout.values, rollout.next_values, rollout.terminated, rollout.dones,
                       cfg.gamma, cfg.gae_lambda)[0]
