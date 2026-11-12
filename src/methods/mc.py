import torch

from .rollout import Rollout, episode_discounted_sum


def mc(rollout: Rollout, cfg) -> torch.Tensor:
    cut_off = rollout.dones * (1.0 - rollout.terminated)
    rewards = rollout.rewards + cfg.gamma * rollout.next_values * cut_off
    return episode_discounted_sum(rewards, rollout.dones, cfg.gamma)
