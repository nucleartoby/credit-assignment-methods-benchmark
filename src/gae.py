import torch


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, terminated: torch.Tensor,
    dones: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95, ) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - terminated[t]) - values[t]
        last_advantage = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values
