import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def mlp(in_dim: int, hidden_size: int, out_dim: int, out_std: float) -> nn.Sequential:
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden_size)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_size, hidden_size)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_size, out_dim), std=out_std),)


class ActorCritic(nn.Module):
    def __init__(self, obs_space: gym.Space, action_space: gym.Space, hidden_size: int = 64):
        super().__init__()
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"Only Box observation spaces are supported, got {obs_space}")
        obs_dim = int(np.prod(obs_space.shape))

        if isinstance(action_space, gym.spaces.Discrete):
            self.continuous = False
            action_dim = int(action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            self.continuous = True
            action_dim = int(np.prod(action_space.shape))
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            raise TypeError(f"Only Discrete and Box action spaces are supported, got {action_space}")

        self.actor = mlp(obs_dim, hidden_size, action_dim, out_std=0.01)
        self.critic = mlp(obs_dim, hidden_size, 1, out_std=1.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs.flatten(1)).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        obs = obs.flatten(1)
        out = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        if self.continuous:
            dist = Normal(out, self.log_std.exp().expand_as(out))
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value
        dist = Categorical(logits=out)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
