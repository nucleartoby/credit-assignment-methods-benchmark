import argparse
import random
import time
from collections import deque
from dataclasses import dataclass, fields

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from .agent import ActorCritic
from .gae import compute_gae


@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    seed: int = 0
    total_timesteps: int = 200_000
    num_envs: int = 8
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    norm_adv: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    hidden_size: int = 64
    device: str = "cpu"
    log_interval: int = 10


def train(cfg: PPOConfig) -> tuple[ActorCritic, list[dict]]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    envs = [gym.make(cfg.env_id) for _ in range(cfg.num_envs)]
    obs_space, action_space = envs[0].observation_space, envs[0].action_space
    agent = ActorCritic(obs_space, action_space, cfg.hidden_size).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    T, N = cfg.num_steps, cfg.num_envs
    batch_size = T * N
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = cfg.total_timesteps // batch_size
    action_shape = action_space.shape if agent.continuous else ()

    obs_buf = torch.zeros((T, N, *obs_space.shape), device=device)
    actions_buf = torch.zeros((T, N, *action_shape), device=device)
    logprobs_buf = torch.zeros((T, N), device=device)
    rewards_buf = torch.zeros((T, N), device=device)
    values_buf = torch.zeros((T, N), device=device)
    terminated_buf = torch.zeros((T, N), device=device)
    dones_buf = torch.zeros((T, N), device=device)
    timeout_values_buf = torch.zeros((T, N), device=device)

    def to_tensor(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    obs = np.stack([env.reset(seed=cfg.seed + i)[0] for i, env in enumerate(envs)])
    episode_returns = np.zeros(N)
    episode_lengths = np.zeros(N, dtype=int)
    recent_returns: deque[float] = deque(maxlen=100)
    recent_lengths: deque[int] = deque(maxlen=100)

    history: list[dict] = []
    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        timeout_values_buf.zero_()
        for t in range(T):
            obs_t = to_tensor(obs)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t)
            obs_buf[t], actions_buf[t], logprobs_buf[t], values_buf[t] = obs_t, action, logprob, value

            action_np = action.cpu().numpy()
            next_obs = np.empty_like(obs)
            for i, env in enumerate(envs):
                if agent.continuous:
                    env_action = np.clip(action_np[i], action_space.low, action_space.high)
                else:
                    env_action = int(action_np[i])
                o, reward, terminated, truncated, _ = env.step(env_action)
                episode_returns[i] += reward
                episode_lengths[i] += 1
                rewards_buf[t, i] = float(reward)
                terminated_buf[t, i] = float(terminated)
                dones_buf[t, i] = float(terminated or truncated)
                if terminated or truncated:
                    if not terminated:
                        with torch.no_grad():
                            timeout_values_buf[t, i] = agent.get_value(to_tensor(o[None]))[0]
                    recent_returns.append(float(episode_returns[i]))
                    recent_lengths.append(int(episode_lengths[i]))
                    episode_returns[i], episode_lengths[i] = 0.0, 0
                    o, _ = env.reset()
                next_obs[i] = o
            obs = next_obs
            global_step += N

        with torch.no_grad():
            last_value = agent.get_value(to_tensor(obs))
        next_values = torch.cat([values_buf[1:], last_value[None]])
        timed_out = (dones_buf - terminated_buf).bool()
        next_values[timed_out] = timeout_values_buf[timed_out]
        advantages, returns = compute_gae(
            rewards_buf, values_buf, next_values, terminated_buf, dones_buf, cfg.gamma, cfg.gae_lambda)

        b_obs = obs_buf.reshape(batch_size, *obs_space.shape)
        b_actions = actions_buf.reshape(batch_size, *action_shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        clip_fracs = []
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb = perm[start : start + minibatch_size]
                mb_actions = b_actions[mb] if agent.continuous else b_actions[mb].long()
                _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[mb], mb_actions)
                log_ratio = new_logprob - b_logprobs[mb]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = -torch.min(
                    mb_adv * ratio,
                    mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef),).mean()

                v_loss_unclipped = (new_value - b_returns[mb]) ** 2
                if cfg.clip_vloss:
                    v_clipped = b_values[mb] + (new_value - b_values[mb]).clamp(-cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * v_loss_unclipped.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        var_returns = b_returns.var()
        explained_var = float("nan") if var_returns == 0 else (1 - (b_returns - b_values).var() / var_returns).item()
        metrics = {
            "update": update,
            "global_step": global_step,
            "mean_return": float(np.mean(recent_returns)) if recent_returns else float("nan"),
            "mean_length": float(np.mean(recent_lengths)) if recent_lengths else float("nan"),
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clip_frac": float(np.mean(clip_fracs)),
            "explained_variance": explained_var,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "sps": int(global_step / (time.time() - start_time)),}
        
        history.append(metrics)
        if cfg.log_interval and (update % cfg.log_interval == 0 or update == num_updates):
            print(
                f"update {update:>4}/{num_updates}  step {global_step:>8}  "
                f"return {metrics['mean_return']:>8.2f}  v_loss {metrics['value_loss']:.3f}  "
                f"kl {metrics['approx_kl']:.4f}  ev {explained_var:.2f}  sps {metrics['sps']}")

    for env in envs:
        env.close()
    return agent, history


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(PPOConfig):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            arg_type = float if f.default is None else type(f.default)
            parser.add_argument(flag, type=arg_type, default=f.default)
    return PPOConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
