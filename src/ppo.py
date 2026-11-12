import argparse
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import get_args

import numpy as np
import torch
import torch.nn as nn

from .agent import ActorCritic
from .methods import METHODS, Rollout
from .methods.gae import compute_gae
from .metrics import credit_metrics, run_summary
from .tasks import make_vec


@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    env_kwargs: dict = field(default_factory=dict)
    method: str = "gae"
    label: str = ""  # optional name for this method variant used to group results
    seed: int = 0
    total_timesteps: int = 200_000
    total_episodes: int = 0  # if > 0, the budget is counted in episodes instead of steps
    num_envs: int = 8
    num_steps: int = 128  # per update, each env runs at least this many steps
    num_episodes: int = 1  # and at least this many episodes then stops at the end of its episode
    gamma: float = 0.99
    gae_lambda: float = 0.95
    critic_lambda: float = 0.95  # the critic's targets are the same for every method
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    update_epochs: int = 4
    minibatch_size: int = 256  # 0 = split every batch into num_minibatches equal parts
    num_minibatches: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    norm_credit: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    hidden_size: int = 64
    device: str = "cpu"
    log_interval: int = 10
    out_dir: str | None = "runs"
    run_name: str | None = None


def _jsonable(metrics: dict) -> dict:
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in metrics.items()}


def train(cfg: PPOConfig) -> tuple[ActorCritic, list[dict]]:
    if cfg.method not in METHODS:
        raise ValueError(f"Unknown method {cfg.method!r}; available: {sorted(METHODS)}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    vec = make_vec(cfg.env_id, cfg.num_envs, cfg.seed, **cfg.env_kwargs)
    has_oracle = hasattr(vec, "oracle")
    if has_oracle and cfg.gamma != 1.0:
        raise ValueError(f"{cfg.env_id}'s oracle is undiscounted, so credit is only scored with gamma = 1 (got {cfg.gamma})")
    agent = ActorCritic(vec.observation_space, vec.action_space, cfg.hidden_size).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    run_dir = metrics_file = None
    if cfg.out_dir is not None:
        run_name = cfg.run_name or f"{cfg.env_id}_{cfg.method}_seed{cfg.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir = Path(cfg.out_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "final.json").unlink(missing_ok=True)
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        metrics_file = open(run_dir / "metrics.jsonl", "w")

    def to_tensor(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    N = cfg.num_envs
    obs = vec.reset()
    episode_returns = np.zeros(N)
    episode_lengths = np.zeros(N, dtype=int)
    recent_episodes: deque[dict] = deque(maxlen=100)
    num_episodes = 0

    history: list[dict] = []
    global_step = 0
    update = 0
    start_time = time.time()

    def progress() -> float:
        return num_episodes / cfg.total_episodes if cfg.total_episodes else global_step / cfg.total_timesteps

    while progress() < 1.0:
        update += 1
        if cfg.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - progress()) * cfg.learning_rate

        buf = {k: [] for k in ("env", "obs", "actions", "logprobs", "values", "rewards", "terminated", "dones", "timeout_values", "states")}
        active = np.ones(N, dtype=bool)
        episodes_done = np.zeros(N, dtype=int)
        tick = 0

        while active.any():
            tick += 1
            ids = np.flatnonzero(active)
            obs_t = to_tensor(obs[ids])
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t)
            action_np = action.cpu().numpy()

            states = vec.states[ids] if has_oracle else np.zeros(len(ids), dtype=np.int64)
            next_obs, rewards, terminated, truncated, final_obs, info = vec.step(ids, action_np)
            dones = terminated | truncated
            timeout_values = np.zeros(len(ids), dtype=np.float32)
            cut_off = truncated & ~terminated
            if cut_off.any():
                with torch.no_grad():
                    timeout_values[cut_off] = agent.get_value(to_tensor(final_obs[cut_off])).cpu().numpy()

            episode_returns[ids] += rewards
            episode_lengths[ids] += 1
            for j in np.flatnonzero(dones):
                i = ids[j]
                episode = {"mean_return": float(episode_returns[i]), "mean_length": float(episode_lengths[i])}
                episode.update({f"info_{k}": float(v[j]) for k, v in info.items() if not np.isnan(v[j])})
                recent_episodes.append(episode)
                num_episodes += 1
                episodes_done[i] += 1
                episode_returns[i], episode_lengths[i] = 0.0, 0
                active[i] = tick < cfg.num_steps or episodes_done[i] < cfg.num_episodes
            obs[ids] = next_obs

            buf["env"].append(torch.as_tensor(ids, device=device))
            buf["obs"].append(obs_t)
            buf["actions"].append(action)
            buf["logprobs"].append(logprob)
            buf["values"].append(value)
            buf["rewards"].append(to_tensor(rewards))
            buf["terminated"].append(to_tensor(terminated))
            buf["dones"].append(to_tensor(dones))
            buf["timeout_values"].append(to_tensor(timeout_values))
            buf["states"].append(torch.as_tensor(states, device=device))
            global_step += len(ids)

        order = torch.argsort(torch.cat(buf["env"]), stable=True)
        data = {k: torch.cat(v)[order] for k, v in buf.items()}
        next_values = torch.where(data["dones"].bool(), data["timeout_values"], torch.roll(data["values"], -1))

        oracle_credit, task_metrics = None, {}
        if has_oracle:
            with torch.no_grad():
                probs = torch.softmax(agent.actor(to_tensor(vec.all_observations())), dim=-1).cpu().numpy()
            true_credit, true_values = vec.oracle(probs)
            oracle_credit = to_tensor(true_credit)[data["states"], data["actions"]]
            task_metrics = vec.task_metrics(probs, true_values)

        rollout = Rollout(obs=data["obs"], actions=data["actions"], logprobs=data["logprobs"], rewards=data["rewards"],
                          values=data["values"], next_values=next_values, terminated=data["terminated"], dones=data["dones"],
                          oracle_credit=oracle_credit)

        raw_credit = METHODS[cfg.method](rollout, cfg)
        credit = raw_credit
        if cfg.norm_credit:
            credit = (credit - credit.mean()) / (credit.std() + 1e-8)
        returns = compute_gae(rollout.rewards, rollout.values, rollout.next_values, rollout.terminated, rollout.dones,
                              cfg.gamma, cfg.critic_lambda)[1]
        if has_oracle:
            groups = vec.observation_ids()[data["states"].cpu().numpy()]
            task_metrics.update(credit_metrics(raw_credit.cpu().numpy(), oracle_credit.cpu().numpy(), groups))

        batch_size = len(credit)
        if cfg.minibatch_size:
            num_minibatches = max(1, round(batch_size / cfg.minibatch_size))
        else:
            num_minibatches = cfg.num_minibatches
        clip_fracs = []
        for _ in range(cfg.update_epochs):
            for mb in torch.randperm(batch_size, device=device).tensor_split(num_minibatches):
                _, new_logprob, entropy, new_value = agent.get_action_and_value(rollout.obs[mb], rollout.actions[mb])
                log_ratio = new_logprob - rollout.logprobs[mb]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                pg_loss = -torch.min(
                    credit[mb] * ratio,
                    credit[mb] * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef),).mean()

                v_loss_unclipped = (new_value - returns[mb]) ** 2
                if cfg.clip_vloss:
                    v_clipped = rollout.values[mb] + (new_value - rollout.values[mb]).clamp(-cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns[mb]) ** 2).mean()
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

        var_returns = returns.var()
        explained_var = float("nan") if var_returns == 0 else (1 - (returns - rollout.values).var() / var_returns).item()
        episode_means = {
            k: float(np.mean([e[k] for e in recent_episodes if k in e])) for k in recent_episodes[-1]}
        metrics = {
            "update": update,
            "global_step": global_step,
            "episodes": num_episodes,
            **episode_means,
            **task_metrics,
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clip_frac": float(np.mean(clip_fracs)),
            "explained_variance": explained_var,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "sps": int(global_step / (time.time() - start_time)),}

        history.append(metrics)
        if metrics_file is not None:
            metrics_file.write(json.dumps(_jsonable(metrics)) + "\n")
            metrics_file.flush()
        if cfg.log_interval and (update % cfg.log_interval == 0 or progress() >= 1.0):
            shown = {k.removeprefix("info_"): v for k, v in metrics.items()
                     if k.startswith("info_") or k in ("p_decisive", "credit_corr", "spurious_frac")}
            print(
                f"update {update:>4}  step {global_step:>8}  episodes {num_episodes:>7}  "
                f"return {metrics['mean_return']:>8.2f}" + "".join(f"  {k} {v:.2f}" for k, v in shown.items())
                + f"  v_loss {metrics['value_loss']:.3f}  kl {metrics['approx_kl']:.4f}  sps {metrics['sps']}")

    vec.close()
    if run_dir is not None:
        metrics_file.close()
        (run_dir / "final.json").write_text(json.dumps(_jsonable(run_summary(history)), indent=2))
    return agent, history


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser()
    for f in fields(PPOConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        elif f.type is dict:
            parser.add_argument(flag, type=json.loads, default={}, help="JSON, e.g. '{\"distractor_steps\": 60}'")
        else:
            arg_type = f.type if isinstance(f.type, type) else get_args(f.type)[0]
            parser.add_argument(flag, type=arg_type, default=f.default)
    return PPOConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
