import numpy as np
import torch
import torch.nn as nn

from .agent import ActorCritic
from .methods import METHODS
from .methods.gae import compute_gae
from .metrics import credit_metrics, failure_metrics
from .ppo import PPOConfig, minibatch_count, value_loss
from .tasks import DelayedEffect
from .verify import collect, to_rollout

SETTINGS = ("offline", "stationary", "drift")


def path_policy(task: DelayedEffect, progress: float, p_end: float = 0.95) -> np.ndarray:
    uniform = np.full((2 * (task.depth + 1), task.num_actions), 1.0 / task.num_actions)
    adv = task.oracle(uniform)[0][task.observation_ids()]
    gap = adv.max(axis=1, keepdims=True) - adv.min(axis=1, keepdims=True)
    beta = progress * np.log(p_end * (task.num_actions - 1) / (1 - p_end))
    logits = beta * adv / np.where(gap > 1e-12, gap, 1.0)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    return probs / probs.sum(axis=1, keepdims=True)


def new_critic(task: DelayedEffect, cfg: PPOConfig, seed: int) -> tuple[ActorCritic, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    agent = ActorCritic(task.observation_space, task.action_space, cfg.hidden_size)
    return agent, torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)


def fit(agent: ActorCritic, optimizer, rollout, obs: torch.Tensor, states: np.ndarray, cfg: PPOConfig, seen: int) -> None:
    if cfg.anneal_lr:
        optimizer.param_groups[0]["lr"] = (1.0 - seen / cfg.total_episodes) * cfg.learning_rate
    returns = compute_gae(rollout.rewards, rollout.values, rollout.next_values, rollout.terminated, rollout.dones,
                          cfg.gamma, cfg.critic_lambda)[1]
    states = torch.as_tensor(states)
    for _ in range(cfg.update_epochs):
        for mb in torch.randperm(len(returns)).tensor_split(minibatch_count(len(returns), cfg)):
            loss = cfg.vf_coef * value_loss(agent.get_value(obs)[states[mb]], rollout.values[mb], returns[mb], cfg)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()


def credits(rollout, configs: list[PPOConfig]) -> dict[str, np.ndarray]:
    return {c.label or c.method: METHODS[c.method](rollout, c).numpy().astype(np.float64) for c in configs}


def scores(credit, truth, groups, roles) -> dict:
    return {**credit_metrics(credit, truth, groups), **failure_metrics(credit, truth, groups, roles)}


def value_error(values: torch.Tensor, true_values: np.ndarray, states: np.ndarray) -> float:
    return float(np.sqrt(((values.numpy() - true_values)[states] ** 2).mean()))


def online(task_kwargs: dict, variants: list[dict], base: dict, seed: int, drift_episodes: float = float("inf"),
           p_end: float = 0.95) -> list[dict]:
    cfg = PPOConfig(**base)
    configs = [PPOConfig(**{**base, **v}) for v in variants]
    per = cfg.num_envs * cfg.num_episodes
    task = DelayedEffect(per, seed=seed, **task_kwargs)
    agent, optimizer = new_critic(task, cfg, seed)
    obs = torch.as_tensor(task.all_observations())
    groups, roles = task.observation_ids(), task.roles()
    rng = np.random.default_rng([seed, 1])
    rows, episodes, update = [], 0, 0
    while episodes < cfg.total_episodes:
        update += 1
        moved = min(1.0, episodes / drift_episodes)
        probs = path_policy(task, moved, p_end)
        truth, true_values = task.oracle(probs)
        with torch.no_grad():
            values = agent.get_value(obs)
        batch = collect(task, probs, rng)
        states = batch[0]
        rollout = to_rollout(probs, truth, values, *batch, obs=obs)
        step = {"update": update, "episodes": episodes + per, "moved": moved, "p_decisive": float(probs[0, 0]),
                "value_error": value_error(values, true_values, states)}
        target, g, r = truth[states, batch[1]], groups[states], roles[states]
        rows += [{"label": label, **step, **scores(credit, target, g, r)} for label, credit in credits(rollout, configs).items()]
        fit(agent, optimizer, rollout, obs, states, cfg, episodes)
        episodes += per
    return rows


def offline(task_kwargs: dict, variants: list[dict], base: dict, seed: int, sizes=(128, 1280, 10000),
            p_end: float = 0.95) -> list[dict]:
    cfg = PPOConfig(**base)
    configs = [PPOConfig(**{**base, **v}) for v in variants]
    per = cfg.num_envs * cfg.num_episodes
    updates = -(-cfg.total_episodes // per)
    rows = []
    for size in sizes:
        task = DelayedEffect(size, seed=seed, **task_kwargs)
        length = task.depth + 1
        probs = path_policy(task, 0.0, p_end)
        truth, true_values = task.oracle(probs)
        rng = np.random.default_rng([seed, 2, size])
        data = collect(task, probs, rng)
        agent, optimizer = new_critic(task, cfg, seed)
        obs = torch.as_tensor(task.all_observations())
        order = np.concatenate([rng.permutation(size) for _ in range(-(-updates * per // size))])
        for k in range(updates):
            steps = (order[k * per:(k + 1) * per, None] * length + np.arange(length)).reshape(-1)
            with torch.no_grad():
                values = agent.get_value(obs)
            fit(agent, optimizer, to_rollout(probs, truth, values, *(x[steps] for x in data), obs=obs), obs, data[0][steps], cfg, k * per)
        with torch.no_grad():
            values = agent.get_value(obs)
        chunks = [credits(to_rollout(probs, truth, values, *(x[e * length:(e + per) * length] for x in data), obs=obs), configs)
                  for e in range(0, size, per)]
        states = data[0]
        step = {"episodes": size, "p_decisive": float(probs[0, 0]), "value_error": value_error(values, true_values, states)}
        target, g, r = truth[states, data[1]], task.observation_ids()[states], task.roles()[states]
        rows += [{"label": label, **step, **scores(np.concatenate([c[label] for c in chunks]), target, g, r)} for label in chunks[0]]
    return rows
