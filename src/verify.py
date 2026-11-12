import argparse
import json
import sys

import numpy as np
import torch

from .methods import METHODS, Rollout
from .metrics import credit_metrics
from .ppo import PPOConfig
from .tasks import DelayedEffect

CRITICS = ("exact", "zero", "noisy")
VARIANTS = (("gae", 0.0), ("gae", 0.95), ("gae", 1.0), ("mc", None), ("oracle", None))


def random_policy(task: DelayedEffect, seed: int = 0, floor: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = (1 - floor) * rng.dirichlet(np.ones(task.num_actions), size=2 * (task.depth + 1)) + floor / task.num_actions
    if not task.observe_flag:
        probs[1::2] = probs[0::2]
    return probs


def visitation(task: DelayedEffect, probs: np.ndarray) -> np.ndarray:
    p_flag = probs[0, 0] if task.decisive else 0.5
    d = np.zeros(2 * (task.depth + 1))
    d[0] = 1.0
    d[2::2] = 1.0 - p_flag
    d[3::2] = p_flag
    return d


def critic_values(task: DelayedEffect, probs: np.ndarray, critic: str, seed: int = 0) -> np.ndarray:
    ids = task.observation_ids()
    d = visitation(task, probs)
    values = task.oracle(probs)[1]
    values = (np.bincount(ids, d * values) / np.maximum(np.bincount(ids, d), 1e-12))[ids]
    if critic == "zero":
        return np.zeros_like(values)
    if critic == "noisy":
        return values + np.random.default_rng([seed, 2]).normal(size=len(values))[ids]
    if critic != "exact":
        raise ValueError(f"Unknown critic {critic!r}; available: {CRITICS}")
    return values


def sample(task: DelayedEffect, probs: np.ndarray, values: np.ndarray, seed: int = 0) -> tuple[Rollout, np.ndarray]:
    rng = np.random.default_rng([seed, 1])
    n = task.num_envs
    ids = np.arange(n)
    task.reset()
    steps = []
    for _ in range(task.depth + 1):
        states = task.states.copy()
        actions = np.minimum((rng.random((n, 1)) > probs[states].cumsum(axis=1)).sum(axis=1), task.num_actions - 1)
        rewards, terminated = task.step(ids, actions)[1:3]
        steps.append((states, actions, rewards, terminated))
    states, actions, rewards, terminated = (np.stack(x, axis=1).reshape(-1) for x in zip(*steps))

    def f(x):
        return torch.as_tensor(x, dtype=torch.float32)

    v = f(values[states])
    dones = f(terminated)
    rollout = Rollout(obs=torch.zeros(len(states), 0), actions=torch.as_tensor(actions), logprobs=f(np.log(probs[states, actions])),
                      rewards=f(rewards), values=v, next_values=torch.where(dones.bool(), torch.zeros_like(v), torch.roll(v, -1)),
                      terminated=dones, dones=dones, oracle_credit=f(task.oracle(probs)[0][states, actions]))
    return rollout, states


def bias_report(credit, truth, states, actions, num_actions: int, min_count: int = 100, z_max: float = 5.0,
                tol: float = 1e-5) -> dict:
    err = np.asarray(credit, np.float64) - np.asarray(truth, np.float64)
    cells = states * num_actions + actions

    def stats(keys):
        n = np.bincount(keys)
        mean = np.bincount(keys, err) / np.maximum(n, 1)
        var = np.maximum(np.bincount(keys, err**2) / np.maximum(n, 1) - mean**2, 0.0)
        return n, mean, var

    n_c, m_c, v_c = stats(cells)
    n_s, m_s, v_s = stats(states)
    owner = np.arange(len(n_c)) // num_actions
    bias = m_c - m_s[owner]
    se = np.sqrt(v_c / np.maximum(n_c, 1) + v_s[owner] / np.maximum(n_s[owner], 1))
    z = np.where(np.abs(bias) < tol, 0.0, np.abs(bias) / np.maximum(se, 1e-12))
    tested = n_c >= min_count
    return {"bias_rms": float(np.sqrt((n_c * bias**2).sum() / n_c.sum())),
            "max_abs_bias": float(np.abs(bias[tested]).max()),
            "max_z": float(z[tested].max()),
            "noise": float(np.sqrt((n_c * v_c).sum() / n_c.sum())),
            "unbiased": bool((z[tested] < z_max).all())}


def guaranteed(task: DelayedEffect, method: str, lam: float | None, critic: str) -> bool | None:
    if method in ("mc", "oracle"):
        return True
    if method == "gae":
        return lam == 1.0 or task.depth == 0 or not task.decisive or (critic == "exact" and task.observe_flag)
    return None


def label(method: str, lam: float | None) -> str:
    return method if lam is None else f"{method}_{lam:g}"


def verify(task_kwargs: dict, critic: str = "exact", variants=VARIANTS, num_episodes: int = 50_000, policy_seed: int = 0,
           seed: int = 0, extra: dict | None = None) -> list[dict]:
    task = DelayedEffect(num_episodes, seed=seed, **task_kwargs)
    probs = random_policy(task, policy_seed)
    rollout, states = sample(task, probs, critic_values(task, probs, critic, seed), seed)
    actions = rollout.actions.numpy()
    truth = rollout.oracle_credit.numpy()
    groups = task.observation_ids()[states]
    methods = {**METHODS, **(extra or {})}
    rows = []
    for method, lam in variants:
        cfg = PPOConfig(env_id="DelayedEffect-v0", method=method, gamma=1.0, gae_lambda=0.95 if lam is None else lam)
        credit = methods[method](rollout, cfg).numpy()
        rows.append({"method": label(method, lam), "critic": critic, "guaranteed": guaranteed(task, method, lam, critic),
                     **bias_report(credit, truth, states, actions, task.num_actions), **credit_metrics(credit, truth, groups)})
    return rows


def verdict(row: dict) -> str:
    if row["guaranteed"] and not row["unbiased"]:
        return "FAIL"
    if row["guaranteed"]:
        return "pass"
    return "unbiased" if row["unbiased"] else "biased"


def parse_variant(text: str) -> tuple[str, float | None]:
    method, _, lam = text.partition(":")
    return method, float(lam) if lam else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=json.loads, default={"depth": 5}, help="DelayedEffect kwargs as JSON")
    parser.add_argument("--critics", nargs="+", default=list(CRITICS), choices=CRITICS)
    parser.add_argument("--methods", nargs="+", type=parse_variant, default=list(VARIANTS), help="e.g. gae:0.95 mc")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [r for c in args.critics for r in verify(args.task, c, args.methods, args.episodes, args.policy_seed, args.seed)]
    cols = ("bias_rms", "max_abs_bias", "max_z", "noise", "credit_corr", "credit_slope")
    print(f"{'method':<10} {'critic':<7}" + "".join(f"{c:>14}" for c in cols) + f"{'verdict':>10}")
    for r in rows:
        print(f"{r['method']:<10} {r['critic']:<7}" + "".join(f"{r[c]:>14.4f}" for c in cols) + f"{verdict(r):>10}")
    return int(any(verdict(r) == "FAIL" for r in rows))


if __name__ == "__main__":
    sys.exit(main())
