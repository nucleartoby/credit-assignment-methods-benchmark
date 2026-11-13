import numpy as np
from gymnasium import spaces


class DelayedEffect:
    def __init__(self, num_envs: int, seed: int = 0, depth: int = 10, num_actions: int = 2, density: float = 0.0, reward_noise: float = 0.0,
                 observe_flag: bool = True, decisive: bool = True, outcome_reward: float = 1.0, distractor_reward: float = 1.0):
        self.num_envs = num_envs
        self.depth = depth
        self.num_actions = num_actions
        self.reward_noise = reward_noise
        self.observe_flag = observe_flag
        self.decisive = decisive
        self.outcome_reward = outcome_reward
        self.distractor_reward = distractor_reward

        steps = np.arange(depth + 1)
        self.paying = np.floor(steps * density + 1e-9) > np.floor((steps - 1) * density + 1e-9)
        self.paying[0] = False

        self.rng = np.random.default_rng(seed)
        self.t = np.zeros(num_envs, dtype=np.int64)
        self.flag = np.zeros(num_envs, dtype=bool)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(depth + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(num_actions)

    @property
    def states(self) -> np.ndarray:
        return 2 * self.t + self.flag

    def _obs(self, t: np.ndarray, flag: np.ndarray) -> np.ndarray:
        obs = np.zeros((len(t), self.depth + 2), dtype=np.float32)
        obs[np.arange(len(t)), t] = 1.0
        obs[:, -1] = flag & self.observe_flag
        return obs

    def all_observations(self) -> np.ndarray:
        t = np.repeat(np.arange(self.depth + 1), 2)
        return self._obs(t, np.tile([False, True], self.depth + 1))

    def observation_ids(self) -> np.ndarray:
        states = np.arange(2 * (self.depth + 1))
        return states if self.observe_flag else states - states % 2

    def roles(self) -> np.ndarray:
        t = np.repeat(np.arange(self.depth + 1), 2)
        return np.where((t == 0) & self.decisive, 0, np.where(self.paying[t], 1, 2))

    def reset(self) -> np.ndarray:
        self.t[:] = 0
        self.flag[:] = False
        return self._obs(self.t, self.flag)

    def step(self, ids: np.ndarray, actions: np.ndarray):
        t, flag = self.t[ids], self.flag[ids]
        set_flag = actions == 0 if self.decisive else self.rng.random(len(ids)) < 0.5
        flag = np.where(t == 0, set_flag, flag)
        last = t == self.depth

        rewards = self.distractor_reward * (self.paying[t] & (actions == 1)) + self.outcome_reward * (last & flag)
        if self.reward_noise:
            rewards = rewards + self.rng.normal(0.0, self.reward_noise, len(ids))

        self.t[ids] = np.where(last, 0, t + 1)
        self.flag[ids] = flag & ~last
        info = {"outcome": np.where(last, flag, np.nan)}
        return self._obs(self.t[ids], self.flag[ids]), rewards.astype(np.float32), last, np.zeros_like(last), None, info

    def oracle(self, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / probs.sum(axis=1, keepdims=True)
        q = np.zeros((2 * (self.depth + 1), self.num_actions))
        v = np.zeros(2 * (self.depth + 1))
        for t in range(self.depth, -1, -1):
            for flag in (0, 1):
                s = 2 * t + flag
                if t > 0:
                    p_flag = np.full(self.num_actions, float(flag))
                elif self.decisive:
                    p_flag = (np.arange(self.num_actions) == 0).astype(np.float64)
                else:
                    p_flag = np.full(self.num_actions, 0.5)
                if t == self.depth:
                    q[s] = self.outcome_reward * p_flag
                else:
                    q[s] = p_flag * v[2 * (t + 1) + 1] + (1.0 - p_flag) * v[2 * (t + 1)]
                if self.paying[t]:
                    q[s, 1] += self.distractor_reward
                v[s] = probs[s] @ q[s]
        return q - v[:, None], v

    def task_metrics(self, probs: np.ndarray, values: np.ndarray) -> dict:
        return {"p_decisive": float(probs[0, 0]), "expected_return": float(values[0])}

    def close(self):
        pass
