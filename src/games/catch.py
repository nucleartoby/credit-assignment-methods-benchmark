import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DelayedCatch(gym.Env):
    def __init__(self, rows: int = 5, cols: int = 5, num_rounds: int = 40, payout_every: int = 1, observe_score: bool = True):
        self.rows = rows
        self.cols = cols
        self.num_rounds = num_rounds
        self.payout_every = payout_every
        self.observe_score = observe_score
        self.observation_space = spaces.Box(0.0, 1.0, shape=(rows * cols + cols + 3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.paddle = self.cols // 2
        self.round = self.catches = self.unpaid = 0
        self.ball = (0, int(self.np_random.integers(self.cols)))
        return self._obs(), self._info()

    def step(self, action):
        self.paddle = min(max(self.paddle + int(action) - 1, 0), self.cols - 1)
        self.ball = (self.ball[0] + 1, self.ball[1])

        reward, terminated = 0.0, False
        if self.ball[0] == self.rows - 1:
            caught = int(self.ball[1] == self.paddle)
            self.catches += caught
            self.unpaid += caught
            self.round += 1
            terminated = self.round == self.num_rounds
            if terminated or self.round % self.payout_every == 0:
                reward, self.unpaid = float(self.unpaid), 0
            self.ball = (0, int(self.np_random.integers(self.cols)))
        return self._obs(), reward, terminated, False, self._info()

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[self.ball[0] * self.cols + self.ball[1]] = 1.0
        obs[self.rows * self.cols + self.paddle] = 1.0
        obs[-3] = self.observe_score * self.unpaid / self.payout_every
        obs[-2] = (self.round % self.payout_every) / self.payout_every
        obs[-1] = self.round / self.num_rounds
        return obs

    def _info(self) -> dict:
        return {"catch_rate": self.catches / max(self.round, 1)}


gym.register(id="DelayedCatch-v0", entry_point=DelayedCatch)
