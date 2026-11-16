import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..key_to_door import DISTRACTOR, DOOR, KEY, MOVES


class PhasedGrid(gym.Env):
    kinds = ("key", "apple", "door")
    num_extras = 1

    def __init__(self, grid_size: int = 5, key_steps: int = 10, distractor_steps: int = 30, door_steps: int = 10,
                 num_apples: int = 3, apple_reward: float = 1.0, apple_reward_std: float = 0.0, door_reward: float = 10.0,
                 observe_key: bool = True):
        self.grid_size = grid_size
        self.phase_steps = (key_steps, distractor_steps, door_steps)
        self.num_apples = num_apples
        self.apple_reward = apple_reward
        self.apple_reward_std = apple_reward_std
        self.door_reward = door_reward
        self.observe_key = observe_key
        self.schedule = [phase for phase in self._schedule() if self.phase_steps[phase]]

        size = (1 + len(self.kinds)) * grid_size**2 + 3 + self.num_extras + 1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(size,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.apple_return = 0.0
        self.door_opened = False
        self.stage = -1
        self.round = 0
        self._begin()
        self._advance()
        return self._obs(), self._info()

    def step(self, action):
        g = self.grid_size
        dr, dc = MOVES[action]
        self.agent = (min(max(self.agent[0] + dr, 0), g - 1), min(max(self.agent[1] + dc, 0), g - 1))
        self.steps_left -= 1

        reward, terminated = 0.0, False
        kind = self.objects.get(self.agent)
        if kind == "apple":
            reward = float(self.np_random.normal(self.apple_reward, self.apple_reward_std))
            self.apple_return += reward
            del self.objects[self.agent]
            free = [c for c in self._all_cells() if c != self.agent and c not in self.objects]
            self.objects[free[self.np_random.integers(len(free))]] = "apple"
        elif kind == "door":
            reward, terminated = self._knock()
        elif kind:
            reward = float(self._take(kind))

        if not terminated and self.steps_left == 0 and not self._advance():
            terminated = True
            if DOOR not in self.schedule:
                reward += self._knock()[0]
        return self._obs(), reward, terminated, False, self._info()

    def _knock(self) -> tuple[float, bool]:
        payout = self._door()
        if payout is None:
            return 0.0, False
        self.door_opened = payout > 0
        return float(payout), True

    def _advance(self) -> bool:
        self.stage += 1
        if self.stage == len(self.schedule):
            return False
        self.phase = self.schedule[self.stage]
        self.round += self.phase == KEY
        self.steps_left = self.phase_steps[self.phase]
        kinds = self._spawn()
        picks = self.np_random.choice(self.grid_size**2, size=len(kinds) + 1, replace=False)
        cells = [divmod(int(i), self.grid_size) for i in picks]
        self.agent = cells[0]
        self.objects = dict(zip(cells[1:], kinds))
        return True

    def _all_cells(self):
        return [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]

    def _obs(self) -> np.ndarray:
        g, n = self.grid_size, self.grid_size**2
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[self.agent[0] * g + self.agent[1]] = 1.0
        for (r, c), kind in self.objects.items():
            obs[n * (1 + self.kinds.index(kind)) + r * g + c] = 1.0
        tail = n * (1 + len(self.kinds))
        obs[tail + self.phase] = 1.0
        obs[tail + 3:-1] = self._extras()
        obs[-1] = self.steps_left / self.phase_steps[self.phase]
        return obs

    def _schedule(self) -> list[int]:
        return [KEY, DISTRACTOR, DOOR]

    def _begin(self):
        self.has_key = False

    def _spawn(self) -> list[str]:
        return (["key"], ["apple"] * self.num_apples, ["door"])[self.phase]

    def _take(self, kind: str) -> float:
        self.objects.clear()
        self.has_key = True
        return 0.0

    def _door(self) -> float | None:
        return self.door_reward if self.has_key else None

    def _extras(self) -> list[float]:
        return [self.has_key and self.observe_key]

    def _info(self) -> dict:
        return {"has_key": self.has_key, "door_opened": self.door_opened, "apple_return": self.apple_return}


class CoinOrKey(PhasedGrid):
    kinds = ("key", "coin", "apple", "door")

    def __init__(self, coin_reward: float = 5.0, door_steps: int = 0, **kwargs):
        self.coin_reward = coin_reward
        super().__init__(door_steps=door_steps, **kwargs)

    def _begin(self):
        self.has_key = self.took_coin = False

    def _spawn(self):
        return ["key", "coin"] if self.phase == KEY else super()._spawn()

    def _take(self, kind):
        self.objects.clear()
        self.has_key, self.took_coin = kind == "key", kind == "coin"
        return self.coin_reward * self.took_coin

    def _info(self):
        return {**super()._info(), "took_coin": self.took_coin}


class LotteryKeys(PhasedGrid):
    kinds = ("gold", "bronze", "apple", "door")
    num_extras = 2

    def __init__(self, gold_prob: float = 0.9, bronze_prob: float = 0.45, **kwargs):
        self.open_prob = {"gold": gold_prob, "bronze": bronze_prob}
        super().__init__(**kwargs)

    def _begin(self):
        self.held = None

    def _spawn(self):
        return ["gold", "bronze"] if self.phase == KEY else super()._spawn()

    def _take(self, kind):
        self.objects.clear()
        self.held = kind
        return 0.0

    def _door(self):
        if self.held is None:
            return None
        return self.door_reward * bool(self.np_random.random() < self.open_prob[self.held])

    def _extras(self):
        return [self.observe_key and self.held == "gold", self.observe_key and self.held == "bronze"]

    def _info(self):
        best = max(self.open_prob, key=self.open_prob.get)
        return {"has_key": self.held is not None, "best_key": self.held == best, "door_opened": self.door_opened,
                "apple_return": self.apple_return}


class ManyKeys(PhasedGrid):
    num_extras = 2

    def __init__(self, num_keys: int = 2, need_all: bool = False, **kwargs):
        self.num_keys = num_keys
        self.need_all = need_all
        super().__init__(**kwargs)

    def _schedule(self):
        return [KEY, DISTRACTOR] * self.num_keys + [DOOR]

    def _begin(self):
        self.keys = 0

    def _take(self, kind):
        self.objects.clear()
        self.keys += 1
        return 0.0

    def _door(self):
        if self.need_all:
            return self.door_reward * self.num_keys if self.keys == self.num_keys else None
        return self.door_reward * self.keys if self.keys else None

    def _extras(self):
        return [self.observe_key * self.keys / self.num_keys, self.round / self.num_keys]

    def _info(self):
        return {"keys_held": self.keys / self.num_keys, "all_keys": self.keys == self.num_keys,
                "door_opened": self.door_opened, "apple_return": self.apple_return}


class Decoy(PhasedGrid):
    kinds = ("key", "charm", "apple", "door")
    num_extras = 3

    def __init__(self, unlocked_prob: float = 0.5, charm_cost: float = 1.0, observe_luck: bool = False, **kwargs):
        self.unlocked_prob = unlocked_prob
        self.charm_cost = charm_cost
        self.observe_luck = observe_luck
        super().__init__(**kwargs)

    def _begin(self):
        self.has_key = self.took_charm = False
        self.unlocked = bool(self.np_random.random() < self.unlocked_prob)

    def _spawn(self):
        return ["key", "charm"] if self.phase == KEY and self.unlocked else super()._spawn()

    def _take(self, kind):
        del self.objects[self.agent]
        if kind == "key":
            self.has_key = True
            return 0.0
        self.took_charm = True
        return -self.charm_cost

    def _door(self):
        return self.door_reward if self.has_key or self.unlocked else None

    def _extras(self):
        return [self.has_key and self.observe_key, self.took_charm, self.unlocked and self.observe_luck]

    def _info(self):
        correct = not self.took_charm if self.unlocked else self.has_key
        return {**super()._info(), "took_charm": self.took_charm, "correct": correct}


gym.register(id="CoinOrKey-v0", entry_point=CoinOrKey)
gym.register(id="LotteryKeys-v0", entry_point=LotteryKeys)
gym.register(id="ManyKeys-v0", entry_point=ManyKeys)
gym.register(id="Decoy-v0", entry_point=Decoy)
