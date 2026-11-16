import gymnasium as gym


class DelayedPayout(gym.Wrapper):
    def __init__(self, env: gym.Env, payout_every: int = 0):
        super().__init__(env)
        self.payout_every = payout_every

    def reset(self, **kwargs):
        self.unpaid, self.t = 0.0, 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.unpaid += float(reward)
        self.t += 1
        done = terminated or truncated
        if done or (self.payout_every and self.t % self.payout_every == 0):
            reward, self.unpaid = self.unpaid, 0.0
        else:
            reward = 0.0
        return obs, reward, done, False, info


def make_delayed(base_id: str = "CartPole-v1", payout_every: int = 0, **kwargs) -> gym.Env:
    return DelayedPayout(gym.make(base_id, **kwargs), payout_every)


gym.register(id="DelayedPayout-v0", entry_point=make_delayed)
