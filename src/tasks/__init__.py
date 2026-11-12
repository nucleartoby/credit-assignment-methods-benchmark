from .delayed_effect import DelayedEffect
from .gym_vec import GymVec

TASKS = {"DelayedEffect-v0": DelayedEffect}


def make_vec(env_id: str, num_envs: int, seed: int = 0, **kwargs):
    if env_id in TASKS:
        return TASKS[env_id](num_envs, seed=seed, **kwargs)
    return GymVec(env_id, num_envs, seed=seed, **kwargs)
