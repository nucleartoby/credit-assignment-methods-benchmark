import torch

from .rollout import Rollout


def oracle(rollout: Rollout, cfg) -> torch.Tensor:
    if rollout.oracle_credit is None:
        raise ValueError(f"{cfg.env_id} has no oracle")
    return rollout.oracle_credit
