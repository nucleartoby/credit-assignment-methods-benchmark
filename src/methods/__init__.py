from .gae import gae
from .mc import mc
from .oracle import oracle
from .rollout import Rollout

METHODS = {"gae": gae, "mc": mc, "oracle": oracle}
