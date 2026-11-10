import random

import numpy as np
import torch

from refusal_direction_replication.runtime import seed_everything


def test_seed_everything_repeats_all_rngs() -> None:
    seed_everything(17)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(17)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second
