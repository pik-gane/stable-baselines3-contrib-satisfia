from typing import Any, Callable, Dict, List, NamedTuple, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th


class ExpendedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    extras: Dict[str, th.Tensor]
