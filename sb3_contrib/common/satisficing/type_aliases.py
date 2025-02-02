from typing import Dict, NamedTuple

import torch as th


class SatisficingReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    lambda_: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    next_lambda: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class SatisficingDictReplayBufferSamples(NamedTuple):
    observations: Dict[str, th.Tensor]
    lambda_: th.Tensor
    actions: th.Tensor
    next_observations: Dict[str, th.Tensor]
    next_lambda: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
