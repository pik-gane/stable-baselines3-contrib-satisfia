import numpy as np
from typing import Literal

from gymnasium import Env, spaces

class PublicGood(Env):
    """A public good game with linear benefits and quadratic individual costs shared proportionally.
    action = quantity q1 of public good to contribute by the agent (=player 1).
    reward = Q - q1² / 2
    where Q = q1 + q2 + ... + qn is the total quantity of public good contributed by all agents.
    In Nash equilibrium, q1 = q2 = ... = qn = 1.
    The socially optimal solution is q1 = q2 = ... = qn = n.
    Opponents (=players 2...n) play the LinC strategy from Heitzig et al., PNAS, 2011,
    but make random mistakes and round their actions to the nearest integer.
    """

    # parameters:
    nb_rounds = None
    """number of rounds to play"""
    n_players = None
    """number of players"""
    alpha = None
    """compensation factor for LinC, >1"""
    sigma = None
    """stddev of random mistakes in LinC, >0"""

    # state:
    t = None
    """The current timestep."""
    last_actions = None
    """The last actions"""
    liabilities = None
    """The liabilities according to LinC"""

    def __init__(
        self,
        nb_rounds: int = 10,
        n_players: int = 10,
        alpha: float = 2,
        sigma: float = 1,
    ):
        super().__init__()
        self.action_space = spaces.Discrete(2*n_players + 1)
        self.observation_space = spaces.Box(low=0, high=2*n_players + 1, shape=(n_players,), dtype=np.float32)
        self.nb_rounds = nb_rounds
        self.n_players = n_players
        self.alpha = alpha
        self.sigma = sigma

    def step(self, own_action):
        self.last_actions = actions = self.calc_actions(own_action)
        self.t += 1
        r = (actions.sum() - own_action**2 / 2) / self.nb_rounds
        return tuple(actions), r, self.t == self.nb_rounds, False, {"liabilities": self.liabilities.copy(), "time": self.t}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        # start by assuming all players have contributed the socially optimal amount so far:
        self.liabilities = np.repeat(self.n_players, self.n_players)
        self.last_actions = self.liabilities.copy()
        return tuple(self.last_actions), {}

    def calc_actions(self, own_action):
        actions = np.zeros(self.n_players)
        actions[0] = own_action
        shortfalls = np.maximum(0, self.liabilities - self.last_actions)
        avg_shortfall = shortfalls.sum() / self.n_players
        l = self.liabilities = self.n_players + self.alpha * (shortfalls - avg_shortfall) 
        actions[1:] = np.minimum(np.maximum(0, np.round(l[1:] + np.random.normal(0, self.sigma, self.n_players - 1))), 2*self.n_players)            
        return actions