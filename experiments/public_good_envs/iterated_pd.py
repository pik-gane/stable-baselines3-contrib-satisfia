from __future__ import annotations

from gymnasium import Env, spaces

# payoffs:
R = 3
"""Reward for mutual cooperation."""
S = 0
"""Sucker's payoff."""
T = 5
"""Temptation to defect."""
P = 1
"""Punishment for mutual defection."""


class IteratedPD(Env):
    """Iterated Prisoner's Dilemma environment."""

    # parameters:
    T = None
    """number of rounds to play"""
    opponent = None
    # state:
    t = None
    """The current timestep."""
    history = None
    """The history of actions taken by the agents."""

    def __init__(self, T, opponent):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(4)
        self.T = T
        self.opponent = opponent

    def step(
        self, action
    ):
        
        pair = (action, self.opponent_action())
        self.history.append(pair)
        self.t += 1
        r = {(0,0): P, (0,1): T, (1,0): S, (1,1): R}[pair]
        return pair[0] + 2*pair[1], r, self.t == self.T, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        self.t = 0
        self.history = []
        return None, {}
    
    def opponent_action(self):
        if self.opponent == "TitForTat":
            if self.t == 0:
                # start with cooperation:
                return 1
            else:
                # do what the opponent did last time:
                return self.history[-1][0]
        elif self.opponent == "GrimTrigger":
            if self.t == 0:
                # start with cooperation:
                return 1
            elif self.history[-1][1] == 0:
                # eternal punishment gets continued:
                return 0
            elif self.history[-1][0] == 0:
                # eternal punishment gets triggered:
                return 0
            else:
                # continue cooperation:
                return 1
        else:
            raise NotImplementedError(f"Unknown opponent type: {self.opponent}")