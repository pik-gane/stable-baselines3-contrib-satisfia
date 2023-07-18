from typing import Literal

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
    """Iterated Prisoner's Dilemma environment.
    Max return is 3 x T against GrimTrigger or TitForTat.
    Against GrimTrigger, min return is 5 (defect once, then cooperate forever).
    Against TitForTat, min return is 5 + T-1 (defect every time).
    Against other opponents, min and max returns are more complicated."""

    # parameters:
    nb_rounds = None
    """number of rounds to play"""
    opponent = None
    # state:
    t = None
    """The current timestep."""
    history = None
    """The history of actions taken by the agents."""

    def __init__(
        self,
        nb_rounds: int = 10,
        opponent: Literal["TitForTat", "STFT", "GTFT", "TFTT", "GrimTrigger", "Pavlov"] = "TitForTat",
    ):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(5)
        self.nb_rounds = nb_rounds
        self.opponent = opponent
        if opponent not in ["TitForTat", "STFT", "GTFT", "TFTT", "GrimTrigger", "Pavlov"]:
            raise NotImplementedError(f"Unknown opponent: {opponent}")

    def step(self, action):
        pair = (action, self.opponent_action())
        self.history.append(pair)
        self.t += 1
        r = {(0, 0): P, (0, 1): T, (1, 0): S, (1, 1): R}[pair] / self.nb_rounds
        return pair[0] + 2 * pair[1], r, self.t == self.nb_rounds, False, {"history": self.history.copy(), "time": self.t}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.history = []
        return 4, {}

    def opponent_action(self):
        """https://plato.stanford.edu/entries/prisoner-dilemma/strategy-table.html"""
        if self.opponent == "TitForTat":
            if self.t == 0:
                # start with cooperation:
                return 1
            else:
                # do what the opponent did last time:
                return self.history[-1][0]

        elif self.opponent == "STFT":
            if self.t == 0:
                # start with defection (!):
                return 0
            else:
                # do what the opponent did last time:
                return self.history[-1][0]

        if self.opponent == "GTFT":
            if self.t == 0:
                # start with cooperation:
                return 1
            elif self.history[-1][0] == 1:
                # cooperate after cooperation:
                return 1
            else:
                # rarely cooperate after defection:
                return 1 if self.np_random.rand() < min(1 - (T - R) / (R - S), (R - P) / (T - P)) else 0

        elif self.opponent == "TFTT":
            if self.t < 2:
                # start with cooperation:
                return 1
            elif self.history[-1][0] == 0 and self.history[-2][0] == 0:
                # defect after two defections:
                return 0
            else:
                return 1

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

        if self.opponent == "Pavlov":
            if self.t == 0:
                # start with cooperation:
                return 1
            elif self.history[-1][1] == self.history[-1][0]:
                # cooperate when both did the same:
                return 1
            else:
                return 0

        else:
            raise NotImplementedError(f"Unknown opponent type: {self.opponent}")
