import os

from sb3_contrib.ar_dqn import ARDQN
from sb3_contrib.ar_q_learning import ARQLearning
from sb3_contrib.ars import ARS
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.q_learning import QLearning
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib.tqc import TQC
from sb3_contrib.trpo import TRPO
from sb3_contrib.lra_dqn import LRADQN

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

__all__ = ["ARS", "MaskablePPO", "RecurrentPPO", "QRDQN", "TQC", "TRPO", "ARDQN", "QLearning", "ARQLearning", "LRADQN"]
