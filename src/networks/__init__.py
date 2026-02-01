"""Neural network architectures для DQN системы."""

from .q_network import QNetwork
from .dueling_network import DuelingNetwork  
from .noisy_linear import NoisyLinear
from .categorical_network import CategoricalNetwork

__all__ = [
    "QNetwork",
    "DuelingNetwork",
    "NoisyLinear", 
    "CategoricalNetwork",
]