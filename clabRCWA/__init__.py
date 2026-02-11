"""
RCWA - Rigorous Coupled-Wave Analysis

A modular PyTorch-based implementation of RCWA for simulating 
electromagnetic wave propagation in periodic structures.
"""

from .rcwa import RCWA
from .environment import Environment
from .incident import Incident
from . import geometry
from . import utils

__version__ = "1.0.0"

__all__ = [
    'RCWA',
    'Environment',
    'Incident',
    'geometry',
    'utils'
]
