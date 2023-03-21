from enum import Enum

from .resonator_punchout import resonator_punchout
from .resonator_spectroscopy import resonator_spectroscopy


class Operation(Enum):
    resonator_spectroscopy = resonator_spectroscopy
    resonator_punchout = resonator_punchout
