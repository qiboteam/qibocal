from enum import Enum

from .qubit_spectroscopy import qubit_spectroscopy
from .rabi import rabi_amplitude
from .ramsey import ramsey
from .resonator_punchout import resonator_punchout
from .resonator_spectroscopy import resonator_spectroscopy
from .t1 import t1


class Operation(Enum):
    resonator_spectroscopy = resonator_spectroscopy
    resonator_punchout = resonator_punchout
    qubit_spectroscopy = qubit_spectroscopy
    rabi_amplitude = rabi_amplitude
    ramsey = ramsey
    t1 = t1
