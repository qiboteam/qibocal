from enum import Enum

from .allxy.allxy import allxy
from .allxy.allxy_drag_pulse_tuning import allxy_drag_pulse_tuning
from .allxy.drag_pulse_tuning import drag_pulse_tuning
from .classification import single_shot_classification
from .flux_depedence.qubit_flux_dependence import qubit_flux
from .flux_depedence.resonator_flux_dependence import resonator_flux
from .qubit_spectroscopy import qubit_spectroscopy
from .rabi.amplitude import rabi_amplitude
from .rabi.length import rabi_length
from .ramsey import ramsey
from .resonator_punchout import resonator_punchout
from .resonator_spectroscopy import resonator_spectroscopy
from .spin_echo import spin_echo
from .t1 import t1


class Operation(Enum):
    resonator_spectroscopy = resonator_spectroscopy
    resonator_punchout = resonator_punchout
    resonator_flux = resonator_flux
    qubit_spectroscopy = qubit_spectroscopy
    qubit_flux = qubit_flux
    rabi_amplitude = rabi_amplitude
    rabi_length = rabi_length
    ramsey = ramsey
    t1 = t1
    single_shot_classification = single_shot_classification
    spin_echo = spin_echo
    allxy = allxy
    allxy_drag_pulse_tuning = allxy_drag_pulse_tuning
    drag_pulse_tuning = drag_pulse_tuning
