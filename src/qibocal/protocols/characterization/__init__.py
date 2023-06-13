from enum import Enum

from .allxy.allxy import allxy
from .allxy.allxy_drag_pulse_tuning import allxy_drag_pulse_tuning
from .allxy.allxy_unrolling import allxy_unrolling
from .allxy.drag_pulse_tuning import drag_pulse_tuning
from .classification import single_shot_classification
from .coherence.spin_echo import spin_echo
from .coherence.t1 import t1
from .coherence.t2 import t2
from .dispersive_shift import dispersive_shift
from .flipping import flipping
from .flux_depedence.qubit_flux_dependence import qubit_flux
from .flux_depedence.resonator_flux_dependence import resonator_flux
from .qubit_spectroscopy import qubit_spectroscopy
from .rabi.amplitude import rabi_amplitude
from .rabi.length import rabi_length
from .ramsey import ramsey
from .ramsey_unrolling import ramsey_unrolling
from .randomized_benchmarking.standard_rb import standard_rb
from .RB.std_rb import StdRB
from .resonator_punchout import resonator_punchout
from .resonator_punchout_attenuation import resonator_punchout_attenuation
from .resonator_spectroscopy import resonator_spectroscopy
from .resonator_spectroscopy_attenuation import resonator_spectroscopy_attenuation


class Operation(Enum):
    resonator_spectroscopy = resonator_spectroscopy
    resonator_spectroscopy_attenuation = resonator_spectroscopy_attenuation
    resonator_punchout = resonator_punchout
    resonator_punchout_attenuation = resonator_punchout_attenuation
    resonator_flux = resonator_flux
    qubit_spectroscopy = qubit_spectroscopy
    qubit_flux = qubit_flux
    rabi_amplitude = rabi_amplitude
    rabi_length = rabi_length
    ramsey = ramsey
    ramsey_unrolling = ramsey_unrolling
    t1 = t1
    t2 = t2
    allxy = allxy
    allxy_unrolling = allxy_unrolling
    allxy_drag_pulse_tuning = allxy_drag_pulse_tuning
    drag_pulse_tuning = drag_pulse_tuning
    flipping = flipping
    dispersive_shift = dispersive_shift
    StdRB = StdRB
    single_shot_classification = single_shot_classification
    standard_rb = standard_rb
