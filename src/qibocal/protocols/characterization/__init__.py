from enum import Enum

from .allxy.allxy import allxy
from .allxy.allxy_drag_pulse_tuning import allxy_drag_pulse_tuning
from .allxy.drag_pulse_tuning import drag_pulse_tuning
from .classification import single_shot_classification
from .coherence.spin_echo import spin_echo
from .coherence.t1 import t1
from .coherence.t1_sequences import t1_sequences
from .coherence.t2 import t2
from .coherence.t2_sequences import t2_sequences
from .coherence.zeno import zeno
from .dispersive_shift import dispersive_shift
from .fast_reset.fast_reset import fast_reset
from .flipping import flipping
from .flux_dependence.qubit_flux_dependence import qubit_flux
from .flux_dependence.resonator_flux_dependence import resonator_flux
from .qubit_spectroscopy import qubit_spectroscopy
from .rabi.amplitude import rabi_amplitude
from .rabi.length import rabi_length
from .rabi.length_sequences import rabi_length_sequences
from .ramsey import ramsey
from .ramsey_sequences import ramsey_sequences
from .randomized_benchmarking.standard_rb import standard_rb
from .readout_characterization import readout_characterization
from .readout_optimization.resonator_frequency import resonator_frequency
from .resonator_punchout import resonator_punchout
from .resonator_punchout_attenuation import resonator_punchout_attenuation
from .resonator_spectroscopy import resonator_spectroscopy
from .resonator_spectroscopy_attenuation import resonator_spectroscopy_attenuation
from .signal_experiments.time_of_flight_readout import time_of_flight_readout
from .two_qubit_interaction.chevron import chevron
from .two_qubit_interaction.cz_virtualz import cz_virtualz


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
    rabi_length_sequences = rabi_length_sequences
    ramsey = ramsey
    ramsey_sequences = ramsey_sequences
    t1 = t1
    t1_sequences = t1_sequences
    t2 = t2
    t2_sequences = t2_sequences
    time_of_flight_readout = time_of_flight_readout
    single_shot_classification = single_shot_classification
    spin_echo = spin_echo
    allxy = allxy
    allxy_drag_pulse_tuning = allxy_drag_pulse_tuning
    drag_pulse_tuning = drag_pulse_tuning
    flipping = flipping
    dispersive_shift = dispersive_shift
    chevron = chevron
    cz_virtualz = cz_virtualz
    standard_rb = standard_rb
    readout_characterization = readout_characterization
    resonator_frequency = resonator_frequency
    fast_reset = fast_reset
    zeno = zeno
