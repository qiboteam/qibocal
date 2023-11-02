from enum import Enum

from .allxy.allxy import allxy
from .allxy.allxy_drag_pulse_tuning import allxy_drag_pulse_tuning
from .allxy.drag_pulse_tuning import drag_pulse_tuning
from .classification import single_shot_classification
from .coherence.spin_echo import spin_echo
from .coherence.spin_echo_msr import spin_echo_msr
from .coherence.t1 import t1
from .coherence.t1_msr import t1_msr
from .coherence.t1_sequences import t1_sequences
from .coherence.t2 import t2
from .coherence.t2_msr import t2_msr
from .coherence.t2_sequences import t2_sequences
from .coherence.zeno import zeno
from .coherence.zeno_msr import zeno_msr
from .couplers.coupler_qubit_spectroscopy import coupler_qubit_spectroscopy
from .couplers.coupler_resonator_spectroscopy import coupler_resonator_spectroscopy
from .dispersive_shift import dispersive_shift
from .dispersive_shift_qutrit import dispersive_shift_qutrit
from .fast_reset.fast_reset import fast_reset
from .flipping import flipping
from .flux_dependence.qubit_flux_dependence import qubit_crosstalk, qubit_flux
from .flux_dependence.resonator_flux_dependence import (
    resonator_crosstalk,
    resonator_flux,
)
from .qubit_spectroscopy import qubit_spectroscopy
from .qubit_spectroscopy_ef import qubit_spectroscopy_ef
from .qutrit_classification import qutrit_classification
from .rabi.amplitude import rabi_amplitude
from .rabi.amplitude_msr import rabi_amplitude_msr
from .rabi.ef import rabi_amplitude_ef
from .rabi.length import rabi_length
from .rabi.length_msr import rabi_length_msr
from .rabi.length_sequences import rabi_length_sequences
from .ramsey import ramsey
from .ramsey_msr import ramsey_msr
from .ramsey_sequences import ramsey_sequences
from .randomized_benchmarking.standard_rb import standard_rb
from .readout_characterization import readout_characterization
from .readout_mitigation_matrix import readout_mitigation_matrix
from .readout_optimization.resonator_amplitude import resonator_amplitude
from .readout_optimization.resonator_frequency import resonator_frequency
from .readout_optimization.twpa_calibration.frequency import twpa_frequency
from .readout_optimization.twpa_calibration.frequency_power import twpa_frequency_power
from .readout_optimization.twpa_calibration.power import twpa_power
from .resonator_punchout import resonator_punchout
from .resonator_punchout_attenuation import resonator_punchout_attenuation
from .resonator_spectroscopy import resonator_spectroscopy
from .resonator_spectroscopy_attenuation import resonator_spectroscopy_attenuation
from .signal_experiments.time_of_flight_readout import time_of_flight_readout
from .two_qubit_interaction import chevron, chsh_circuits, chsh_pulses, cz_virtualz


class Operation(Enum):
    resonator_spectroscopy = resonator_spectroscopy
    resonator_spectroscopy_attenuation = resonator_spectroscopy_attenuation
    resonator_punchout = resonator_punchout
    resonator_punchout_attenuation = resonator_punchout_attenuation
    resonator_flux = resonator_flux
    resonator_crosstalk = resonator_crosstalk
    qubit_spectroscopy = qubit_spectroscopy
    qubit_flux = qubit_flux
    qubit_crosstalk = qubit_crosstalk
    rabi_amplitude = rabi_amplitude
    rabi_length = rabi_length
    rabi_length_sequences = rabi_length_sequences
    rabi_amplitude_msr = rabi_amplitude_msr
    rabi_length_msr = rabi_length_msr
    ramsey = ramsey
    ramsey_msr = ramsey_msr
    ramsey_sequences = ramsey_sequences
    t1 = t1
    t1_msr = t1_msr
    t1_sequences = t1_sequences
    t2 = t2
    t2_msr = t2_msr
    t2_sequences = t2_sequences
    time_of_flight_readout = time_of_flight_readout
    single_shot_classification = single_shot_classification
    spin_echo = spin_echo
    spin_echo_msr = spin_echo_msr
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
    zeno_msr = zeno_msr
    chsh_pulses = chsh_pulses
    chsh_circuits = chsh_circuits
    readout_mitigation_matrix = readout_mitigation_matrix
    twpa_frequency = twpa_frequency
    twpa_power = twpa_power
    twpa_frequency_power = twpa_frequency_power
    rabi_amplitude_ef = rabi_amplitude_ef
    qubit_spectroscopy_ef = qubit_spectroscopy_ef
    qutrit_classification = qutrit_classification
    resonator_amplitude = resonator_amplitude
    dispersive_shift_qutrit = dispersive_shift_qutrit
    coupler_resonator_spectroscopy = coupler_resonator_spectroscopy
    coupler_qubit_spectroscopy = coupler_qubit_spectroscopy
