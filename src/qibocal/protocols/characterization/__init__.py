from enum import Enum

from .allxy.allxy import allxy
from .allxy.allxy_drag_pulse_tuning import allxy_drag_pulse_tuning
from .classification import single_shot_classification
from .coherence.spin_echo import spin_echo
from .coherence.spin_echo_signal import spin_echo_signal
from .coherence.t1 import t1
from .coherence.t1_sequences import t1_sequences
from .coherence.t1_signal import t1_signal
from .coherence.t2 import t2
from .coherence.t2_sequences import t2_sequences
from .coherence.t2_signal import t2_signal
from .coherence.zeno import zeno
from .coherence.zeno_signal import zeno_signal
from .couplers.coupler_chevron import coupler_chevron
from .couplers.coupler_qubit_spectroscopy import coupler_qubit_spectroscopy
from .couplers.coupler_resonator_spectroscopy import coupler_resonator_spectroscopy
from .dispersive_shift import dispersive_shift
from .dispersive_shift_qutrit import dispersive_shift_qutrit
from .drag import drag_tuning
from .fast_reset.fast_reset import fast_reset
from .flipping import flipping
from .flipping_signal import flipping_signal
from .flux_dependence.avoided_crossing import avoided_crossing
from .flux_dependence.qubit_crosstalk import qubit_crosstalk
from .flux_dependence.qubit_flux_dependence import qubit_flux
from .flux_dependence.qubit_flux_tracking import qubit_flux_tracking
from .flux_dependence.resonator_crosstalk import resonator_crosstalk
from .flux_dependence.resonator_flux_dependence import resonator_flux
from .qubit_spectroscopy import qubit_spectroscopy
from .qubit_spectroscopy_ef import qubit_spectroscopy_ef
from .qutrit_classification import qutrit_classification
from .rabi.amplitude import rabi_amplitude
from .rabi.amplitude_signal import rabi_amplitude_signal
from .rabi.ef import rabi_amplitude_ef
from .rabi.length import rabi_length
from .rabi.length_sequences import rabi_length_sequences
from .rabi.length_signal import rabi_length_signal
from .ramsey.ramsey import ramsey
from .ramsey.ramsey_signal import ramsey_signal
from .randomized_benchmarking.filtered_rb import filtered_rb
from .randomized_benchmarking.standard_rb import standard_rb
from .readout_characterization import readout_characterization
from .readout_mitigation_matrix import readout_mitigation_matrix
from .readout_optimization.resonator_amplitude import resonator_amplitude
from .readout_optimization.resonator_frequency import resonator_frequency
from .readout_optimization.twpa_calibration.frequency import twpa_frequency
from .readout_optimization.twpa_calibration.frequency_power import twpa_frequency_power
from .readout_optimization.twpa_calibration.frequency_SNR import twpa_frequency_snr
from .readout_optimization.twpa_calibration.power import twpa_power
from .readout_optimization.twpa_calibration.power_SNR import twpa_power_snr
from .resonator_punchout import resonator_punchout
from .resonator_punchout_attenuation import resonator_punchout_attenuation
from .resonator_spectroscopy import resonator_spectroscopy
from .signal_experiments.calibrate_state_discrimination import (
    calibrate_state_discrimination,
)
from .signal_experiments.time_of_flight_readout import time_of_flight_readout
from .two_qubit_interaction import (
    chevron,
    chevron_signal,
    chsh_circuits,
    chsh_pulses,
    cz_virtualz,
    cz_virtualz_signal,
)


class Operation(Enum):
    resonator_spectroscopy = resonator_spectroscopy
    resonator_punchout = resonator_punchout
    resonator_punchout_attenuation = resonator_punchout_attenuation
    resonator_flux = resonator_flux
    resonator_crosstalk = resonator_crosstalk
    qubit_spectroscopy = qubit_spectroscopy
    qubit_flux = qubit_flux
    qubit_flux_tracking = qubit_flux_tracking
    qubit_crosstalk = qubit_crosstalk
    rabi_amplitude = rabi_amplitude
    rabi_length = rabi_length
    rabi_length_sequences = rabi_length_sequences
    rabi_amplitude_signal = rabi_amplitude_signal
    rabi_length_signal = rabi_length_signal
    ramsey = ramsey
    ramsey_signal = ramsey_signal
    t1 = t1
    t1_signal = t1_signal
    t1_sequences = t1_sequences
    t2 = t2
    t2_signal = t2_signal
    t2_sequences = t2_sequences
    time_of_flight_readout = time_of_flight_readout
    single_shot_classification = single_shot_classification
    spin_echo = spin_echo
    spin_echo_signal = spin_echo_signal
    allxy = allxy
    allxy_drag_pulse_tuning = allxy_drag_pulse_tuning
    drag_tuning = drag_tuning
    flipping = flipping
    dispersive_shift = dispersive_shift
    chevron = chevron
    chevron_signal = chevron_signal
    cz_virtualz = cz_virtualz
    standard_rb = standard_rb
    filtered_rb = filtered_rb
    resonator_frequency = resonator_frequency
    fast_reset = fast_reset
    zeno = zeno
    zeno_signal = zeno_signal
    chsh_pulses = chsh_pulses
    readout_characterization = readout_characterization
    chsh_circuits = chsh_circuits
    readout_mitigation_matrix = readout_mitigation_matrix
    twpa_frequency = twpa_frequency
    twpa_frequency_SNR = twpa_frequency_snr
    twpa_power = twpa_power
    twpa_power_SNR = twpa_power_snr
    twpa_frequency_power = twpa_frequency_power
    rabi_amplitude_ef = rabi_amplitude_ef
    qubit_spectroscopy_ef = qubit_spectroscopy_ef
    qutrit_classification = qutrit_classification
    resonator_amplitude = resonator_amplitude
    avoided_crossing = avoided_crossing
    dispersive_shift_qutrit = dispersive_shift_qutrit
    coupler_resonator_spectroscopy = coupler_resonator_spectroscopy
    coupler_qubit_spectroscopy = coupler_qubit_spectroscopy
    cz_virtualz_signal = cz_virtualz_signal
    coupler_chevron = coupler_chevron
    flipping_signal = flipping_signal
    calibrate_state_discrimination = calibrate_state_discrimination
