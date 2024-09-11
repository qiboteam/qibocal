from enum import Enum

from .allxy.allxy import allxy
from .allxy.allxy_drag_pulse_tuning import allxy_drag_pulse_tuning
from .allxy.drag_pulse_tuning import drag_pulse_tuning
from .classification import single_shot_classification
from .coherence.spin_echo import spin_echo
from .coherence.spin_echo_sequence import spin_echo_sequence

# from .coherence.spin_echo_signal import spin_echo_signal
from .coherence.t1 import t1
from .coherence.t1_sequences import t1_sequences
from .coherence.t1_signal import t1_signal
from .coherence.t2 import t2
from .coherence.t2_sequences import t2_sequences
from .coherence.t2_signal import t2_signal
from .coherence.zeno import zeno
from .coherence.zeno_signal import zeno_signal
from .couplers.coupler_qubit_spectroscopy import coupler_qubit_spectroscopy
from .couplers.coupler_resonator_spectroscopy import coupler_resonator_spectroscopy
from .dispersive_shift import dispersive_shift
from .dispersive_shift_qutrit import dispersive_shift_qutrit
from .fast_reset.fast_reset import fast_reset
from .flipping import flipping

# from .flipping_signal import flipping_signal
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
from .ramsey import ramsey
from .ramsey_sequences import ramsey_sequences
from .ramsey_signal import ramsey_signal

# from .ramsey.ramsey import ramsey
# from .ramsey.ramsey_signal import ramsey_signal
from .randomized_benchmarking.standard_rb import standard_rb
from .readout_characterization import readout_characterization
from .readout_mitigation_matrix import readout_mitigation_matrix
from .readout_optimization.resonator_amplitude import resonator_amplitude
from .readout_optimization.resonator_frequency import resonator_frequency
from .readout_optimization.twpa_calibration.frequency import twpa_frequency
from .readout_optimization.twpa_calibration.frequency_power import twpa_frequency_power

# from .readout_optimization.twpa_calibration.frequency_SNR import twpa_frequency_snr
from .readout_optimization.twpa_calibration.power import twpa_power

# from .readout_optimization.twpa_calibration.power_SNR import twpa_power_snr
from .resonator_punchout import resonator_punchout
from .resonator_punchout_attenuation import resonator_punchout_attenuation
from .resonator_spectroscopy import resonator_spectroscopy
from .resonator_spectroscopy_attenuation import (  # will be removed
    resonator_spectroscopy_attenuation,
)

# from .signal_experiments.calibrate_state_discrimination import (
#     calibrate_state_discrimination,
# )
from .signal_experiments.time_of_flight_readout import time_of_flight_readout
from .two_qubit_interaction import chevron, chsh_circuits, chsh_pulses, cz_virtualz

####################################################################################
from .z.allxy.allxy import allxy
from .z.allxy.allxy_drag_pulse_tuning import allxy_drag_pulse_tuning
from .z.allxy.allxy_resonator_depletion_tuning import allxy_resonator_depletion_tuning

# from .z.t1_t2_vs_temperature import t1_t2_vs_temperature
from .z.coherence.t2 import t2
from .z.coherence.t2_signal import t2_signal
from .z.couplers.coupler_chevron import coupler_chevron
from .z.couplers.coupler_chevron_signal import coupler_chevron_signal_amplitude
from .z.couplers.coupler_qubit_spectroscopy import coupler_qubit_spectroscopy_bias
from .z.couplers.coupler_resonator_spectroscopy import (
    coupler_resonator_spectroscopy_amplitude,
    coupler_resonator_spectroscopy_bias,
)
from .z.dispersive_shift import dispersive_shift
from .z.ef.dispersive_shift_qutrit import dispersive_shift_qutrit
from .z.ef.qubit_spectroscopy_ef import qubit_spectroscopy_ef
from .z.ef.qutrit_classification import qutrit_classification
from .z.ef.rabi_ef import rabi_amplitude_ef
from .z.flux_dependence.qubit_flux_dependence import qubit_flux
from .z.in_progress.cryoscope import cryoscope
from .z.in_progress.cryoscope_signal import cryoscope_signal
from .z.in_progress.qubit_mixer_calibration import qubit_mixer_calibration
from .z.in_progress.qubit_power_spectroscopy import qubit_power_spectroscopy
from .z.in_progress.qubit_spectroscopy_with_lo import qubit_spectroscopy_with_lo
from .z.in_progress.resonator_mixer_calibration import resonator_mixer_calibration
from .z.in_progress.resonator_spectroscopy_with_lo import resonator_spectroscopy_with_lo
from .z.in_progress.ro_resonator_amplitude import resonator_amplitude
from .z.qubit_spectroscopy import qubit_spectroscopy
from .z.qubit_state_crosstalk import qubit_state_crosstalk
from .z.rabi.length import rabi_length
from .z.rabi.length_signal import rabi_length_signal
from .z.ramsey.ramsey import ramsey

# from .z.rabi_frequency_length_signal import rabi_frequency_length_signal
from .z.ramsey.ramsey_signal import ramsey_signal
from .z.randomized_benchmarking.standard_rb import standard_rb

# from .z.resonator_qubit_spectroscopy import resonator_qubit_spectroscopy
from .z.resonator_twpa_freq import resonator_twpa_freq
from .z.resonator_twpa_pow import resonator_twpa_pow
from .z.spurious_identification import spurious_identification
from .z.two_qubit_interaction.chevron import chevron
from .z.two_qubit_interaction.cz_sweep import cz_sweep
from .z.two_qubit_interaction.cz_virtualz import cz_virtualz


class Operation(Enum):
    resonator_spectroscopy = resonator_spectroscopy
    resonator_spectroscopy_attenuation = (
        resonator_spectroscopy_attenuation  # will be removed
    )
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
    ramsey_sequences = ramsey_sequences  # will be removed
    t1 = t1
    t1_signal = t1_signal
    t1_sequences = t1_sequences
    t2 = t2
    t2_signal = t2_signal
    t2_sequences = t2_sequences
    time_of_flight_readout = time_of_flight_readout
    single_shot_classification = single_shot_classification
    spin_echo = spin_echo
    spin_echo_sequence = spin_echo_sequence
    allxy = allxy
    allxy_drag_pulse_tuning = allxy_drag_pulse_tuning
    flipping = flipping
    dispersive_shift = dispersive_shift
    standard_rb = standard_rb
    readout_characterization = readout_characterization
    resonator_frequency = resonator_frequency
    fast_reset = fast_reset
    zeno = zeno
    zeno_signal = zeno_signal
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
    avoided_crossing = avoided_crossing
    dispersive_shift_qutrit = dispersive_shift_qutrit
    coupler_resonator_spectroscopy_bias = coupler_resonator_spectroscopy_bias
    coupler_resonator_spectroscopy_amplitude = coupler_resonator_spectroscopy_amplitude
    coupler_qubit_spectroscopy = coupler_qubit_spectroscopy_bias
    # flipping_signal = flipping_signal
    # calibrate_state_discrimination = calibrate_state_discrimination

    # rabi_frequency_length_signal = rabi_frequency_length_signal
    # resonator_qubit_spectroscopy = resonator_qubit_spectroscopy
    resonator_twpa_pow = resonator_twpa_pow
    resonator_twpa_freq = resonator_twpa_freq
    spurious_identification = spurious_identification
    # t1_t2_vs_temperature = t1_t2_vs_temperature

    # compatibility with future changes future_name = current_name
    spin_echo_signal = spin_echo
    drag_tuning = drag_pulse_tuning
    chevron_signal = chevron
    twpa_power_SNR = twpa_power
    twpa_frequency_SNR = twpa_frequency
    cz_virtualz_signal = cz_virtualz

    qubit_mixer_calibration = qubit_mixer_calibration
    resonator_mixer_calibration = resonator_mixer_calibration
    resonator_spectroscopy_with_lo = qubit_spectroscopy_with_lo
    qubit_spectroscopy_with_lo = qubit_spectroscopy_with_lo
    coupler_chevron = coupler_chevron
    coupler_chevron_signal_amplitude = coupler_chevron_signal_amplitude
    cryoscope = cryoscope
    allxy_resonator_depletion_tuning = allxy_resonator_depletion_tuning
    qubit_power_spectroscopy = qubit_power_spectroscopy
    cz_sweep = cz_sweep

    qubit_state_crosstalk = qubit_state_crosstalk
