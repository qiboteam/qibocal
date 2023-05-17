from enum import Enum

from .allxy.allxy import allxy
from .allxy.allxy_drag_pulse_tuning import allxy_drag_pulse_tuning
from .allxy.drag_pulse_tuning import drag_pulse_tuning
from .classification import single_shot_classification
from .coherence.spin_echo import spin_echo
from .coherence.t1 import t1
from .coherence.t1_sweep import t1_sweep
from .coherence.t2 import t2
from .coherence.t2_sweep import t2_sweep
from .couplers.coupler_frequency import coupler_frequency
from .couplers.coupler_frequency_flux import coupler_frequency_flux
from .couplers.coupler_swap_frequency import coupler_swap_frequency
from .couplers.coupler_swap_frequency_flux import coupler_swap_frequency_flux
from .dispersive_shift import dispersive_shift
from .flipping import flipping
from .flux_depedence.qubit_flux_dependence import qubit_flux
from .flux_depedence.resonator_flux_dependence import resonator_flux
from .qubit_spectroscopy import qubit_spectroscopy
from .rabi.amplitude import rabi_amplitude
from .rabi.length import rabi_length
from .rabi.length_sweep import rabi_length_sweep
from .ramsey import ramsey
from .ramsey_sweep import ramsey as ramsey_sweep
from .resonator_punchout import resonator_punchout
from .resonator_punchout_attenuation import resonator_punchout_attenuation
from .resonator_spectroscopy import resonator_spectroscopy


class Operation(Enum):
    resonator_spectroscopy = resonator_spectroscopy
    resonator_punchout = resonator_punchout
    resonator_punchout_attenuation = resonator_punchout_attenuation
    resonator_flux = resonator_flux
    qubit_spectroscopy = qubit_spectroscopy
    qubit_flux = qubit_flux
    rabi_amplitude = rabi_amplitude
    rabi_length = rabi_length
    rabi_length_sweep = rabi_length_sweep
    ramsey = ramsey
    ramsey_sweep = ramsey_sweep
    t1 = t1
    t1_sweep = t1_sweep
    t2 = t2
    t2_sweep = t2_sweep
    single_shot_classification = single_shot_classification
    spin_echo = spin_echo
    allxy = allxy
    allxy_drag_pulse_tuning = allxy_drag_pulse_tuning
    drag_pulse_tuning = drag_pulse_tuning
    flipping = flipping
    dispersive_shift = dispersive_shift
    coupler_swap_frequency = coupler_swap_frequency
    coupler_frequency = coupler_frequency
    coupler_frequency_flux = coupler_frequency_flux
    coupler_swap_frequency_flux = coupler_swap_frequency_flux
