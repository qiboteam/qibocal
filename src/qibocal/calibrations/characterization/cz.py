# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import FluxPulse, Pulse, PulseSequence, PulseType, Rectangular
from qibocal.data import DataUnits


def tune_transition(
    platform: AbstractPlatform,
    qubit: int,
    flux_pulse_duration_start,
    flux_pulse_duration_end,
    flux_pulse_duration_step,
    flux_pulse_amplitude_start,
    flux_pulse_amplitude_end,
    flux_pulse_amplitude_step,
):
    
    platform.reload_settings()
    
    initialize_1 = platform.create_RX_pulse(qubit, start=0, relative_phase=0)
    initialize_2 = platform.create_RX_pulse(2, start=0, relative_phase=0)
    
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2
    
    flux_pulse_plus = FluxPulse(
        start=initialize_1.se_finish,
        duration=flux_pulse_duration_start,  
        amplitude=flux_pulse_amplitude_start,  
        relative_phase=0,
        shape=Rectangular(), 
        channel=platform.qubit_channel_map[highfreq][2],
        qubit=highfreq,
    )
    
    flux_pulse_minus = FluxPulse(
        start=flux_pulse_plus.se_finish,
        duration=flux_pulse_duration_start,  
        amplitude=-flux_pulse_amplitude_start,  
        relative_phase=0,
        shape=Rectangular(), 
        channel=platform.qubit_channel_map[highfreq][2],
        qubit=highfreq,
    )
    
    measure_lowfreq = platform.create_qubit_readout_pulse(lowfreq, start=flux_pulse.se_finish)
    
    data = DataUnits(
        name=f"data_q{12}",
        quantities={
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
        },
    )

    amplitudes = np.arange(
        flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step
    )
    durations = np.arange(
        flux_pulse_duration_start, flux_pulse_duration_end, flux_pulse_duration_step
    )
    
    # Might want to fix duration to expected time for 2 qubit gate.
    for amplitude in amplitudes:
        for duration in durations:
            flux_pulse_plus.amplitude = amplitude
            flux_pulse_minus.amplitude = -amplitude
            flux_pulse_plus.duration = duration
            flux_pulse_minus.duration = duration
            
            seq = initialize_1 + initialize_2 + flux_pulse_plus + flux_pulse_minus + measure_lowfreq
            res = platform.execute_pulse_sequence(seq)[measure_lowfreq.serial]
            
            results = {
                "MSR[V]": res[0],
                "i[V]": res[2],
                "q[V]": res[3],
                "phase[rad]": res[1],
                "flux_pulse_duration[ns]": duration,
                "flux_pulse_amplitude[dimensionless]": amplitude,
            }
            data.add(results)
            
    yield data
    

def get_remnant_phase(
    platform: AbstractPlatform,
    qubit: int,
    theta_start,
    theta_end,
    theta_step,
    flux_pulse_duration,
    flux_pulse_amplitude,
):
    
    platform.reload_settings()
    
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2
    
    initialize_1 = platform.create_RX_pulse(highfreq, start=0, relative_phase=0)
    initialize_2 = platform.create_RX90_pulse(lowfreq, start=0, relative_phase=np.pi/2)
    
    flux_pulse = FluxPulse(
        start=initialize_2.se_finish,
        duration=flux_pulse_duration,  
        amplitude=flux_pulse_amplitude,  
        relative_phase=0,
        shape=Rectangular(), 
        channel=platform.qubit_channel_map[highfreq][2],
        qubit=highfreq,
    )
    
    RX90_pulse_1 = self.create_RX90_pulse(
        qubit, flux_pulse.se_finish, relative_phase=0
    )
    RX90_pulse_2 = self.create_RX90_pulse(
        qubit, RX90_pulse_1.se_finish, relative_phase=theta - np.pi
    )
    
    measure_lowfreq = platform.create_qubit_readout_pulse(lowfreq, start=flux_pulse.se_finish)
    
    data = DataUnits(
        name=f"data_q{12}",
        quantities={
            "theta": "ns",
            "flux_pulse_amplitude": "dimensionless",
        },
    )

    amplitudes = np.arange(
        flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step
    )
    durations = np.arange(
        flux_pulse_duration_start, flux_pulse_duration_end, flux_pulse_duration_step
    )

    for amplitude in amplitudes:
        for duration in durations:
            
            seq = initialize_1 + initialize_2 + flux_pulse + measure_lowfreq
            res = platform.execute_pulse_sequence(seq)[measure_lowfreq.serial]
            
            results = {
                "MSR[V]": res[0],
                "i[V]": res[2],
                "q[V]": res[3],
                "phase[rad]": res[1],
                "flux_pulse_duration[ns]": duration,
                "flux_pulse_amplitude[dimensionless]": amplitude,
            }
            data.add(results)
            
    yield data
