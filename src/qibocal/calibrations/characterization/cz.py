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
    single_flux = True
):
    
    platform.reload_settings()
    
    initialize_1 = platform.create_RX_pulse(qubit, start=0, relative_phase=0)
    initialize_2 = platform.create_RX_pulse(2, start=0, relative_phase=0)
    
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2
        
    if single_flux:
        flux_pulse = FluxPulse(
            start=initialize_1.se_finish,
            duration=flux_pulse_duration_start,  
            amplitude=flux_pulse_amplitude_start,  
            relative_phase=0,
            shape=Rectangular(), 
            channel=platform.qubit_channel_map[highfreq][2],
            qubit=highfreq,
        )
        measure_lowfreq = platform.create_qubit_readout_pulse(lowfreq, start=flux_pulse.se_finish)
        measure_highfreq = platform.create_qubit_readout_pulse(highfreq, start=flux_pulse.se_finish)
        
    else:
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
        measure_lowfreq = platform.create_qubit_readout_pulse(lowfreq, start=flux_pulse_minus.se_finish)
        measure_highfreq = platform.create_qubit_readout_pulse(highfreq, start=flux_pulse_minus.se_finish)

    
    data_low = DataUnits(
        name=f"data_q{lowfreq}{highfreq}_low",
        quantities={
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
        },
    )
    
    data_high = DataUnits(
        name=f"data_q{lowfreq}{highfreq}_high",
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
    
    if single_flux:
        seq = initialize_1 + initialize_2 + flux_pulse + measure_lowfreq + measure_highfreq
    else:
        seq = initialize_1 + initialize_2 + flux_pulse_plus + flux_pulse_minus + measure_lowfreq + measure_highfreq

    # Might want to fix duration to expected time for 2 qubit gate.
    for amplitude in amplitudes:
        for duration in durations:
            if single_flux:
                flux_pulse.amplitude = amplitude
                flux_pulse.duration = duration
            else:
                flux_pulse_plus.amplitude = amplitude
                flux_pulse_minus.amplitude = -amplitude
                flux_pulse_plus.duration = duration
                flux_pulse_minus.duration = duration
            
            res = platform.execute_pulse_sequence(seq)[measure_lowfreq.serial]
            
            res_low = res[measure_lowfreq.serial]
            results_low = {
                "MSR[V]": res_low[0],
                "i[V]": res_low[2],
                "q[V]": res_low[3],
                "phase[rad]": res_low[1],
                "flux_pulse_duration[ns]": duration,
                "flux_pulse_amplitude[dimensionless]": amplitude,
            }
            
            res_high = res[measure_highfreq.serial]
            results_high = {
                "MSR[V]": res_high[0],
                "i[V]": res_high[2],
                "q[V]": res_high[3],
                "phase[rad]": res_high[1],
                "flux_pulse_duration[ns]": duration,
                "flux_pulse_amplitude[dimensionless]": amplitude,
            }
            
            data_low.add(results_low)
            data_high.add(results_high)
            
    yield data_low, data_high
    

def tune_landscape(
    platform: AbstractPlatform,
    qubit: int,
    theta_start,
    theta_end,
    theta_step,
    flux_pulse_duration,
    flux_pulse_amplitude,
    single_flux = True
):
    
    platform.reload_settings()
    
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2
    
    x_pulse_start = platform.create_RX_pulse(highfreq, start=0, relative_phase=0)
    y90_pulse = platform.create_RX90_pulse(lowfreq, start=0, relative_phase=np.pi/2)
    
    if single_flux:
        flux_pulse = FluxPulse(
            start=y90_pulse.se_finish,
            duration=flux_pulse_duration,  
            amplitude=flux_pulse_amplitudet,  
            relative_phase=0,
            shape=Rectangular(), 
            channel=platform.qubit_channel_map[highfreq][2],
            qubit=highfreq,
        )
        theta_pulse = self.create_RX90_pulse(lowfreq, flux_pulse.se_finish, relative_phase=theta_start)
        x_pulse_end = platform.create_RX_pulse(highfreq, start=flux_pulse.se_finish, relative_phase=0)
        
    else:
        flux_pulse_plus = FluxPulse(
            start=y90_pulse.se_finish,
            duration=flux_pulse_duration,  
            amplitude=flux_pulse_amplitude,  
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
        theta_pulse = self.create_RX90_pulse(lowfreq, flux_pulse_minus.se_finish, relative_phase=theta_start)
        x_pulse_end = platform.create_RX_pulse(highfreq, start=flux_pulse_minus.se_finish, relative_phase=0)
        
    measure_lowfreq = platform.create_qubit_readout_pulse(lowfreq, start=theta_pulse.se_finish)
    measure_highfreq = platform.create_qubit_readout_pulse(highfreq, start=theta_pulse.se_finish)
        
    data_low_I = DataUnits(
        name=f"data_q{lowfreq}{highfreq}_low_I",
        quantities={
            "theta": "ns",
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
        },
    )
    
    data_high_I = DataUnits(
        name=f"data_q{lowfreq}{highfreq}_high_I",
        quantities={
            "theta": "ns",
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
        },
    )
    
    data_low_X = DataUnits(
        name=f"data_q{lowfreq}{highfreq}_low_X",
        quantities={
            "theta": "ns",
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
        },
    )
    
    data_high_X = DataUnits(
        name=f"data_q{lowfreq}{highfreq}_high_X",
        quantities={
            "theta": "ns",
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
        },
    )

    thetas = np.arange(
        theta_start, theta_end, theta_step
    )
    
    setups = ['I', 'X']
    
    for setup in setups:
        if setup = 'I':
            if single_flux:
                seq = y90_pulse + flux_pulse + theta_pulse + measure_lowfreq + measure_highfreq
            else:
                seq = y90_pulse + flux_pulse_pulse + flux_pulse_minus + theta_pulse + measure_lowfreq + measure_highfreq
        elif setup = 'X':
            if single_flux:
                seq = x_pulse_start + y90_pulse + flux_pulse + theta_pulse + x_pulse_end + measure_lowfreq + measure_highfreq
            else:
                seq = x_pulse_start + y90_pulse + flux_pulse_plus + flux_pulse_minus + theta_pulse + x_pulse_end + measure_lowfreq + measure_highfreq

        for theta in thetas:
            theta_pulse.relative_phase = theta
            
            res = platform.execute_pulse_sequence(seq)[measure_lowfreq.serial]
            res_low = res[measure_lowfreq.serial]
            results_low = {
                "MSR[V]": res_low[0],
                "i[V]": res_low[2],
                "q[V]": res_low[3],
                "phase[rad]": res_low[1],
                "flux_pulse_duration[ns]": duration,
                "flux_pulse_amplitude[dimensionless]": amplitude,
            }
            
            res_high = res[measure_highfreq.serial]
            results_high = {
                "MSR[V]": res_high[0],
                "i[V]": res_high[2],
                "q[V]": res_high[3],
                "phase[rad]": res_high[1],
                "flux_pulse_duration[ns]": duration,
                "flux_pulse_amplitude[dimensionless]": amplitude,
            }
            
            if setup = 'I':
                data_low_I.add(results_low)
                data_high_I.add(results_high)
            elif setup = 'X':
                data_low_X.add(results_low)
                data_high_X.add(results_high)
            
    yield data_low_I, data_high_I, data_low_X, data_high_X
