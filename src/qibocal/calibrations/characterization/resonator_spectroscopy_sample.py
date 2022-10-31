import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal.calibrations.characterization.utils import snr, choose_freq, get_noise, update_f, plot_punchout, plot_flux
from qibocal.data import DataUnits

def scan_level(best_f, best_msr, max_runs, thr, span, resolution, noise, platform, ro_pulse, qubit, sequence):
    for _ in range(max_runs):
        if _ == 0:
            freq = best_f
        else:
            freq = choose_freq(best_f, span, resolution)
        platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
        msr, phase, i, q = platform.execute_pulse_sequence(sequence)[ro_pulse.serial]
        
        if platform.resonator_type == "3D":
            if msr > best_msr:
                if abs(snr(msr, noise)) >= thr:
                    best_f, best_msr = freq, msr
                    return best_f, best_msr
        else:
            if msr < best_msr:
                if abs(snr(msr, noise)) >= thr:
                    best_f, best_msr = freq, msr
                    return best_f, best_msr
    return best_f, best_msr


def scan_small(best_f, best_msr, max_runs, thr, span, resolution, noise, platform, ro_pulse, qubit, sequence):
    start_f = best_f
    scan = np.linspace(-span/2, span/2, resolution)
    for s in scan:
        freq = start_f + s
        platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
        msr, phase, i, q = platform.execute_pulse_sequence(sequence)[ro_pulse.serial]
        if platform.resonator_type == "3D":
            if msr > best_msr:
                best_f, best_msr = freq, msr
        else:
            if msr < best_msr:
                best_f, best_msr = freq, msr
    return best_f, best_msr
    
               
def resonator_punchout_sample(
    platform: AbstractPlatform,
    qubit: int,
    min_att,
    max_att,
    step_att,
    max_runs,
    thr,
    spans,
    small_spans,
    resolution
    ):
    
    data = DataUnits(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "attenuation": "dB"}
    )
        
    platform.reload_settings()
    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)
    
    resonator_frequency = platform.characterization["single_qubit"][qubit]["resonator_freq"]
    
    attenuation_range = np.arange(min_att, max_att, step_att)
    best_f = resonator_frequency
    
    opt_att = 30
    opt_snr = 0
    freqs = []
    for att in attenuation_range:
        platform.ro_port[qubit].attenuation = att
        
        background = [best_f+1e7, best_f-1e7]
        noise = get_noise(background, platform, ro_pulse, qubit, sequence)
        best_msr = noise
        for span in spans:
            best_f, best_msr = scan_level(best_f, best_msr, max_runs, thr, span, resolution, noise, platform, ro_pulse, qubit, sequence)
        for span in small_spans:
            best_f, best_msr = scan_small(best_f, best_msr, max_runs, thr, span, 11, noise, platform, ro_pulse, qubit, sequence)
        freqs.append(best_f)
        results = {
                    "MSR[V]": best_msr,
                    "i[V]": 455,
                    "q[V]": 455,
                    "phase[rad]": 455,
                    "frequency[Hz]": best_f,
                    "attenuation[dB]": att,
                }
        data.add(results)
        if att >= 30:
            if abs(snr(best_msr, noise)) > opt_snr:
                opt_snr = abs(snr(best_msr, noise))
                opt_att = att
                opt_f = best_f
                
    plot_punchout(attenuation_range, freqs, qubit)
    
    print(f'For qubit {qubit}:')
    print(f'Best response found at frequency {opt_f} Hz for attenuation value of {opt_att} dB.\n')
    
    yield data


def resonator_flux_sample(
    platform: AbstractPlatform,
    qubit: int,
    current_min,
    current_max, 
    current_step,
    fluxline,
    max_runs, 
    thr,
    spans,
    small_spans,
    resolution,
    ):
    
    data = DataUnits(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "current": "A"}
    )
    
    if fluxline == "qubit":
        fluxline = qubit
        
    platform.reload_settings()
    
    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)
    
    resonator_frequency = platform.characterization["single_qubit"][qubit]["resonator_freq"]
    qubit_biasing_current = platform.characterization["single_qubit"][qubit]["sweetspot"]
    
    platform.qf_port[fluxline].current = qubit_biasing_current
    
    current_range = np.arange(current_min, current_max, current_step)
    
    for i in range(len(current_range)):
        if qubit_biasing_current >= current_range[i]:
            start = i
            break
    
    start_f = resonator_frequency
    
    #platform.ro_port[qubit].attenuation = att
    
    background = [start_f+1e7, start_f-1e7]
    noise = get_noise(background, platform, ro_pulse, qubit, sequence)
    
    freqs1 = []
    best_f = start_f
    for curr in current_range[start:]:
        best_msr = noise
        platform.qf_port[fluxline].current = curr
        for span in spans:
            best_f, best_msr = scan_level(best_f, best_msr, max_runs, thr, span, resolution, noise, platform, ro_pulse, qubit, sequence)
        for span in small_spans:
            best_f, best_msr = scan_small(best_f, best_msr, max_runs, thr, span, 11, noise, platform, ro_pulse, qubit, sequence)
        freqs1.append(best_f)
        
    freqs2 = []
    best_f = start_f
    for curr in reversed(current_range[:start]):
        best_msr = noise
        platform.qf_port[fluxline].current = curr
        for span in spans:
            best_f, best_msr = scan_level(best_f, best_msr, max_runs, thr, span, resolution, noise, platform, ro_pulse, qubit, sequence)
        for span in small_spans:
            best_f, best_msr = scan_small(best_f, best_msr, max_runs, thr, span, 11, noise, platform, ro_pulse, qubit, sequence)
        freqs2.append(best_f)
        
    freqs = np.array(list(reversed(freqs2))+freqs1)
    
    for i in range(len(freqs)):
        results = {
                    "MSR[V]": 455,
                    "i[V]": 455,
                    "q[V]": 455,
                    "phase[rad]": 455,
                    "frequency[Hz]": freqs[i],
                    "current[A]": current_range[i],
                }
        data.add(results)
    
    sweet_freq = np.max(freqs)
    sweet_curr = current_range[np.argmax(freqs)]
    
    plot_flux(current_range, freqs, qubit)
    
    print(f'For qubit {qubit}:')
    print(f'Sweet spot found at frequency {sweet_freq} Hz at current value of {sweet_curr} A.\n')
    
    yield data
                         
        
    
    
    
    
    
    
