# -*- coding: utf-8 -*-
import numpy as np


def variable_resolution_scanrange(
    lowres_width, lowres_step, highres_width, highres_step
):
    """Helper function for sweeps."""
    return np.concatenate(
        (
            np.arange(-lowres_width, -highres_width, lowres_step),
            np.arange(-highres_width, highres_width, highres_step),
            np.arange(highres_width, lowres_width, lowres_step),
        )
    )


def snr(signal, noise):
    """Signal to Noise Ratio to detect peaks and valleys."""
    return 20*np.log(signal/noise)


def choose_freq(freq, span, resolution):
    """Choose a new frequency gaussianly distributed around initial one."""
    g = np.random.normal(0, span/10, 1)
    f = np.linspace(-span/2, span/2, resolution)
    for ff in f:
        if g <= ff:
            break
    return freq+ff


def get_noise(background, platform, ro_pulse, qubit, sequence):
    noise = 0
    for b_freq in background:
        platform.ro_port[qubit].lo_frequency = b_freq - ro_pulse.frequency
        res = platform.execute_pulse_sequence(sequence, 1024)
        msr, phase, i, q = platform.execute_pulse_sequence(sequence)[ro_pulse.serial]
        noise += msr
    return noise/len(background)


def update_f(best_f, f, best_v, v, noise, thr):
    """Updates the best frequency and voltage found if SNR is above a threshold."""
    if abs(snr(v, noise)) >= thr:
        best_v = v
        best_f = f
    return best_f, best_v


def plot_flux(currs, freqs, qubit):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,8))
    plt.plot(currs, freqs)
    plt.savefig(f'qubit_{qubit}_flux.png', bbox_inches='tight')
    
    
def plot_punchout(atts, freqs, qubit):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,8))
    plt.plot(freqs, atts)
    plt.savefig(f'qubit_{qubit}_punchout.png', bbox_inches='tight')