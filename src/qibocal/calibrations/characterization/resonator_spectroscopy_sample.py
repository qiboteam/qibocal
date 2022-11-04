# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.calibrations.characterization.utils import (
    choose_freq,
    get_noise,
    plot_flux,
    plot_punchout,
    snr,
)
from qibocal.data import DataUnits
from qibocal.fitting.methods import res_spectrocopy_flux_fit
from qibocal.decorators import plot

def scan_level(
    best_f,
    best_msr,
    max_runs,
    thr,
    span,
    resolution,
    noise,
    platform,
    ro_pulse,
    qubit,
    sequence,
    software_averages
):
    """Scans for the feature by sampling a gaussian near the previously found best point. Decides if the freature
    is found by checking if the snr against background noise is above a threshold.

    Args:
        best_f (int): Best found frequency in previous scan. Used as starting point.
        best_msr (float): MSR found for the previous best frequency. Used to check if a better value is found.
        max_runs (int): Maximum amount of tries before stopping if feature is not found.
        thr (float): SNR value used as threshold for detection of the feature.
        span (int): Search space around previous best value where the next frequency is sampled.
        resolution (int): How many points are taken in the span.
        noise (float): MSR value for the background noise.
        platform ():
        ro_pulse (): Used in order to execute the pulse sequence with the right parameters in the right qubit.
        qubit (int): TODO: might be useful to make this parameters implicit and not given.
        sequence ():
        software_averages (int):

    Returns:
        best_f (int): New best frequency found for the feature.
        best_msr (float): MSR found for the feature.

    """
    for _ in range(max_runs):
        if _ == 0:
            freq = best_f
        else:
            freq = choose_freq(best_f, span, resolution)
        platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
        avg=np.zeros(software_averages)
        for j in range(len(avg)):
            avg[j], phase, i, q = platform.execute_pulse_sequence(sequence)[ro_pulse.serial]
        msr=np.mean(avg)
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


def scan_small(
    best_f, best_msr, span, resolution, platform, ro_pulse, qubit, sequence, software_averages
):
    """Small scan around the found feature to fine-tune the measurement up to given precision.

    Args:
        best_f (int): Best found frequency in previous scan. Used as starting point.
        best_msr (float): MSR found for the previous best frequency. Used to check if a better value is found.
        span (int): Search space around previous best value where the next frequency is sampled.
        resolution (int): How many points are taken in the span. Taken as 10 for the small scan.
        platform ():
        ro_pulse (): Used in order to execute the pulse sequence with the right parameters in the right qubit.
        qubit (int): TODO: might be useful to make this parameters implicit and not given.
        sequence ():
        software_averages (int):

    Returns:
        best_f (int): New best frequency found for the feature.
        best_msr (float): MSR found for the feature.

    """
    start_f = best_f
    scan = np.linspace(-span / 2, span / 2, resolution)
    for s in scan:
        freq = start_f + s
        platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
        avg=np.zeros(software_averages)
        for j in range(len(avg)):
            avg[j], phase, i, q = platform.execute_pulse_sequence(sequence)[ro_pulse.serial]
        msr=np.mean(avg)
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
    resolution,
    software_averages
):
    """Use gaussian samples to extract the punchout of the resonator for different values of attenuation.

    Args:
        platform (AbstractPlatform): Platform the experiment is executed on.
        qubit (int): qubit coupled to the resonator that we are probing.
        min_att (int): minimum attenuation value where the experiment starts. Less attenuation -> more power.
        max_att (int): maximum attenuation reached in the scan.
        step_att (int): change in attenuation after every step.
        max_runs (int): Maximum amount of tries before stopping if feature is not found.
        thr (float): SNR value used as threshold for detection of the feature.
        spans (list): Different spans to search for the feature at different precisions.
        small_spans (list): Different spans for the small sweeps to fine-tune the feature.
        resolution (int): How many points are taken in the span for the scan_level() function.
        software_averages (int):

    Returns:
        data (Data): Data file with information on the feature response at each attenuation point.

    """

    data = DataUnits(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "attenuation": "dB"}
    )

    platform.reload_settings()
    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    attenuation_range = np.arange(min_att, max_att, step_att)
    best_f = resonator_frequency

    opt_att = 30
    opt_snr = 0
    freqs = []
    for att in attenuation_range:
        platform.ro_port[qubit].attenuation = att

        background = [best_f + 1e7, best_f - 1e7]
        noise = get_noise(background, platform, ro_pulse, qubit, sequence)
        best_msr = noise
        for span in spans:
            best_f, best_msr = scan_level(
                best_f,
                best_msr,
                max_runs,
                thr,
                span,
                resolution,
                noise,
                platform,
                ro_pulse,
                qubit,
                sequence,
                software_averages
            )
        for span in small_spans:
            best_f, best_msr = scan_small(
                best_f, best_msr, span, 11, platform, ro_pulse, qubit, sequence, software_averages
            )
        # freqs.append(best_f)
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

    print(f"For qubit {qubit}:")
    print(
        f"Best response found at frequency {opt_f} Hz for attenuation value of {opt_att} dB.\n"
    )

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
    software_averages
):
    """Use gaussian samples to extract the flux-frequency response of the resonator for different values of current.

    Args:
        platform (AbstractPlatform): Platform the experiment is executed on.
        qubit (int): qubit coupled to the resonator that we are probing.
        current_min (float): minimum current value where the experiment starts.
        current_max (float): maximum current reached in the scan.
        current_step (float): change in current after every step.
        fluxline (int or "qubit"): id of the current line to use for the experiment. Use qubit if matching the qubit id.
        max_runs (int): Maximum amount of tries before stopping if feature is not found.
        thr (float): SNR value used as threshold for detection of the feature.
        spans (list): Different spans to search for the feature at different precisions.
        small_spans (list): Different spans for the small sweeps to fine-tune the feature.
        resolution (int): How many points are taken in the span for the scan_level() function.
        software_averages (int):

    Returns:
        data (Data): Data file with information on the feature response at each current point.

    """

    data = DataUnits(
        name=f"data_q{qubit}_f{fluxline}", quantities={"frequency": "Hz", "current": "A"}
    )

    if fluxline == "qubit":
        fluxline = qubit

    platform.reload_settings()

    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]
    qubit_biasing_current = platform.characterization["single_qubit"][qubit][
        "sweetspot"
    ]

    platform.qf_port[fluxline].current = qubit_biasing_current

    current_range = np.arange(current_min, current_max, current_step)

    for i in range(len(current_range)):
        if qubit_biasing_current >= current_range[i]:
            start = i
            break

    start_f = resonator_frequency

    background = [start_f + 1e7, start_f - 1e7]
    noise = get_noise(background, platform, ro_pulse, qubit, sequence)

    # We scan starting from the sweet spot to higher currents
    freqs1 = []
    best_f = start_f
    for curr in current_range[start:]:
        best_msr = noise
        platform.qf_port[fluxline].current = curr
        for span in spans:
            best_f, best_msr = scan_level(
                best_f,
                best_msr,
                max_runs,
                thr,
                span,
                resolution,
                noise,
                platform,
                ro_pulse,
                qubit,
                sequence,
                software_averages
            )
        for span in small_spans:
            best_f, best_msr = scan_small(
                best_f, best_msr, span, 11, platform, ro_pulse, qubit, sequence, software_averages
            )
        freqs1.append(best_f)

    # and continue starting from the sweet spot to lower currents.
    freqs2 = []
    best_f = start_f
    for curr in reversed(current_range[:start]):
        best_msr = noise
        platform.qf_port[fluxline].current = curr
        for span in spans:
            best_f, best_msr = scan_level(
                best_f,
                best_msr,
                max_runs,
                thr,
                span,
                resolution,
                noise,
                platform,
                ro_pulse,
                qubit,
                sequence,
                software_averages
            )
        for span in small_spans:
            best_f, best_msr = scan_small(
                best_f, best_msr, span, 11, platform, ro_pulse, qubit, sequence, software_averages
            )
        freqs2.append(best_f)

    freqs = np.array(list(reversed(freqs2)) + freqs1)

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

    plot_flux(current_range, freqs, qubit, fluxline)

    print(f"For qubit {qubit}:")
    print(
        f"Sweet spot found at frequency {sweet_freq} Hz at current value of {sweet_curr} A.\n"
    )

    yield data

@plot("Frequency vs Current", plots.frequency_current_flux)
def resonator_flux_sample_matrix(
    platform: AbstractPlatform,
    qubit: int,
    current_min,
    current_max,
    current_step,
    fluxlines,
    max_runs,
    thr,
    spans,
    small_spans,
    resolution,
    software_averages
):  
    """Use gaussian samples to extract the flux-frequency response of the resonator for different values of current.

    Args:
        platform (AbstractPlatform): Platform the experiment is executed on.
        qubit (int): qubit coupled to the resonator that we are probing.
        current_min (float): minimum current value where the experiment starts.
        current_max (float): maximum current reached in the scan.
        current_step (float): change in current after every step.
        fluxlines (list): ids of the current lines to use for the experiment.
        max_runs (int): Maximum amount of tries before stopping if feature is not found.
        thr (float): SNR value used as threshold for detection of the feature.
        spans (list): Different spans to search for the feature at different precisions.
        small_spans (list): Different spans for the small sweeps to fine-tune the feature.
        resolution (int): How many points are taken in the span for the scan_level() function.
        software_averages (int):

    Returns:
        data (Data): Data file with information on the feature response at each current point.

    """
    for fluxline in fluxlines:
        fluxline=int(fluxline)
        data = DataUnits(
        name=f"data_q{qubit}_f{fluxline}", quantities={"frequency": "Hz", "current": "A"}
        )

        if fluxline == "qubit":
            fluxline = qubit

        platform.reload_settings()

        sequence = PulseSequence()
        ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulse)

        resonator_frequency = platform.characterization["single_qubit"][qubit][
            "resonator_freq"
        ]
        qubit_biasing_current = platform.characterization["single_qubit"][qubit][
            "sweetspot"
        ]

        platform.qf_port[fluxline].current = qubit_biasing_current

        current_range = np.arange(current_min, current_max, current_step)

        for i in range(len(current_range)):
            if qubit_biasing_current >= current_range[i]:
                start = i
                break

        start_f = resonator_frequency

        background = [start_f + 1e7, start_f - 1e7]
        noise = get_noise(background, platform, ro_pulse, qubit, sequence)

        # We scan starting from the sweet spot to higher currents
        freqs1 = []
        best_f = start_f
        for curr in current_range[start:]:
            best_msr = noise
            platform.qf_port[fluxline].current = curr
            for span in spans:
                best_f, best_msr = scan_level(
                    best_f,
                    best_msr,
                    max_runs,
                    thr,
                    span,
                    resolution,
                    noise,
                    platform,
                    ro_pulse,
                    qubit,
                    sequence,
                    software_averages
                )
            for span in small_spans:
                best_f, best_msr = scan_small(
                    best_f, best_msr, span, 11, platform, ro_pulse, qubit, sequence, software_averages
                )
            freqs1.append(best_f)

        # and continue starting from the sweet spot to lower currents.
        freqs2 = []
        best_f = start_f
        for curr in reversed(current_range[:start]):
            best_msr = noise
            platform.qf_port[fluxline].current = curr
            for span in spans:
                best_f, best_msr = scan_level(
                    best_f,
                    best_msr,
                    max_runs,
                    thr,
                    span,
                    resolution,
                    noise,
                    platform,
                    ro_pulse,
                    qubit,
                    sequence,
                    software_averages
                )
            for span in small_spans:
                best_f, best_msr = scan_small(
                    best_f, best_msr, span, 11, platform, ro_pulse, qubit, sequence, software_averages
                )
            freqs2.append(best_f)

        freqs = np.array(list(reversed(freqs2)) + freqs1)

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
        
        yield data
        
        if qubit==fluxline:
            labels=["sweet_curr", "sweet_curr_err", "sweet_freq", "sweet_freq_err", "C_ii", "C_ii_err", "freq_offset", "freq_offset_err"]
        else:
            labels=["C_ij", "C_ij_err"]

        yield res_spectrocopy_flux_fit(
                    data,
                    x="current[A]",
                    y="frequency[Hz]",
                    qubit=qubit,
                    fluxline=fluxline,
                    labels=labels,
                )

        yield data