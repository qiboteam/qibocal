import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal.calibrations.characterization.utils import choose_freq, get_noise, snr
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import (
    res_spectrocopy_flux_fit,
    res_spectrocopy_flux_matrix,
)
from qibocal.plots import frequency_attenuation, frequency_current_flux


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
        platform (AbstractPlatform): Platform the experiment is executed on.
        ro_pulse (ReadoutPulse): Used in order to execute the pulse sequence with the right parameters in the right qubit.
        qubit (int): qubit coupled to the resonator that we are probing.
        sequence (PulseSequence):

    Returns:
        best_f (float): Best frequency found for the feature.
        best_msr (float): MSR found for the feature.
        phase (float): Phase found for the feature.
        i (float): In-phase signal
        q (float): Quadrature signal.

    """
    freq = best_f
    for _ in range(max_runs):
        platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
        msr, phase, i, q = platform.execute_pulse_sequence(sequence)[ro_pulse.serial]
        if abs(snr(msr, noise)) >= thr:
            msr1 = msr
            if platform.resonator_type == "3D":
                msr = -msr
                best_msr = -best_msr
            if msr < best_msr:
                best_f, best_msr = freq, msr1
                return best_f, best_msr, phase, i, q
        freq = choose_freq(best_f, span, resolution)
    return best_f, best_msr, phase, i, q


def scan_small(best_f, best_msr, span, resolution, platform, ro_pulse, qubit, sequence):
    """Small scan around the found feature to fine-tune the measurement up to given precision.

    Args:
        best_f (int): Best found frequency in previous scan. Used as starting point.
        best_msr (float): MSR found for the previous best frequency. Used to check if a better value is found.
        span (int): Search space around previous best value where the next frequency is sampled.
        resolution (int): How many points are taken in the span. Taken as 10 for the small scan.
        platform (AbstractPlatform): Platform the experiment is executed on.
        ro_pulse (ReadoutPulse): Used in order to execute the pulse sequence with the right parameters in the right qubit.
        qubit (int): qubit coupled to the resonator that we are probing.
        sequence (PulseSequence):

    Returns:
        best_f (float): Best frequency found for the feature.
        best_msr (float): MSR found for the feature.
        phase (float): Phase found for the feature.
        i (float): In-phase signal.
        q (float): Quadrature signal.

    """
    start_f = best_f
    scan = np.linspace(-span / 2, span / 2, resolution)
    for s in scan:
        freq = start_f + s
        platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
        msr, phase, i, q = platform.execute_pulse_sequence(sequence)[ro_pulse.serial]
        msr1 = msr
        if platform.resonator_type == "3D":
            msr = -msr
            best_msr = -best_msr
        if msr < best_msr:
            best_f, best_msr = freq, msr1
    return best_f, best_msr, phase, i, q


@plot("Frequency vs Attenuation", frequency_attenuation)
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
    points,
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
        points (int): Number of points plotted at a time in qq-live.

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
    for k, att in enumerate(attenuation_range):
        platform.ro_port[qubit].attenuation = att

        background = [best_f + 1e7, best_f - 1e7]
        noise = get_noise(background, platform, ro_pulse, qubit, sequence)
        best_msr = noise
        for span in spans:
            best_f, best_msr, phase, i, q = scan_level(
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
            )
        for span in small_spans:
            best_f, best_msr, phase, i, q = scan_small(
                best_f, best_msr, span, 11, platform, ro_pulse, qubit, sequence
            )
        results = {
            "MSR[V]": best_msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "frequency[Hz]": best_f,
            "attenuation[dB]": att,
        }
        data.add(results)
        if k % points == 0:
            yield data
        if att >= opt_att:
            if abs(snr(best_msr, noise)) > opt_snr:
                opt_snr = abs(snr(best_msr, noise))
                opt_att = att
                opt_f = best_f
    data1 = DataUnits(
        name=f"results_q{qubit}",
        quantities={"snr": "dimensionless", "frequency": "Hz", "attenuation": "dB"},
    )
    f_err = len(str(int(small_spans[-1] / 10)))
    results = {
        "snr[dimensionless]": opt_snr,
        "frequency[Hz]": round(opt_f, -f_err),
        "attenuation[dB]": opt_att,
    }
    data1.add(results)
    yield data1
    yield data


@plot("Frequency vs Current", frequency_current_flux)
def resonator_flux_sample(
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
    params_fit,
    points,
    matrix,
):
    """Use gaussian samples to extract the flux-frequency response of the resonator for different values of current.

    Args:
        platform (AbstractPlatform): Platform the experiment is executed on.
        qubit (int): qubit coupled to the resonator that we are probing.
        current_min (float): minimum current value where the experiment starts.
        current_max (float): maximum current reached in the scan.
        current_step (float): change in current after every step.
        fluxlines (list): ids of the current lines to use for the experiment. Use qubit if matching the qubit id.
        max_runs (int): Maximum amount of tries before stopping if feature is not found.
        thr (float): SNR value used as threshold for detection of the feature.
        spans (list): Different spans to search for the feature at different precisions.
        small_spans (list): Different spans for the small sweeps to fine-tune the feature.
        resolution (int): How many points are taken in the span for the scan_level() function.
        params_fit (dic): Dictionary of parameters for the fit. {qubit : [freq_rh, g, Ec, Ej], ... }.
                          freq_rh is the resonator frequency at high power and g in the readout coupling.
                          If Ec and Ej are missing, the fit is valid in the transmon limit and if they are indicated,
                          contains the next-order correction.
        points (int): Number of points plotted at a time in qq-live.

    Returns:
        data (Data): Data file with information on the feature response at each current point.

    """
    if matrix == True:
        import os

        def all_subdirs_of(b="."):
            result = []
            for d in os.listdir(b):
                bd = os.path.join(b, d)
                if os.path.isdir(bd):
                    result.append(bd)
            return result

        folder = max(all_subdirs_of("reports"), key=os.path.getmtime)

    params_fit = params_fit[qubit]
    for fluxline in fluxlines:
        data = DataUnits(
            name=f"data_q{qubit}_f{fluxline}",
            quantities={"frequency": "Hz", "current": "A"},
        )

        if fluxline == "qubit":
            fluxline = qubit

        platform.reload_settings()
        for f in fluxlines:
            platform.qf_port[f].current = 0

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
        current_range = (
            np.arange(current_min, current_max, current_step) + qubit_biasing_current
        )
        start = next(
            (
                index
                for index, curr in enumerate(current_range)
                if curr >= qubit_biasing_current
            )
        )
        start_f = resonator_frequency
        background = [start_f + 1e7, start_f - 1e7]
        noise = get_noise(background, platform, ro_pulse, qubit, sequence)
        current_range = np.concatenate(
            (current_range[start:], current_range[:start][::-1])
        )
        index = len(current_range[start:])
        best_f = start_f
        for k, curr in enumerate(current_range):
            if k == index:
                best_f = start_f
            best_msr = noise
            platform.qf_port[fluxline].current = curr
            for span in spans:
                best_f, best_msr, phase, i, q = scan_level(
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
                )
            for span in small_spans:
                best_f, best_msr, phase, i, q = scan_small(
                    best_f, best_msr, span, 11, platform, ro_pulse, qubit, sequence
                )
            results = {
                "MSR[V]": best_msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": best_f,
                "current[A]": curr,
            }
            data.add(results)
            if k % points == 0:
                yield data
        yield res_spectrocopy_flux_fit(
            data,
            x="current[A]",
            y="frequency[Hz]",
            qubit=qubit,
            fluxline=fluxline,
            params_fit=params_fit,
        )

        if matrix == True:
            yield res_spectrocopy_flux_matrix(folder, fluxlines)

        yield data
