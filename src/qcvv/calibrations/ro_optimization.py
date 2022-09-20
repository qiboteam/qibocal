# -*- coding: utf-8 -*-
from unittest import result

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qcvv import plots
from qcvv.calibrations.utils import check_frequency, get_fidelity
from qcvv.data import Dataset
from qcvv.decorators import plot


@plot("Fidelity vs Frequency", plots.frequency_fidelity)
def ro_frequency_optimization(
    platform: AbstractPlatform, qubit, freq_width, freq_step, niter, points=5
):
    """
    Optimize the Read-out frequency to get best fidelities

    Params:
    freq_width: width around the resonator's frequency to set the sweeping range
    freq_step: step size to vary the frequency
    niter: number of shots used in the binning to calculate the fidelity

    Return:
    data: fidelity as a function of RO frequency
    """
    platform.reload_settings()
    check_frequency(platform, write=True)
    platform.qrm[qubit].ports["i1"].hardware_demod_en = True

    freqrange = (
        np.arange(-freq_width / 2, freq_width / 2, freq_step)
        + platform.characterization["single_qubit"][qubit]["resonator_freq"]
    )

    data = Dataset(
        name=f"data_q{qubit}",
        quantities={"frequency": "Hz", "fidelity": "dimensionless"},
    )

    count = 0
    for freq in freqrange:
        if count % points == 0:
            yield data
        platform.ro_port[qubit].lo_frequency = (
            freq - platform.create_qubit_readout_pulse(qubit, start=0).frequency
        )
        fidelity = get_fidelity(
            platform, qubit, niter, param={"frequency[Hz]": freq}, save=True
        )
        results = {
            "fidelity[dimensionless]": np.array(fidelity * 1.0),
            "frequency[Hz]": freq,
        }
        data.add(results)
        count += 1
    yield data


@plot("Fidelity vs Frequency", plots.power_fidelity)
def ro_power_optimization(
    platform: AbstractPlatform,
    qubit,
    amplitude_min,
    amplitude_step,
    amplitude_max,
    niter,
    points=5,
):
    """
    Maximize the Read-out power to get best fidelities

    Params:
    amplitude_min: starting amplitude of the RO pulse
    amplitude_max: ending amplitude of the RO pulse
    amplitude_step: step size to vary the amplitude
    niter: number of shots used in the binning to calculate the fidelity

    Return:
    data: fidelity as a function of amplitude
    """
    platform.reload_settings()
    check_frequency(platform, write=True)
    platform.qrm[qubit].ports["i1"].hardware_demod_en = True
    platform.ro_port[qubit].attenuation = (
        platform.ro_port[qubit].attenuation - 2
    )  # to be able to increase power

    amplitudes = np.arange(amplitude_min, amplitude_max, amplitude_step)

    data = Dataset(
        name=f"data_q{qubit}",
        quantities={"amplitude": "dimensionless", "fidelity": "dimensionless"},
    )

    count = 0
    for amp in amplitudes:
        if count % points == 0:
            yield data
        fidelity = get_fidelity(
            platform,
            qubit,
            niter,
            param={"amplitude[dimensionless]": amp},
            save=True,
            amplitude_ro_pulse=amp,
        )
        results = {
            "fidelity[dimensionless]": np.array(fidelity * 1.0),
            "amplitude[dimensionless]": amp,
        }
        data.add(results)
        count += 1
    yield data
