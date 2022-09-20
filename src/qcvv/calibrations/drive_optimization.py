# -*- coding: utf-8 -*-
from unittest import result

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qcvv import plots
from qcvv.calibrations.utils import check_frequency, get_fidelity
from qcvv.data import Dataset
from qcvv.decorators import plot


@plot("Fidelity vs Frequency", plots.power_fidelity)
def drive_power_optimization(
    platform: AbstractPlatform, qubit, amplitude_offset, amplitude_step, niter, points=5
):
    """
    This routine is meant to check if the RX amplitude is correct. It is not super accurate, though I am not sure how much would the gate fidelity
    vary with the amplitude. It is kept here for now. It is a first step before ALLXY.

    Params:
    amplitude_offset: value that sets a symetric amplitude range around the RX pulse amplitude
    amplitude_step: step size to vary the amplitude
    niter: number of shots used in the binning to calculate the fidelity

    Return:
    data: fidelity as a function of amplitude
    """
    platform.reload_settings()
    check_frequency(platform, write=True)
    platform.qrm[qubit].ports["i1"].hardware_demod_en = True

    amplitudes = (
        np.arange(-amplitude_offset, amplitude_offset, amplitude_step)
    ) + platform.create_RX_pulse(qubit, start=0).amplitude

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
            amplitude_qd_pulse=amp,
        )
        results = {
            "fidelity[dimensionless]": np.array(fidelity * 1.0),
            "amplitude[dimensionless]": amp,
        }
        data.add(results)
        count += 1
    yield data
