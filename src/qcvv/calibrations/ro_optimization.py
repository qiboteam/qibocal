# -*- coding: utf-8 -*-
from unittest import result

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qcvv import plots
from qcvv.calibrations.utils import get_fidelity
from qcvv.data import Dataset
from qcvv.decorators import plot


@plot("Fidelity vs Frequency", plots.frequency_fidelity)
def ro_frequency_optimization(
    platform: AbstractPlatform, qubit, freq_width, freq_step, niter, points=5
):
    platform.reload_settings()
    platform.qrm[qubit].ports["i1"].hardware_demod_en = True

    freqrange = (
        np.arange(-freq_width / 2, freq_width / 2, freq_step)
        + platform.ro_port[qubit].lo_frequency
    )

    data = Dataset(
        name=f"data_q{qubit}",
        quantities={"frequency": "Hz", "fidelity": "dimensionless"},
    )

    count = 0
    for freq in freqrange:
        if count % points == 0:
            yield data
        platform.ro_port[qubit].lo_frequency = freq
        fidelity = get_fidelity(
            platform, qubit, niter, param={"frequency[Hz]": freq}, save=False
        )
        results = {
            "fidelity[dimensionless]": np.array(fidelity * 1.0),
            "frequency[Hz]": freq,
        }
        data.add(results)
        count += 1
    yield data
