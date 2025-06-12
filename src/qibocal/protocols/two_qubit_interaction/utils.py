from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from qibolab import Platform

from qibocal.auto.operation import Data, QubitId, QubitPairId
from qibocal.config import log

from ..utils import fallback_period, guess_period
from .virtual_z_phases import fit_sinusoid, phase_diff


def order_pair(pair: QubitPairId, platform: Platform) -> tuple[QubitId, QubitId]:
    """Order a pair of qubits by drive frequency."""
    q0, q1 = pair

    drive0 = platform.config(platform.qubits[q0].drive)
    drive1 = platform.config(platform.qubits[q1].drive)
    return (q1, q0) if drive0.frequency > drive1.frequency else (q0, q1)


def fit_flux_amplitude(matrix, amps, times):
    """Estimate amplitude for CZ gate.


    Given the pattern of a chevron plot (see for example Fig. 2 here
    https://arxiv.org/pdf/1907.04818.pdf). This function estimates
    the CZ amplitude by finding the amplitude which gives the standard
    deviation, indicating that there are oscillation along the z axis.

    Args:
     matrix (np.ndarray): signal matrix
     amps (np.ndarray): amplitudes swept
     times (np.ndarray): duration swept

    Returns:
     amplitude (float): estimated amplitude
     index (int): amplitude index
     delta (float):  omega for estimated amplitude
    """

    size_amp = len(amps)
    fs = []
    std = []
    for i in range(size_amp):
        y = matrix[i, :]
        period = fallback_period(guess_period(times, y))
        fs.append(1 / period)
        std.append(np.std(y))

    amplitude = amps[np.argmax(std)]
    delta = fs[np.argmax(std)]
    index = int(np.where(np.unique(amps) == amplitude)[0])
    return amplitude, index, delta


def fit_snz_optimize(
    data: Data,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Repetition of correct virtual phase fit for all configurations."""
    fitted_parameters = {}
    pairs = data.pairs
    virtual_phases = {}
    angles = {}
    leakages = {}
    # FIXME: experiment should be for single pair
    for pair in pairs:
        for amplitude in data.amplitudes[pair]:
            for target, control, setup, foo_parameter in data[pair]:
                selected_data = data[pair][target, control, setup, foo_parameter]
                target_data = selected_data.prob_target[selected_data.amp == amplitude,]
                try:
                    params = fit_sinusoid(
                        np.array(data.swept_virtual_phases),
                        target_data,
                        gate_repetition=1,
                    )
                    fitted_parameters[
                        target, control, setup, amplitude, foo_parameter
                    ] = params
                except Exception as e:
                    log.warning(f"Fit failed for pair ({target, control}) due to {e}.")

            for target, control, setup, foo_parameter in data[pair]:
                if setup == "I":  # The loop is the same for setup I or X
                    angles[target, control, amplitude, foo_parameter] = phase_diff(
                        fitted_parameters[
                            target, control, "X", amplitude, foo_parameter
                        ][2],
                        fitted_parameters[
                            target, control, "I", amplitude, foo_parameter
                        ][2],
                    )
                    virtual_phases[target, control, amplitude, foo_parameter] = (
                        fitted_parameters[
                            target, control, "I", amplitude, foo_parameter
                        ][2]
                    )

                    # leakage estimate: L = m /2
                    # See NZ paper from Di Carlo
                    # approximation which does not need qutrits
                    # https://arxiv.org/pdf/1903.02492.pdf
                    data_x = data[pair][target, control, "X", foo_parameter]
                    data_i = data[pair][target, control, "I", foo_parameter]
                    leakages[target, control, amplitude, foo_parameter] = 0.5 * np.mean(
                        data_x[data_x.amp == amplitude].prob_control
                        - data_i[data_i.amp == amplitude].prob_control
                    )
    return virtual_phases, fitted_parameters, leakages, angles
