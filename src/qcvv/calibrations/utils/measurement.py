# -*- coding: utf-8 -*-
import pathlib
from turtle import update

import numpy as np
import qibolab
import yaml
from qibo.config import log, raise_error
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from scipy.optimize import curve_fit

from qcvv import plots
from qcvv.data import Dataset
from qcvv.decorators import plot


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


def get_fidelity(platform: AbstractPlatform, qubit, niter, param=None, save=True):
    """
    Returns the read-out fidelity for the measurement.

    Param:
    platform: Qibolab platform for QPU
    qubit: Qubit number under investigation
    niter: number of iterations
    param: name and units of the varied parameters to save the data in a dictionary format {"name[PintUnit]": vals, ...}
    save: bool to save the data or not

    Returns:
    fidelity: float C [0,1]
    """
    if save:
        if param is None:
            raise_error(
                ValueError,
                "Please provide the varied parameters in a dict of QCVV type",
            )

    platform.qrm[qubit].ports[
        "i1"
    ].hardware_demod_en = True  # binning only works with hardware demodulation enabled
    # create exc sequence
    exc_sequence = PulseSequence()
    RX_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX_pulse.duration)
    exc_sequence.add(RX_pulse)
    exc_sequence.add(ro_pulse)

    param_dict = {}
    for key in param:
        param_dict[key.split("[")[0]] = key.split("[")[1].replace("]", "")
    quantities = {"iteration": "dimensionless"}
    quantities.update(param_dict)
    data_exc = Dataset(name=f"data_exc_{param}_q{qubit}", quantities=quantities)
    shots_results = platform.execute_pulse_sequence(exc_sequence, nshots=niter)[
        "shots"
    ][ro_pulse.serial]
    for n in np.arange(niter):
        msr, phase, i, q = shots_results[n]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "iteration[dimensionless]": n,
        }
        results.update(param)
        data_exc.add(results)
    if save:
        print("save")

    gnd_sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    gnd_sequence.add(ro_pulse)

    data_gnd = Dataset(name=f"data_gnd_q{qubit}", quantities=quantities)

    shots_results = platform.execute_pulse_sequence(gnd_sequence, nshots=niter)[
        "shots"
    ][ro_pulse.serial]
    for n in np.arange(niter):
        msr, phase, i, q = shots_results[n]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "iteration[dimensionless]": n,
        }
        results.update(param)
        data_gnd.add(results)
    if save:
        print("save")

    exc, gnd = rotate_to_distribution(data_exc, data_gnd)

    exc_cumul = np.histogram(exc, bins=int(niter / 50))
    exc_cumul = np.array([exc_cumul[0], exc_cumul[1][1:]])
    exc_cumul[0, :] = np.cumsum(exc_cumul[0, :])
    exc_cumul[0, :] = exc_cumul[0, :] / max(exc_cumul[0, :])
    gnd_cumul = np.histogram(gnd, bins=int(niter / 50))
    gnd_cumul = np.array([gnd_cumul[0], gnd_cumul[1][1:]])
    gnd_cumul[0, :] = np.cumsum(gnd_cumul[0, :])
    gnd_cumul[0, :] = gnd_cumul[0, :] / max(gnd_cumul[0, :])

    thres = (
        gnd_cumul[1, :][np.argmin(abs(gnd_cumul[0, :] - 0.5))]
        + exc_cumul[1, :][np.argmin(abs(exc_cumul[0, :] - 0.5))]
    ) / 2

    fidelity = (
        gnd_cumul[0, np.argmin(abs(gnd_cumul[1, :] - thres))]
        - exc_cumul[0, np.argmin(abs(exc_cumul[1, :] - thres))]
    )

    return fidelity


def rotate_to_distribution(data_exc, data_gnd):
    iq_exc = (
        data_exc.get_values("i", "V").to_numpy()
        + 1j * data_exc.get_values("q", "V").to_numpy()
    )
    iq_gnd = (
        data_gnd.get_values("i", "V").to_numpy()
        + 1j * data_gnd.get_values("q", "V").to_numpy()
    )

    origin = np.mean(iq_gnd)
    iq_gnd = iq_gnd - origin
    iq_exc = iq_exc - origin
    angle = np.angle(np.mean(iq_exc))
    iq_exc = iq_exc * np.exp(-1j * angle) + origin
    iq_gnd = iq_gnd * np.exp(-1j * angle) + origin

    return np.real(iq_exc), np.real(iq_gnd)
