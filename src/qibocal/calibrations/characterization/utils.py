# -*- coding: utf-8 -*-
import glob
import os
import pathlib

import numpy as np
from qibo.config import raise_error
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal.data import Dataset


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


def get_latest_datafolder():
    cwd = pathlib.Path()
    list_dir = sorted(glob.glob(os.path.join(cwd, "*/")), key=os.path.getmtime)
    for i in range(len(list_dir)):
        if os.path.isdir(cwd / list_dir[-i] / "data"):
            return cwd / list_dir[-i]


def get_fidelity(
    platform: AbstractPlatform,
    qubit,
    niter,
    param=None,
    save=True,
    amplitude_ro_pulse=0.9,
    amplitude_qd_pulse=None,
):
    """
    Returns the read-out fidelity for the measurement.
    Param:
    platform: Qibolab platform for QPU
    qubit: Qubit number under investigation
    niter: number of iterations
    param: name and units of the varied parameters to save the data in a dictionary format {"name[PintUnit]": vals, ...}
    save: bool to save the data or not
    optional parameters for designed routines: #FIXME: find a better way to do it, extracting pulses from sequence?
        amplitude_ro_pulse = 0.9 #Not 1 for a reason I forgot
    Returns:
    fidelity: float C [0,1]
    """
    if save:
        if param is None:
            raise_error(
                ValueError,
                "Please provide the varied parameters in a dict of QCVV type",
            )
        path = get_latest_datafolder() / "data_param"
        os.makedirs(path, exist_ok=True)

    platform.qrm[qubit].ports[
        "i1"
    ].hardware_demod_en = True  # binning only works with hardware demodulation enabled
    # create exc sequence
    exc_sequence = PulseSequence()
    RX_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX_pulse.duration)
    ro_pulse.amplitude = amplitude_ro_pulse
    if amplitude_qd_pulse is not None:
        RX_pulse.amplitude = amplitude_qd_pulse
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
        data_exc.to_csv(path)

    gnd_sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    ro_pulse.amplitude = amplitude_ro_pulse
    gnd_sequence.add(ro_pulse)

    data_gnd = Dataset(name=f"data_gnd_{param}_q{qubit}", quantities=quantities)

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
        data_gnd.to_csv(path)

    real_values_exc, real_values_gnd = rotate_to_distribution(data_exc, data_gnd)

    real_values_combined = real_values_exc + real_values_gnd
    real_values_combined.sort()

    cum_distribution_exc = [
        sum(map(lambda x: x.real >= real_value, real_values_exc))
        for real_value in real_values_combined
    ]
    cum_distribution_gnd = [
        sum(map(lambda x: x.real >= real_value, real_values_gnd))
        for real_value in real_values_combined
    ]
    cum_distribution_diff = np.abs(
        np.array(cum_distribution_exc) - np.array(cum_distribution_gnd)
    )
    argmax = np.argmax(cum_distribution_diff)
    threshold = real_values_combined[argmax]
    errors_exc = niter - cum_distribution_exc[argmax]
    errors_gnd = cum_distribution_gnd[argmax]
    fidelity = cum_distribution_diff[argmax] / niter
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
