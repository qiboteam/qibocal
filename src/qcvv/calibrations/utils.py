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


def check_frequency(platform, write=False):
    """
    Temporary function to change the IF of the QRM accordingly. It will be depriciated with multiple feedlines.
    """
    path = pathlib.Path(qibolab.__file__).parent / "runcards" / platform.runcard
    with open(path, "r") as f:
        settings = yaml.safe_load(f)

    for inst in settings["instruments"]:
        if "readout" in settings["instruments"][inst]["roles"]:
            freq = []
            freq_qrm = []
            for i in range(settings["nqubits"]):
                freq += [
                    settings["characterization"]["single_qubit"][i]["resonator_freq"]
                ]
                freq_qrm += [
                    settings["instruments"]["qrm_rf"]["settings"]["ports"]["o1"][
                        "lo_frequency"
                    ]
                    + settings["native_gates"]["single_qubit"][i]["MZ"]["frequency"]
                ]
                if abs(freq[i] - freq_qrm[i]) > 1:  # leaving a 1Hz resolution
                    log.info(
                        f"WARNING: Instrument parameters not matching with the characterization frequency of qubit {i}: {freq_qrm[i]} for {freq[i]}"
                    )
            if write:
                for i in range(settings["nqubits"]):
                    settings["instruments"]["qrm_rf"]["settings"]["ports"]["o1"][
                        "lo_frequency"
                    ] = (max(freq) + min(freq)) / 2
                    settings["native_gates"]["single_qubit"][i]["MZ"]["frequency"] = (
                        freq[i]
                        - settings["instruments"]["qrm_rf"]["settings"]["ports"]["o1"][
                            "lo_frequency"
                        ]
                    )

        if "flux" in settings["instruments"][inst]["roles"]:
            sweetspot = []
            sweetspot_spi = []
            for i in range(settings["nqubits"]):
                chan = settings["qubit_channel_map"][i][2]
                sweetspot += [
                    settings["characterization"]["single_qubit"][i]["sweetspot"]
                ]
                sweetspot_spi += [
                    settings["instruments"]["SPI"]["settings"]["s4g_modules"][chan][2]
                ]
                if (
                    abs(sweetspot[i] - sweetspot_spi[i]) > 1.0e-9
                ):  # leaving a 1uA resolution
                    log.info(
                        f"WARNING: Instrument parameters not matching with the characterization sweetspot of qubit {i}: {sweetspot[i]} for {sweetspot_spi[i]}"
                    )

            if write:
                chan = settings["qubit_channel_map"][i][2]
                settings["instruments"]["SPI"]["settings"]["s4g_modules"][chan][
                    2
                ] = sweetspot[i]

        if "control" in settings["instruments"][inst]["roles"]:
            freq = {}
            freq_qcm = {}
            for i in range(settings["nqubits"]):
                chan = settings["qubit_channel_map"][i][1]
                if (
                    chan
                    in settings["instruments"][inst]["settings"]["channel_port_map"]
                ):
                    port = settings["instruments"][inst]["settings"][
                        "channel_port_map"
                    ][chan]
                    freq[f"{i}"] = settings["characterization"]["single_qubit"][i][
                        "qubit_freq"
                    ]
                    freq_qcm[f"{i}"] = (
                        settings["instruments"][inst]["settings"]["ports"][port][
                            "lo_frequency"
                        ]
                        + settings["native_gates"]["single_qubit"][i]["RX"]["frequency"]
                    )
                    if (
                        abs(freq[f"{i}"] - freq_qcm[f"{i}"]) > 1
                    ):  # leaving a 1Hz resolution
                        log.info(
                            f"WARNING: Instrument parameters not matching with the characterization frequency of qubit {i}: {freq_qcm[f'{i}']} for {freq[f'{i}']}"
                        )
                    if write:
                        settings["instruments"][inst]["settings"]["ports"][port][
                            "lo_frequency"
                        ] = (
                            freq[f"{i}"]
                            - settings["native_gates"]["single_qubit"][i]["RX"][
                                "frequency"
                            ]
                        )
    if write:
        log.info(f"WARNING: Writting YAML")
        with open(path, "w") as f:
            yaml.dump(settings, f, sort_keys=False, indent=4, default_flow_style=None)


def get_fidelity(platform: AbstractPlatform, qubit, niter, param=None, save=True):
    """
    Returns the read-out fidelity for a

    """

    if save:
        if param is None:
            raise_error(ValueError, "Please provide the varied parameters")

    platform.reload_settings()
    platform.settings["hardware_avg"] = 1
    platform.qrm[qubit].ports[
        "i1"
    ].hardware_demod_en = True  # binning only works with hardware demodulation enabled
    # create exc sequence
    exc_sequence = PulseSequence()
    RX_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX_pulse.duration)
    exc_sequence.add(RX_pulse)
    exc_sequence.add(ro_pulse)

    data_exc = Dataset(
        name=f"data_exc_q{qubit}", quantities={"iteration": "dimensionless"}
    )
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
        data_exc.add(results)
    if save:
        yield data_exc

    gnd_sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    gnd_sequence.add(ro_pulse)

    data_gnd = Dataset(
        name=f"data_gnd_q{qubit}", quantities={"iteration": "dimensionless"}
    )

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
        data_gnd.add(results)
    yield data_gnd


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

    platform.reload_settings()
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
    # import matplotlib.pyplot as plt
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
    # plt.axvline(x=thres)
    # plt.plot(gnd_cumul[1,:], gnd_cumul[0,:])
    # plt.plot(exc_cumul[1,:], exc_cumul[0,:])
    # plt.savefig("fig.png")
    # plt.close()
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
    # Debug
    # import matplotlib.pyplot as plt

    # plt.plot(np.real(iq_gnd), np.imag(iq_gnd), "ok", alpha=0.3)
    # plt.plot(np.real(iq_exc), np.imag(iq_exc), "or", alpha=0.3)
    # plt.plot(np.real(np.mean(iq_gnd)), np.imag(np.mean(iq_gnd)), "ok", markersize=10)
    # plt.plot(np.real(np.mean(iq_exc)), np.imag(np.mean(iq_exc)), "or", markersize=10)

    # iq_mid = np.mean(iq_exc + iq_gnd) / 2
    # angle = np.pi / 2 - np.arctan(np.imag(iq_mid / np.real(iq_mid)))
    # iq_exc = iq_exc * np.exp(1j * angle)
    # iq_gnd = iq_gnd * np.exp(1j * angle)
    origin = np.mean(iq_gnd)
    iq_gnd = iq_gnd - origin
    iq_exc = iq_exc - origin
    angle = np.angle(np.mean(iq_exc))
    iq_exc = iq_exc * np.exp(-1j * angle) + origin
    iq_gnd = iq_gnd * np.exp(-1j * angle) + origin

    # plt.plot(np.real(iq_gnd), np.imag(iq_gnd), "ob", alpha=0.3)
    # plt.plot(np.real(iq_exc), np.imag(iq_exc), "og", alpha=0.3)
    # plt.plot(np.real(np.mean(iq_gnd)), np.imag(np.mean(iq_gnd)), "ob", markersize=10)
    # plt.plot(np.real(np.mean(iq_exc)), np.imag(np.mean(iq_exc)), "og", markersize=10)
    # plt.savefig("fig.png")
    # plt.close()
    return np.real(iq_exc), np.real(iq_gnd)
