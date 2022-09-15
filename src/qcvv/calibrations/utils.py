# -*- coding: utf-8 -*-
import pathlib

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
