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


def check_frequency(platform, write=False):
    """
    Temporary function to change the IF of the QRM accordingly. It will be depriciated with multiple feedlines.
    """
    path = pathlib.Path(qibolab.__file__).parent / "runcards" / platform.runcard
    with open(path, "r") as f:
        settings = yaml.safe_load(f)

    for inst in settings["instruments"]:
        if "readout" in settings["instruments"][inst]["roles"]:
            freq = {}
            freq_qrm = {}
            for i in platform.qubits:
                freq[i] = settings["characterization"]["single_qubit"][i][
                    "resonator_freq"
                ]

                freq_qrm[i] = (
                    settings["instruments"]["qrm_rf"]["settings"]["ports"]["o1"][
                        "lo_frequency"
                    ]
                    + settings["native_gates"]["single_qubit"][i]["MZ"]["frequency"]
                )

                if abs(freq[i] - freq_qrm[i]) > 1:  # leaving a 1Hz resolution
                    log.info(
                        f"WARNING: Instrument parameters not matching with the characterization frequency of qubit {i}: {freq_qrm[i]} for {freq[i]}"
                    )
            if write:
                lo = _frequency_allocation(list(freq.values()))
                for i in platform.qubits:
                    settings["instruments"]["qrm_rf"]["settings"]["ports"]["o1"][
                        "lo_frequency"
                    ] = float(lo)
                    settings["native_gates"]["single_qubit"][i]["MZ"][
                        "frequency"
                    ] = int(freq[i] - float(lo))

        if "flux" in settings["instruments"][inst]["roles"]:
            sweetspot = {}
            sweetspot_spi = {}
            for i in platform.qubits:
                chan = settings["qubit_channel_map"][i][2]
                sweetspot[i] = settings["characterization"]["single_qubit"][i][
                    "sweetspot"
                ]
                sweetspot_spi[i] = settings["instruments"]["SPI"]["settings"][
                    "s4g_modules"
                ][chan][2]

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
            for i in platform.qubits:
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


def _frequency_allocation(freq, bandwidth=600e6, weights=[1, 1, 1, 1]):
    """
    Function suppose to set the center frequency (LO) to avoid that spurs overlap the desired frequencies. Few problems:
    - scipy.minimize not working well, so it is simply iterating through all values
    - weights are too hard to choose, so a different weighting method should be used
    - for most weights, the solution is to set the LO to be furthest to most peaks, this would result in spurs greatly spaced
    which might work well

    Param:
    bandwidth: bandwidth of the instrument
    weights: factor to which to value important of a peak [image_spur, image, lo_leake, rf_spur]

    Return:
    LO_optimal: optimal lo frequency
    if: IF frequencies to setup
    """
    scale = 1 / max(freq)
    bandwidth = np.array(bandwidth) * scale
    freq = np.array(freq) * scale

    rn = list(range(len(freq)))

    dx = bandwidth - (max(freq) - min(freq))
    if dx <= 0:
        raise ValueError("Bandwitch too small for resonator's spacing")
    freq_max = max(freq) - (bandwidth / 2 - dx)
    freq_min = min(freq) + (bandwidth / 2 - dx)

    def cost(LO):
        LO = LO[0]
        freqs = np.zeros((len(rn), 4))

        for i in rn:
            freqs[i, :] = np.array(
                [-2 * (freq[i] - LO), -(freq[i] - LO), 0, 2 * (freq[i] - LO)]
            ) + np.array(LO)
        c = 0
        for i in rn:
            for j in rn:
                if i != j:
                    c += np.sum((1) / (1 + weights * np.abs(freq[i] - freqs[j, :])))
        return c

    # result = minimize(lambda x: cost(x), x0 = (min(freq)+max(freq))/2, method="Nelder-Mead", bounds=[(freq_min, freq_max)])
    x = np.arange(freq_min, freq_max, 100e3 * scale)
    y = []
    for f in x:
        y += [cost([f])]

    LO_optimal = x[np.argmin(y)] / scale

    return LO_optimal
