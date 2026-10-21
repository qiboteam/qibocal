"""Coupler flux dependence from the avoided crossings of a single qubit.

Variant of :mod:`.anti_crossing`: instead of probing both qubits of a pair once, only
the first qubit is probed, at several of its own flux biases. Each qubit bias moves the
qubit frequency, and with it the two crossings with the coupler, to another point of
the coupler curve. The crossings and tangents of all qubit biases are fitted together
with the coupler parabola, which is therefore constrained by many more points.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from qibocal.auto.operation import Data, Parameters, Protocol, QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import RangeLike, reshaping_raw_signal, to_range

from .anti_crossing import (
    CouplerAnticrossingResults,
    _update,
    crossings,
    parabola_fit,
    plot_crossings,
    spectroscopy,
)

__all__ = ["coupler_anticrossing_flux"]

CouplerAnticrossingFluxType = np.dtype(
    [("qubit_bias", float), ("bias", float), ("frequency", float), ("signal", float)]
)
"""One point of a spectroscopy map, labelled by the qubit flux bias."""


@dataclass
class CouplerAnticrossingFluxParameters(Parameters):
    """Coupler anticrossing versus qubit flux runcard inputs."""

    bias: RangeLike
    """Coupler flux pulse amplitude range [a.u.], on top of the coupler DC offset."""
    qubit_bias: RangeLike
    """Flux offsets [a.u.] of the probed qubit, relative to its current offset."""
    frequency: RangeLike
    """Drive frequency range [Hz] around the qubit drive frequency; it must contain
    the qubit at every qubit bias."""
    pulse_duration: float
    """Duration of the drive and coupler flux pulses [ns]."""
    pulse_amplitude: float
    """Amplitude of the drive pulse [a.u.]."""


@dataclass
class CouplerAnticrossingFluxData(Data):
    """Spectroscopy maps of the first qubit of each pair, for all qubit biases."""

    data: dict[QubitPairId, npt.NDArray] = field(default_factory=dict)


def _acquisition(
    params: CouplerAnticrossingFluxParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerAnticrossingFluxData:
    """Probe the first qubit of each pair once per qubit flux bias."""
    data = CouplerAnticrossingFluxData()
    for pair in map(tuple, targets):
        channel = platform.qubits[pair[0]].flux
        offset = platform.config(channel).offset
        for value in np.arange(*to_range(params.qubit_bias)):
            # The qubit bias is a DC offset update, applied only to this acquisition.
            updates = {channel: {"offset": offset + value}}
            record = spectroscopy(
                platform, pair, pair[0], params.frequency, params, updates
            )
            record["qubit_bias"] = np.full(len(record["bias"]), value)
            data.register_qubit(CouplerAnticrossingFluxType, pair, record)
    return data


def _fit(data: CouplerAnticrossingFluxData) -> CouplerAnticrossingResults:
    """Fit the crossings of every qubit bias, then the coupler on all of them."""
    results = CouplerAnticrossingResults()
    for pair, record in data.data.items():
        found = []
        for value in np.unique(record["qubit_bias"]):
            spectrum = record[record["qubit_bias"] == value]
            frequency, bias, signal = reshaping_raw_signal(
                spectrum["frequency"], spectrum["bias"], spectrum["signal"]
            )
            labels = {"qubit": pair[0], "qubit_bias": float(value)}
            found += [c | labels for c in crossings(bias, frequency, signal)]
        results.crossings[tuple(pair)] = found
        if (coupler := parabola_fit(found)) is not None:
            results.coupler[tuple(pair)] = coupler
    return results


def _plot(
    data: CouplerAnticrossingFluxData,
    target: QubitPairId,
    fit: CouplerAnticrossingResults | None = None,
):
    """Crossings of all qubit biases, their tangents and the coupler parabola."""
    pair = tuple(target)
    if fit is None:
        return [], ""
    return plot_crossings(
        fit.crossings[pair], fit.coupler.get(pair), data[pair]["bias"]
    )


coupler_anticrossing_flux = Protocol(_acquisition, _fit, _plot, _update)
"""CouplerAnticrossingFlux Routine object."""
