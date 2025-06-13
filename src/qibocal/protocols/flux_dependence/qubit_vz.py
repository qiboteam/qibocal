"""Experiment to measure the Z rotation of a qubit under a rectangular flux pulse."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Platform,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)
from scipy.optimize import curve_fit

from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitId,
    Results,
    Routine,
)
from qibocal.config import log

from ..utils import table_dict, table_html

__all__ = ["qubit_vz"]


@dataclass
class QubitVzParameters(Parameters):
    """Input parameters of the experiment."""

    amplitude: float = 0.1
    """The amplitude of the flux pulse."""
    duration: int = 40
    """The duration of flux pulse (ns)."""
    use_flux_pulse: bool = True
    """If false, will not apply the flux pulse."""


QubitVzType = np.dtype([("phi", np.float64), ("prob", np.float64)])
"""
Datatype for the execution results.
phi: the values of the elative phase sweep of the second pi/2 pulse, prob: probability of excited state.
"""


@dataclass
class QubitVzData(Data):
    """Acquisition results."""

    data: dict[QubitId, npt.NDArray[QubitVzType]] = field(default_factory=dict)
    """Raw data acquired."""


@dataclass
class QubitVzResults(Results):
    """Fitting results."""

    virtual_phase: dict[QubitId, float]
    """The calculated Z rotation angle for each target."""
    fitted_parameters: dict[QubitId, list[float]]
    """Parameters of the fit."""


def _acquisition(
    params: QubitVzParameters, platform: Platform, targets: list[QubitId]
) -> QubitVzData:
    """
    The pulse sequence for this experiment is as follows:
        1. X90 rotation
        2. Flux pulse
        3. XY90 rotation (i.e. same as X90, but with a relative phase)
        4. Measure

    The relative phase of the second rotation (step 3. above) is swetp in the range [0, 2*pi).
    This phase is refered to as phi throughout the experiment.
    """
    sequence = PulseSequence()
    phi_pulses = []
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        qd_channel = platform.qubits[qubit].drive
        qf_channel = platform.qubits[qubit].flux

        qubit_sequence = natives.R(theta=np.pi / 2)

        if params.use_flux_pulse:
            flux_pulse = Pulse(
                duration=params.duration,
                amplitude=params.amplitude,
                envelope=Rectangular(),
            )
            qubit_sequence.append((qf_channel, Delay(duration=qubit_sequence.duration)))
            qubit_sequence.append((qf_channel, flux_pulse))
            qubit_sequence.append((qd_channel, Delay(duration=flux_pulse.duration)))

        rx90_sequence = natives.R(theta=np.pi / 2)

        qubit_sequence += rx90_sequence
        for _, pulse in rx90_sequence:
            phi_pulses.append(pulse)

        mz_channel, mz_pulse = natives.MZ()[0]
        qubit_sequence.append((mz_channel, Delay(duration=qubit_sequence.duration)))
        qubit_sequence.append((mz_channel, mz_pulse))

        sequence.extend(qubit_sequence)

    log.info(f"Built the following sequence:\n {sequence}")

    phi_range = np.arange(0.0, 2 * np.pi, 0.11)

    sweeper = Sweeper(
        parameter=Parameter.relative_phase,
        values=phi_range,
        pulses=phi_pulses,
    )

    data = QubitVzData()

    results = platform.execute(
        [sequence],
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for qubit in targets:
        acq_handle = list(sequence.channel(platform.qubits[qubit].acquisition))[-1].id
        data.register_qubit(
            QubitVzType,
            qubit,
            dict(
                phi=phi_range,
                prob=results[acq_handle],
            ),
        )
    return data


def _fit_function(phi, theta, amp, offset):
    """
    Definition of the model to fit the data.

    This is the probabilty of excited state as derived from theory, plus some additional parameters (amp and offset)
    to account for unidealities of the reality.
    """
    return amp * (np.cos(phi / 2 - theta / 2) ** 2) + offset


def _fit(data: QubitVzData) -> QubitVzResults:
    """Fit the data."""
    fitted_parameters = {}
    virtual_phases = {}
    for qubit in data.qubits:
        phi = data[qubit].phi
        prob = data[qubit].prob

        theta_guess = np.argmax(prob) * 2 * np.pi / len(phi)
        theta_guess = theta_guess if theta_guess <= np.pi else theta_guess - 2 * np.pi
        pguess = [
            theta_guess,
            np.max(prob) - np.min(prob),
            np.min(prob),
        ]

        popt, _ = curve_fit(
            _fit_function,
            phi,
            prob,
            p0=pguess,
        )
        popt = popt.tolist()
        fitted_parameters[qubit] = popt
        virtual_phases[qubit] = popt[0]

    return QubitVzResults(
        virtual_phase=virtual_phases, fitted_parameters=fitted_parameters
    )


def _plot(data: QubitVzData, target: QubitId, fit: QubitVzResults = None):
    """Plot the raw data, the fit, and tabulate the calculated result."""
    figures = []
    fitting_report = ""

    phi = data[target].phi
    prob = data[target].prob

    fig = go.Figure(
        [
            go.Scatter(
                x=phi,
                y=prob,
                opacity=1,
                name="Raw data",
                showlegend=True,
                mode="lines",
            ),
        ]
    )
    fig.update_layout(
        showlegend=True,
        xaxis_title="Phi [rad]",
        yaxis_title="Excited state probability",
    )

    figures.append(fig)

    if fit is not None:
        phi = data[target].phi
        params = fit.fitted_parameters[target]
        fig.add_trace(
            go.Scatter(
                x=phi,
                y=_fit_function(phi, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
        )

        fitting_report = table_html(
            table_dict(
                target,
                ["Virtual phase"],
                [fit.virtual_phase[target]],
            )
        )

    return figures, fitting_report


def _update(results: QubitVzResults, platform: Platform, target: QubitId):
    """This experiment does not update any parameters in the platform."""
    pass


qubit_vz = Routine(_acquisition, _fit, _plot, _update)
