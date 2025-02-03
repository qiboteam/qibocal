from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log

from .utils import table_dict, table_html


@dataclass
class QubitVzParameters(Parameters):
    """"""

    use_flux_pulse: bool = True
    amplitude: float = 0.1
    duration: int = 40


QubitVzType = np.dtype([("phi", np.float64), ("prob", np.float64)])


@dataclass
class QubitVzData(Data):
    """"""

    data: dict[QubitId, npt.NDArray[QubitVzType]] = field(default_factory=dict)
    """Raw data acquired."""


@dataclass
class QubitVzResults(Results):
    """"""

    virtual_phase: dict[QubitId, float]
    fitted_parameters: dict[QubitId, list[float]]


def _acquisition(
    params: QubitVzParameters, platform: Platform, targets: list[QubitId]
) -> QubitVzData:
    """"""
    sequence = PulseSequence()
    phi_pulses = []
    for qubit in targets:
        qubit_sequence = PulseSequence()

        qubit_sequence.add(
            platform.create_RX90_pulse(qubit=qubit, start=0, relative_phase=0)
        )

        if params.use_flux_pulse:
            flux_pulse = FluxPulse(
                start=qubit_sequence.finish,
                duration=params.duration,
                amplitude=params.amplitude,
                shape=Rectangular(),
                qubit=qubit,
                channel=platform.qubits[qubit].flux.name,
            )
            qubit_sequence.add(flux_pulse)

        phi_pulse = platform.create_RX90_pulse(qubit, start=qubit_sequence.finish)
        qubit_sequence.add(phi_pulse)
        phi_pulses.append(phi_pulse)

        qubit_sequence.add(
            platform.create_MZ_pulse(qubit=qubit, start=qubit_sequence.finish)
        )

        sequence.add(qubit_sequence)

    log.info(f"Built the following sequence:\n {sequence}")

    phi_range = np.arange(0.0, 2 * np.pi, 0.01)

    sweeper = Sweeper(
        Parameter.relative_phase,
        phi_range,
        phi_pulses,
        type=SweeperType.ABSOLUTE,
    )

    data = QubitVzData()

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )
    for qubit in targets:
        prob = results[qubit].probability(state=1)
        data.register_qubit(
            QubitVzType,
            qubit,
            dict(
                phi=phi_range,
                prob=prob.tolist(),
            ),
        )
    return data


def fit_function(phi, theta, amp, offset):
    return amp * (np.cos(phi / 2 - theta / 2) ** 2) + offset


def _fit(data: QubitVzData) -> QubitVzResults:
    """"""
    fitted_parameters = {}
    virtual_phases = {}
    for qubit in data.qubits:

        phi = data[qubit].phi
        prob = data[qubit].prob

        pguess = [
            0,
            np.max(prob) - np.min(prob),
            np.min(prob),
        ]

        popt, _ = curve_fit(
            fit_function,
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
    """"""
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
                name="Probability",
                showlegend=True,
                legendgroup="Probability",
                mode="lines",
            ),
        ]
    )

    figures.append(fig)

    if fit is not None:
        phi = data[target].phi
        params = fit.fitted_parameters[target]
        fig.add_trace(
            go.Scatter(
                x=phi,
                y=fit_function(phi, *params),
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
    pass


qubit_vz = Routine(_acquisition, _fit, _plot, _update)
""""""
