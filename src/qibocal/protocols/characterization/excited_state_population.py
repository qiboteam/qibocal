"""Find remaining excited state population and qubit effective temperature."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.constants import hbar, k
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.protocols.characterization.utils import V_TO_UV, table_dict, table_html

from .rabi import utils
from .rabi.utils import rabi_length_fit as fitting


@dataclass
class ResidualPopulationParameters(Parameters):
    """ResidualPopulation runcard inputs."""

    pulse_duration_start: float
    """Initial RX12 pulse duration (ns)."""
    pulse_duration_end: float
    """Final RX12 pulse duration (ns)."""
    pulse_duration_step: float
    """Step RX12 pulse duration (ns)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class ResidualPopulationResults(Results):
    """ResidualPopulation outputs."""

    residual_excited_population: dict[QubitId, float]
    """Residual population in state |1>."""
    effective_qubit_temperature: dict[QubitId, float]
    """Effective qubit teperature."""
    fitted_parameters: dict[QubitId, tuple[dict[str, float]]]
    """Raw fitting output."""


ResPopType = np.dtype(
    [("length", np.float64), ("msr_g_start", np.float64), ("msr_e_start", np.float64)]
)
"""Custom dtype for residual population."""


@dataclass
class ResidualPopulationData(Data):
    """ResidualPopulation acquisition outputs."""

    data: dict[QubitId, npt.NDArray[ResPopType]] = field(default_factory=dict)
    """Raw data acquired."""

    frequencies: dict[QubitId, np.float64] = field(default_factory=dict)
    """Transition frequencies 0-1."""

    def register_qubit(self, qubit, length, msr_g_start, msr_e_start, frequency):
        """Store output for single qubit."""
        # to be able to handle the non-sweeper case
        shape = (1,) if np.isscalar(length) else length.shape
        ar = np.empty(shape, dtype=ResPopType)
        ar["length"] = length
        ar["msr_g_start"] = msr_g_start
        ar["msr_e_start"] = msr_e_start
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)

        self.frequencies[qubit] = frequency


def _acquisition(
    params: ResidualPopulationParameters, platform: Platform, qubits: Qubits
) -> ResidualPopulationData:
    """Perform the experiment.

    Data acquisition for ResidualPopulation Experiment.
    In the residual population experiment we probe a rabi-like oscillation between the
    first and second excited state. First, by applying nothing before the RX12, then
    by applying a pi-pulse first. The relative amplitude between the two oscillations
    is related to the residual excited population in a qubit and its effective temperature.
    """
    # create a sequence of pulses for the experiment
    sequence_g_start = PulseSequence()
    sequence_e_start = PulseSequence()
    rx12_g_pulses = {}
    rx12_e_pulses = {}
    ro_g_pulses = {}
    ro_e_pulses = {}
    for qubit in qubits:
        if qubits[qubit].native_gates.RX12 is None:
            raise ValueError(f"Qubit {qubit} does not have a RX12 calibrated.")

        # sequence starting from g
        rx12_g_pulses[qubit] = platform.create_RX12_pulse(qubit, start=0)
        ro_g_pulses[qubit] = platform.create_MZ_pulse(
            qubit, start=rx12_g_pulses[qubit].finish
        )

        sequence_g_start.add(rx12_g_pulses[qubit])  # RX12
        sequence_g_start.add(ro_g_pulses[qubit])  # MZ

        # sequence starting from e
        rx_pulse = platform.create_RX_pulse(qubit, start=0)
        rx12_e_pulses[qubit] = platform.create_RX_pulse(qubit, start=rx_pulse.finish)
        ro_e_pulses[qubit] = platform.create_MZ_pulse(
            qubit, start=rx12_e_pulses[qubit].finish
        )

        sequence_e_start.add(rx_pulse)  # RX
        sequence_e_start.add(rx12_g_pulses[qubit])  # RX12
        sequence_e_start.add(ro_e_pulses[qubit])  # MZ

    # define the parameter to sweep and its range:
    # qubit drive pulse duration time
    rx12_duration_range = np.arange(
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )

    sweeper_g_start = Sweeper(
        Parameter.duration,
        rx12_duration_range,
        [rx12_g_pulses[qubit] for qubit in qubits],
        type=SweeperType.ABSOLUTE,
    )
    sweeper_e_start = Sweeper(
        Parameter.duration,
        rx12_duration_range,
        [rx12_e_pulses[qubit] for qubit in qubits],
        type=SweeperType.ABSOLUTE,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse length
    data = ResidualPopulationData()

    # execute the sweep
    results_g_start = platform.sweep(
        sequence_g_start,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper_g_start,
    )
    results_e_start = platform.sweep(
        sequence_e_start,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper_e_start,
    )

    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        result_g = results_g_start[ro_g_pulses[qubit].serial]
        result_e = results_e_start[ro_e_pulses[qubit].serial]

        data.register_qubit(
            qubit,
            length=rx12_duration_range,
            msr_g_start=result_g.magnitude,
            msr_e_start=result_e.magnitude,
            frequency=qubits[qubit].native_gates.RX.frequency,
        )
    return data


def _fit_and_get_amplitude(
    rabi_parameter: npt.NDArray[np.float64], voltages: npt.NDArray[np.float64]
):
    """Fit."""
    y_min = np.min(voltages)
    y_max = np.max(voltages)
    x_min = np.min(rabi_parameter)
    x_max = np.max(rabi_parameter)
    x = (rabi_parameter - x_min) / (x_max - x_min)
    y = (voltages - y_min) / (y_max - y_min)

    # Guessing period using fourier transform
    ft = np.fft.rfft(y)
    mags = abs(ft)
    local_maxima = find_peaks(mags, threshold=1)[0]
    index = local_maxima[0] if len(local_maxima) > 0 else None
    # 0.5 hardcoded guess for less than one oscillation
    f = x[index] / (x[1] - x[0]) if index is not None else 0.5

    pguess = [1, 1, f, np.pi / 2, 0]
    try:
        popt, pcov = curve_fit(
            utils.rabi_length_fit,
            x,
            y,
            p0=pguess,
            maxfev=100000,
            bounds=(
                [0, 0, 0, -np.pi, 0],
                [1, 1, np.inf, np.pi, np.inf],
            ),
        )
        translated_popt = [
            (y_max - y_min) * popt[0] + y_min,
            (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
            popt[2] / (x_max - x_min),
            popt[3] - 2 * np.pi * x_min * popt[2] / (x_max - x_min),
            popt[4] / (x_max - x_min),
        ]
        pi_pulse_parameter = np.abs((1.0 / translated_popt[2]) / 2)
    except:
        log.warning("rabi_fit: the fitting was not succesful")
        pi_pulse_parameter = 0
        translated_popt = [0] * 5

    return pi_pulse_parameter, translated_popt


def _fit(data: ResidualPopulationData) -> ResidualPopulationResults:
    """Post-processing for ResidualPopulation experiment."""
    qubits = data.qubits
    fitted_parameters = {}
    res_population = {}
    temperature = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        rabi_parameter = qubit_data.length
        voltages_g_start = qubit_data.msr_g_start
        voltages_e_start = qubit_data.msr_e_start

        g_start_amp, raw_fit_g = _fit_and_get_amplitude(
            rabi_parameter, voltages_g_start
        )
        e_start_amp, raw_fit_e = _fit_and_get_amplitude(
            rabi_parameter, voltages_e_start
        )

        fitted_parameters[qubit] = (raw_fit_g, raw_fit_e)
        r = e_start_amp / g_start_amp
        res_population[qubit] = r / (r + 1)
        temperature[qubit] = -(hbar * data.frequencies[qubit]) / (np.log(r) * k)

    return ResidualPopulationResults(res_population, temperature, fitted_parameters)


def _update(results: ResidualPopulationResults, platform: Platform, qubit: QubitId):
    """Do not update anything."""


def _plot(data: ResidualPopulationData, fit: ResidualPopulationResults, qubit):
    """Plot function for ResidualPopulation experiment."""

    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )

    qubit_data = data[qubit]

    fig.add_trace(
        go.Scatter(
            x=qubit_data.length,
            y=qubit_data.msr_g_start * V_TO_UV,
            name="Rabi starting from g",
            opacity=1,
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=qubit_data.length,
            y=qubit_data.msr_e_start * V_TO_UV,
            opacity=1,
            name="Rabi starting from e",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    if fit is not None:
        rabi_parameter_range = np.linspace(
            min(qubit_data.length),
            max(qubit_data.length),
            2 * len(qubit_data.length),
        )
        params = fit.fitted_parameters[qubit]
        fig.add_trace(
            go.Scatter(
                x=rabi_parameter_range,
                y=fitting(rabi_parameter_range, *(params[0])) * V_TO_UV,
                name="Fit starting from g",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(102, 0, 204)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=rabi_parameter_range,
                y=fitting(rabi_parameter_range, *(params[1])) * V_TO_UV,
                name="Fit starting from e",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 67, 130)",
            ),
            row=1,
            col=1,
        )

        fitting_report = table_html(
            table_dict(
                qubit,
                ["Residual excited population", "Effective qubit temperature [K]"],
                [
                    fit.residual_excited_population[qubit],
                    fit.effective_qubit_temperature[qubit],
                ],
            )
        )

        fig.update_layout(
            showlegend=True,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title="Length pulse [ns]",
            yaxis_title="MSR (uV)",
        )

    figures.append(fig)

    return figures, fitting_report


residual_excited_population = Routine(_acquisition, _fit, _plot, _update)
"""ResidualPopulation Routine object."""
