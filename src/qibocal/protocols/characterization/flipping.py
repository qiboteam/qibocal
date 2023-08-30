from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log

from .utils import V_TO_UV


@dataclass
class FlippingParameters(Parameters):
    """Flipping runcard inputs."""

    nflips_max: int
    """Maximum number of flips ([RX(pi) - RX(pi)] sequences). """
    nflips_step: int
    """Flip step."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class FlippingResults(Results):
    """Flipping outputs."""

    amplitude: dict[QubitId, float] = field(metadata=dict(update="drive amplitude"))
    """Drive amplitude for each qubit."""
    amplitude_factors: dict[QubitId, float]
    """Drive amplitude correction factor for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


FlippingType = np.dtype([("flips", np.float64), ("msr", np.float64)])


@dataclass
class FlippingData(Data):
    """Flipping acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    pi_pulse_amplitudes: dict[QubitId, float]
    """Pi pulse amplitudes for each qubit."""
    data: dict[QubitId, npt.NDArray[FlippingType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, flips, msr):
        """Store output for single qubit."""
        ar = np.empty((1,), dtype=FlippingType)
        ar["flips"] = flips
        ar["msr"] = msr
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: FlippingParameters,
    platform: Platform,
    qubits: Qubits,
) -> FlippingData:
    r"""
    Data acquisition for flipping.

    The flipping experiment correct the delta amplitude in the qubit drive pulse. We measure a qubit after applying
    a Rx(pi/2) and N flips (Rx(pi) rotations). After fitting we can obtain the delta amplitude to refine pi pulses.

    Args:
        params (:class:`SingleShotClassificationParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`FlippingData`)
    """

    # create a DataUnits object to store MSR, phase, i, q and the number of flips
    data = FlippingData(
        resonator_type=platform.resonator_type,
        pi_pulse_amplitudes={
            qubit: qubits[qubit].pi_pulse_amplitude for qubit in qubits
        },
    )
    # sweep the parameter
    for flips in range(0, params.nflips_max, params.nflips_step):
        # create a sequence of pulses for the experiment
        sequence = PulseSequence()
        ro_pulses = {}
        for qubit in qubits:
            RX90_pulse = platform.create_RX90_pulse(qubit, start=0)
            sequence.add(RX90_pulse)
            # execute sequence RX(pi/2) - [RX(pi) - RX(pi)] from 0...flips times - RO
            start1 = RX90_pulse.duration
            for j in range(flips):
                RX_pulse1 = platform.create_RX_pulse(qubit, start=start1)
                start2 = start1 + RX_pulse1.duration
                RX_pulse2 = platform.create_RX_pulse(qubit, start=start2)
                sequence.add(RX_pulse1)
                sequence.add(RX_pulse2)
                start1 = start2 + RX_pulse2.duration

            # add ro pulse at the end of the sequence
            ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=start1)
            sequence.add(ro_pulses[qubit])
        # execute the pulse sequence
        results = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
        )
        for qubit in qubits:
            result = results[ro_pulses[qubit].serial]
            data.register_qubit(qubit=qubit, flips=flips, msr=result.magnitude)

    return data


def flipping_fit(x, p0, p1, p2, p3):
    # A fit to Flipping Qubit oscillation
    # Epsilon?? should be Amplitude : p[0]
    # Offset                        : p[1]
    # Period of oscillation         : p[2]
    # phase for the first point corresponding to pi/2 rotation   : p[3]
    return np.sin(x * 2 * np.pi / p2 + p3) * p0 + p1


# FIXME: not working
def _fit(data: FlippingData) -> FlippingResults:
    r"""Post-processing function for Flipping.

    The used model is

    .. math::

        y = p_0 sin\Big(\frac{2 \pi x}{p_2} + p_3\Big) + p_1.
    """
    qubits = data.qubits
    corrected_amplitudes = {}
    fitted_parameters = {}
    amplitude_correction_factors = {}
    for qubit in qubits:
        qubit_data = data[qubit]
        pi_pulse_amplitude = data.pi_pulse_amplitudes[qubit]
        voltages = qubit_data.msr * V_TO_UV
        flips = qubit_data.flips
        if data.resonator_type == "3D":
            pguess = [
                pi_pulse_amplitude / 2,
                np.mean(voltages),
                -40,
                0,
            ]  # epsilon guess parameter
        else:
            pguess = [
                pi_pulse_amplitude / 2,
                np.mean(voltages),
                40,
                0,
            ]  # epsilon guess parameter

        try:
            popt, _ = curve_fit(flipping, flips, voltages, p0=pguess, maxfev=2000000)

        except:
            log.warning("flipping_fit: the fitting was not succesful")
            popt = [0, 0, 2, 0]

        # sen fitting succesful
        if popt[2] != 2:
            eps = -1 / popt[2]
            amplitude_correction_factor = eps / (eps - 1)
            corrected_amplitude = amplitude_correction_factor * pi_pulse_amplitude
        # sen fitting not succesful = amplitude well adjusted
        else:
            amplitude_correction_factor = 1
            corrected_amplitude = amplitude_correction_factor * pi_pulse_amplitude

        corrected_amplitudes[qubit] = corrected_amplitude
        fitted_parameters[qubit] = popt
        amplitude_correction_factors[qubit] = amplitude_correction_factor

    return FlippingResults(
        corrected_amplitudes, amplitude_correction_factors, fitted_parameters
    )


def _plot(data: FlippingData, qubit, fit: FlippingResults = None):
    """Plotting function for Flipping."""

    figures = []
    fig = go.Figure()

    fitting_report = None
    qubit_data = data[qubit]

    fig.add_trace(
        go.Scatter(
            x=qubit_data.flips,
            y=qubit_data.msr * V_TO_UV,
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        ),
    )

    if fit is not None:
        fitting_report = ""
        flips_range = np.linspace(
            min(qubit_data.flips),
            max(qubit_data.flips),
            2 * len(qubit_data),
        )

        fig.add_trace(
            go.Scatter(
                x=flips_range,
                y=flipping_fit(
                    flips_range,
                    float(fit.fitted_parameters[qubit][0]),
                    float(fit.fitted_parameters[qubit][1]),
                    float(fit.fitted_parameters[qubit][2]),
                    float(fit.fitted_parameters[qubit][3]),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
        )
        fitting_report = fitting_report + (
            f"{qubit} | Amplitude correction factor: {fit.amplitude_factors[qubit]:.4f}<br>"
            + f"{qubit} | Corrected amplitude: {fit.amplitude[qubit]:.4f}<br><br>"
        )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Flips (dimensionless)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


flipping = Routine(_acquisition, _fit, _plot)
"""Flipping Routine  object."""
