from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Data, Routine, Parameters, Results
from qibocal.config import log

from qibocal.protocols.rabi.amplitude_signal import RabiAmplitudeVoltParameters, RabiAmplitudeVoltResults


@dataclass
class RabiAmplitudeParameters(Parameters):
    """RabiAmplitude for frustration test runcard inputs."""

    min_amp_factor: float
    """Minimum amplitude multiplicative factor."""
    max_amp_factor: float
    """Maximum amplitude multiplicative factor."""
    step_amp_factor: float
    """Step amplitude multiplicative factor."""
    measure_qubits: Optional[QubitId]
    """List of qubits to measure."""
    pulse_length: Optional[float]
    """RX pulse duration [ns]."""


@dataclass
class RabiAmplitudeResults(Results):
    """RabiAmplitude for frustration test results outputs."""

PROJECTIONS = ['X', 'Y', 'Z']
"""Standard projections for measurements."""

RabiAmpType = np.dtype(
    [("amp", np.float64), ("prob", np.float64), ("error", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeData(Data):
    """RabiAmplitude data acquisition."""

    durations: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""

    qubit_ros: list[QubitId] = None
    """Measured qubits."""

    data: dict[QubitId, QubitId, str, npt.NDArray[RabiAmpType]] = field(default_factory=dict)
    """Raw data acquired."""


def ro_projection_pulse(platform: Platform, qubit, start=0, projection = 'X'):
    sequence = PulseSequence()
    """Create a readout pulse for a given qubit."""
    if projection == PROJECTIONS[2]:
        return platform.create_qubit_readout_pulse(qubit, start=start)
    elif projection == PROJECTIONS[1]:
        drive_pulse = platform.create_RX90_pulse(qubit, start=start)
        sequence.add(drive_pulse)
        sequence.add(
            platform.create_qubit_readout_pulse(
                qubit, 
                start=drive_pulse.finish)
        )
    elif projection == PROJECTIONS[0]:
        drive_pulse = platform.create_RX90_pulse(qubit, start=start, relative_phase=90)
        sequence.add(drive_pulse)
        sequence.add(
            platform.create_qubit_readout_pulse(
                qubit, 
                start=drive_pulse.finish)
        )
    else:
        raise ValueError(f"Invalid projection {projection}")

    return sequence

def _acquisition(
    params: RabiAmplitudeParameters, platform: Platform, targets: list[QubitId]
) -> RabiAmplitudeData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    In the case of quantum frustrated qubits, exciting one of the qubits should change the state of another one sharing the same frequency. 
    """
    
    # create a sequence of pulses for the experiment
    
    durations = {}

    data = RabiAmplitudeData(durations=durations, qubit_ros=params.measure_qubits)
    for qubit in targets:
        for projection in PROJECTIONS:
            sequence = PulseSequence()
            qd_pulse = platform.create_RX_pulse(qubit, start=0)

            if params.pulse_length is None:
                durations[qubit] = qd_pulse.duration
            else:
                qd_pulse.duration = params.pulse_length 
                durations[qubit] = params.pulse_length

            sequence.add(qd_pulse)

            if params.measure_qubits is None:
                ro_qubits = [qubit]
            else:
                ro_qubits = params.measure_qubits

            for ro_qubit in ro_qubits:
                sequence.add(ro_projection_pulse(
                        platform,
                        qubit=ro_qubit, 
                        start=qd_pulse.finish,
                        projection=projection)
                        )

            # define the parameter to sweep and its range:
            # qubit drive pulse amplitude
            qd_pulse_amplitude_range = np.arange(
                params.min_amp_factor,
                params.max_amp_factor,
                params.step_amp_factor,
            )
            sweeper_amplitude = Sweeper(
                parameter= Parameter.amplitude,
                values = qd_pulse_amplitude_range,
                pulses = [qd_pulse],
                type = SweeperType.FACTOR,
            )
    
            results = platform.sweep(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.SINGLESHOT,
                ),
                sweeper_amplitude,
            )

            for ro_qubit in  ro_qubits:
                prob = results[ro_qubit].probability(state=1)
                data.register_qubit(
                    dtype = RabiAmpType,
                    data_keys = (qubit, ro_qubit, projection), 
                    data_dict=dict(
                        amp=qd_pulse.amplitude * qd_pulse_amplitude_range,
                        prob=prob.tolist(),
                        error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
                    ),
                )
    
    return data


def _fit(data: RabiAmplitudeData) -> RabiAmplitudeResults:
    """Post-processing for RabiAmplitude."""
    return RabiAmplitudeResults()


def _plot(data: RabiAmplitudeData, target: QubitId, fit: RabiAmplitudeResults = None):
    """Plotting function for RabiAmplitude."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from ..utils import COLORBAND, COLORBAND_LINE, table_dict, table_html

    figures = []
    fitting_report = ""
    ro_qubits = data.qubit_ros

    fig = go.Figure()
    for ro_qubit in ro_qubits:
        for projection in PROJECTIONS:
            qubit_data = data.data[target, ro_qubit, projection]

            probs = qubit_data.prob
            error_bars = qubit_data.error

            rabi_parameters = getattr(qubit_data, "amp")
            fig.add_trace(
                    go.Scatter(
                        x=rabi_parameters,
                        y=qubit_data.prob,
                        opacity=1,
                        name=f"Q{ro_qubit} <{projection}> ",
                        showlegend=True,
                        legendgroup=f"Prob Qubit {ro_qubit}",
                        mode="lines",
                    ))
            fig.add_trace(
                    go.Scatter(
                        x=np.concatenate((rabi_parameters, rabi_parameters[::-1])),
                        y=np.concatenate((probs + error_bars, (probs - error_bars)[::-1])),
                        fill="toself",
                        fillcolor=COLORBAND,
                        line=dict(color=COLORBAND_LINE),
                        showlegend=False,
                        legendgroup=f"Prob Qubit {ro_qubit}",
                        name=f"Errors Q{ro_qubit}",
                    ),
            )

    figures.append(fig)


    return figures, ""


def _update(results: RabiAmplitudeResults, platform: Platform, target: QubitId):
    update.drive_amplitude(results.amplitude[target], platform, target)


frustration_rabi = Routine(_acquisition, _fit, _plot)
"""RabiAmplitude Routine object."""
