"""Cryoscope experiment for two qubit gates, cryscope plot."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.qubits import QubitId
from qibolab.result import AveragedSampleResults
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

MX_tag = "MX"
MY_tag = "MY"
MZ_tag = "MZ"
M0_tag = "M0"
M1_tag = "M1"
TAGS = [MX_tag, MY_tag, MZ_tag, M0_tag, M1_tag]


@dataclass
class CryscopeParameters(Parameters):
    """Cryoscope runcard inputs."""

    flux_pulse_duration_start: int
    """Duration minimum."""
    flux_pulse_duration_end: int
    """Duration maximum."""
    flux_pulse_duration_step: int
    """Duration step."""

    flux_pulse_amplitude: float
    """Amplitude."""

    delay_before_flux: int
    delay_after_flux: int

    flux_pulse_shapes: Optional[dict] = None
    flux_pulse_shape: Optional[str] = None
    nshots: Optional[int] = None
    """Number of shots per point."""
    relaxation_time: Optional[int] = None


@dataclass
class CryscopeResults(Results):
    """Cryoscope outputs."""

    filter_coefs: list[float]
    """FIR IIR coefficients to predistor a flux pulse"""


CryscopeType = np.dtype(
    [
        ("flux_pulse_duration", np.float64),
        ("flux_pulse_amplitude", np.float64),
        ("probability", np.float64),
    ]
)
"""Custom dtype for cryscope."""


@dataclass
class CryscopeData(Data):
    """Cryscope acquisition outputs."""

    data: dict[tuple[QubitId, str], npt.NDArray[CryscopeType]] = field(
        default_factory=dict
    )

    def register_qubit(
        self,
        qubit,
        component,
        flux_pulse_duration,
        flux_pulse_amplitude,
        probability,
    ):
        """Store output for single qubit."""
        size = len(flux_pulse_duration) * len(flux_pulse_amplitude)
        _flux_pulse_duration, _flux_pulse_amplitude = np.meshgrid(
            flux_pulse_duration, flux_pulse_amplitude
        )
        ar = np.empty(size, dtype=CryscopeType)
        ar["flux_pulse_duration"] = _flux_pulse_duration.ravel()
        ar["flux_pulse_amplitude"] = _flux_pulse_amplitude.ravel()
        ar["probability"] = probability.ravel()
        self.data[qubit, component] = np.rec.array(ar)


def _aquisition(
    params: CryscopeParameters,
    platform: Platform,
    targets: list[QubitId],
) -> CryscopeData:
    flux_pulse_duration_start: int = params.flux_pulse_duration_start
    flux_pulse_duration_end: int = params.flux_pulse_duration_end
    flux_pulse_duration_step: int = params.flux_pulse_duration_step
    flux_pulse_amplitude: float = params.flux_pulse_amplitude
    delay_before_flux: int = params.delay_before_flux
    delay_after_flux: int = params.delay_after_flux
    flux_pulse_shapes: dict = params.flux_pulse_shapes
    flux_pulse_shape: str = params.flux_pulse_shape
    nshots: int = params.nshots
    relaxation_time = params.relaxation_time

    # define the sequences of pulses to be executed
    MX_seq = PulseSequence()
    MY_seq = PulseSequence()
    MZ_seq = PulseSequence()
    M0_seq = PulseSequence()
    M1_seq = PulseSequence()

    # create a CryscopeData object to store the results,
    data = CryscopeData()

    initial_RY90_pulses = {}
    flux_pulses = {}
    RX90_pulses = {}
    RY90_pulses = {}
    MZ_ro_pulses = {}
    for qubit in targets:
        # start at |+> by rotating Ry(pi/2)
        initial_RY90_pulses[qubit] = platform.create_RX90_pulse(
            qubit, start=0, relative_phase=np.pi / 2
        )

        if not flux_pulse_shape:
            if flux_pulse_shapes and len(flux_pulse_shapes) == len(targets):
                flux_pulse_shape = eval(flux_pulse_shapes[qubit])
            else:
                flux_pulse_shape = Rectangular()

        # wait before applying flux pulse (delay_before_flux)

        # apply a detuning flux pulse
        flux_pulses[qubit] = FluxPulse(
            start=initial_RY90_pulses[qubit].finish + delay_before_flux,
            duration=flux_pulse_duration_end,  # sweep to produce oscillations
            amplitude=flux_pulse_amplitude,  # fix for each run
            shape=flux_pulse_shape,
            channel=platform.qubits[qubit].flux.name,
            qubit=platform.qubits[qubit].name,
        )

        # wait after applying flux pulse (delay_after_flux)

        # rotate around the X asis Rx(-pi/2) to meassure Y component
        RX90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=flux_pulses[qubit].finish + delay_after_flux,
            relative_phase=np.pi,
        )

        # rotate around the Y asis Ry(pi/2) to meassure X component
        RY90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=flux_pulses[qubit].finish + delay_after_flux,
            relative_phase=np.pi / 2,
        )

        # add ro pulse at the end of each sequence
        MZ_ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses[qubit].finish
        )

        # add pulses to the sequences
        MX_seq.add(
            initial_RY90_pulses[qubit],
            flux_pulses[qubit],
            RY90_pulses[qubit],
            MZ_ro_pulses[qubit],
        )
        MY_seq.add(
            initial_RY90_pulses[qubit],
            flux_pulses[qubit],
            RX90_pulses[qubit],
            MZ_ro_pulses[qubit],
        )
        MZ_seq.add(
            initial_RY90_pulses[qubit],
            flux_pulses[qubit],
            MZ_ro_pulses[qubit],
        )
        M0_seq.add(
            MZ_ro_pulses[qubit],
        )
        M1_seq.add(
            platform.create_RX_pulse(qubit, start=0, relative_phase=np.pi / 2),
            MZ_ro_pulses[qubit],
        )

        # DEBUG: Plot Cryoscope Sequences
        print("MX_seq\n", MX_seq)
        print("MY_seq\n", MY_seq)
        print("MZ_seq\n", MZ_seq)
        print("M0_seq\n", M0_seq)
        print("M1_seq\n", M1_seq)

        # define the parameters to sweep and their range:
        # flux pulse duration

        flux_pulse_duration_range = np.arange(
            flux_pulse_duration_start, flux_pulse_duration_end, flux_pulse_duration_step
        )
        duration_sweeper = Sweeper(
            Parameter.duration,
            flux_pulse_duration_range,
            # pulses=[flux_pulses[qubit] for qubit in qubits],
            pulses=[flux_pulses[qubit]],
            type=SweeperType.ABSOLUTE,
        )

        # execute the pulse sequences
        for sequence, tag in [
            (MX_seq, MX_tag),
            # (MY_seq, MY_tag),
            # (MZ_seq, MZ_tag),
            # (M0_seq, M0_tag),
            # (M1_seq, M1_tag),
        ]:
            results: AveragedSampleResults = platform.sweep(
                sequence,
                ExecutionParameters(
                    nshots=nshots,
                    relaxation_time=relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
                duration_sweeper,
            )
            # store the results
            data.register_qubit(
                qubit,
                component=tag,
                flux_pulse_duration=flux_pulse_duration_range,
                flux_pulse_amplitude=np.array([flux_pulse_amplitude]),
                probability=results[MZ_ro_pulses[qubit].serial].probability(state=1),
            )
    return data


def _plot(data: CryscopeData, fit: CryscopeResults, target: QubitId):
    """Plot the experiment result for a single pair."""
    fig = go.Figure()
    for component in TAGS:
        qc_data = data[target, component]
        fig.add_trace(
            go.Scatter(
                x=qc_data["flux_pulse_duration"],
                y=qc_data["probability"],
                name=f"{component}",
            )
        )

        fig.update_layout(
            title=f"Cryoscope - amplitude: {qc_data['flux_pulse_amplitude'][0]}",
            xaxis_title="Duration [ns]",
            yaxis_title="Probability [dimensionless]",
            legend_title="Components",
        )

    fit_report = ""

    return [fig], fit_report


def _fit(data: CryscopeData):
    return CryscopeResults(filter_coefs=[])


cryoscope = Routine(_aquisition, _fit, _plot)
"""Cryscope routine."""