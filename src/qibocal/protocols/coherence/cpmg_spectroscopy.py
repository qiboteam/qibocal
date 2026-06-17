from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibocal.protocols.coherence.spin_echo import SpinEchoParameters
from qibolab import AcquisitionType, AveragingMode
from qibolab._core.native import SingleQubitNatives

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.result import probability

from ..utils import table_dict, table_html
from .utils import dynamical_decoupling_sequence, exp_decay, single_exponential_fit

__all__ = ["cpmg_spectroscopy"]


@dataclass
class CpmgSpectroscopyParameters(SpinEchoParameters):
    """CpmgSpectroscopy runcard inputs."""

    max_duration: int | None = None
    """Maximum total free evolution time ``n * tau`` [ns] explored for each ``tau``.
    """
    min_number_pulses: int = 1
    """Minimum number of CPMG pulses swept for each ``tau``. ``1`` corresponds to a spin-echo sequence.
    """
    max_points_per_tau: int = 20
    """Maximum number of distinct sequences (number of pulses values) swept for each ``tau``.

    The step in the number of pulses ``n`` is chosen adaptively, for each ``tau``, so that at most this many points are sampled between
    ``min_number_pulses`` and the maximum number of pulses allowed by ``max_duration``. 
    """

    unrolling: bool = True
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    """

    @property
    def _max_duration(self) -> int:
        """Coerced maximum total free evolution time ``n * tau`` [ns] explored for each ``tau``."""
        return (
            self.max_duration
            if self.max_duration is not None
            else self.min_number_pulses * self.delay_between_pulses_end
        )

@dataclass
class CpmgSpectroscopyResults(Results):
    """CpmgSpectroscopy outputs."""

    t2: dict[tuple[QubitId, int], list[float]]
    """T2 filtered at the CPMG passband centered at each ``tau``."""
    fitted_parameters: dict[tuple[QubitId, int], list[float]]
    """Raw fitting output."""
    pcov: dict[tuple[QubitId, int], list[float]]
    """Approximate covariance of fitted parameters."""
    chi2: dict[tuple[QubitId, int], tuple[float, float | None]]
    """Chi squared estimate mean value and error."""

    def __contains__(self, target: QubitId) -> bool:
        """Check if a qubit has been fitted for at least one ``tau``.
        """
        return any(key[0] == target for key in self.t2)


CpmgSpectroscopyType = np.dtype(
    [
        ("n", np.float64),
        ("wait", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
    ]
)
"""Custom dtype for CpmgSpectroscopy."""


@dataclass
class CpmgSpectroscopyData(Data):
    """CpmgSpectroscopy acquisition outputs."""

    data: dict[tuple[QubitId, int], npt.NDArray[CpmgSpectroscopyType]] = field(
        default_factory=dict
    )
    """Raw data acquired, keyed by ``(qubit, tau)``."""

    @property
    def taus(self):
        """Access tau values from data structure."""
        return np.unique([key[1] for key in self.data])


def _acquisition(
    params: CpmgSpectroscopyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> CpmgSpectroscopyData:
    """Data acquisition for CpmgSpectroscopy.

    For each fixed inter-pulse delay ``tau`` (outer loop) the number of CPMG
    pulses ``n`` is swept (inner loop). The time axis used to fit ``T2`` is
    the actual elapsed time of the sequence

        ``t = n * (tau + RY.duration) + 2 * RX90.duration``,
    """
    data = CpmgSpectroscopyData()

    tau_range = np.arange(params.delay_between_pulses_start, params.delay_between_pulses_end, params.delay_between_pulses_step)
    rx90_duration = {}
    ry_duration = {}
    for qubit in targets:
        natives: SingleQubitNatives = platform.natives.single_qubit[qubit]
        rx90_duration[qubit] = natives.R(theta=np.pi / 2).duration
        ry_duration[qubit] = natives.R(phi=np.pi / 2).duration

    options = dict(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    for tau in tau_range:
        n_max = int(params._max_duration // (tau + ry_duration[targets[0]]))  # conservative estimate using the first target qubit)
        if n_max < params.min_number_pulses:
            n_values = np.array([params.min_number_pulses])
        else:
            span = n_max - params.min_number_pulses + 1
            step = max(1, int(np.ceil(span / params.max_points_per_tau)))
            n_values = np.arange(
                params.min_number_pulses, n_max + 1, step
            ).astype(int)

        sequences = []
        delays = []
        ro_pulses = []
        for n in n_values:
            _sequence, _delays = dynamical_decoupling_sequence(
                platform, targets, wait=tau / 2, n=n, kind="CPMG"
            )
            _ro_pulses = {
                qubit: list(_sequence.channel(platform.qubits[qubit].acquisition))[-1]
                for qubit in targets
            }
            delays.append(_delays)
            sequences.append(_sequence)
            ro_pulses.append(_ro_pulses)

        if params.unrolling:
            results = platform.execute(sequences, **options)
        else:
            results = {}
            for sequence in sequences:
                results.update(platform.execute([sequence], **options))

        for n, _ro_pulses in zip(n_values, ro_pulses):
            for qubit in targets:
                prob = probability(results[_ro_pulses[qubit].id], state=1)
                error = np.sqrt(prob * (1 - prob) / params.nshots)
                data.register_qubit(
                    CpmgSpectroscopyType,
                    (qubit, float(tau)),
                    dict(
                        n=np.array([n]),
                        wait=np.array(
                            [
                                n * (tau + ry_duration[qubit])
                                + 2 * rx90_duration[qubit]
                            ]
                        ),
                        prob=np.array([prob]),
                        error=np.array([error]),
                    ),
                )
    return data


def _fit(data: CpmgSpectroscopyData) -> CpmgSpectroscopyResults:
    """Post-processing for CpmgSpectroscopy.

    A T2 decay is fitted independently for each ``tau``, extracting the
    coherence time filtered at the CPMG passband centered at that ``tau``.
    """
    t2s, fitted_parameters, pcovs, chi2 = {}, {}, {}, {}

    for key in data.data:
        try:
            t2s[key], fitted_parameters[key], pcovs[key], chi2[key] = (
                single_exponential_fit(
                    data[key].wait,
                    data[key].prob,
                    data[key].error,
                )
            )
        except Exception as e:
            log.warning(f"Exponential decay fit failed for {key} due to {e}")
            t2s[key] = [np.nan, np.nan]
            fitted_parameters[key] = [np.nan, np.nan, np.nan]
            pcovs[key] = np.full((3, 3), np.nan)
            chi2[key] = (np.nan, np.nan)
    return CpmgSpectroscopyResults(t2s, fitted_parameters, pcovs, chi2)


def _plot(
    data: CpmgSpectroscopyData, target: QubitId, fit: CpmgSpectroscopyResults = None
):
    """Plotting function for CpmgSpectroscopy."""

    fitting_report = ""
    taus = data.taus

    decay_fig = go.Figure()
    for tau in taus:
        qubit_data = data[target, tau]
        order = np.argsort(qubit_data.wait)
        waits = qubit_data.wait[order]
        probs = qubit_data.prob[order]
        ns = qubit_data.n[order]

        decay_fig.add_trace(
            go.Scatter(
                x=waits,
                y=probs,
                mode="markers",
                name=f"tau = {tau:.0f} ns",
                line=go.scatter.Line(dash="dot"),
                customdata=ns,
                hovertemplate=(
                    "t = %{x:.0f} ns<br>"
                    "P(1) = %{y:.3f}<br>"
                    "n = %{customdata:.0f}<extra></extra>"
                ),
            ),
        )

        if fit is not None and (target, tau) in fit.fitted_parameters:
            waitrange = np.linspace(min(waits), max(waits), 2 * len(waits))
            fit_params = fit.fitted_parameters[(target, tau)]
            decay_fig.add_trace(
                go.Scatter(
                    x=waitrange,
                    y=exp_decay(waitrange, *fit_params),
                    name=f"Fit tau = {tau:.0f} ns",
                    line=go.scatter.Line(dash="dot"),
                    showlegend=False,
                ),
            )

    decay_fig.update_layout(
        showlegend=True,
        xaxis_title="Total sequence duration [ns]",
        yaxis_title="Probability of State 1",
    )

    figures = [decay_fig]

    if fit is not None:
        valid_taus = [tau for tau in taus if (target, tau) in fit.t2]
        t2_values = [fit.t2[(target, tau)][0] for tau in valid_taus]
        t2_errors = [fit.t2[(target, tau)][1] for tau in valid_taus]

        t2_fig = go.Figure(
            go.Scatter(
                x=valid_taus,
                y=t2_values,
                error_y=dict(type="data", array=t2_errors),
                mode="markers+lines",
                name="T2 (tau)",
            )
        )
        t2_fig.update_layout(
            showlegend=True,
            xaxis_title="Inter-pulse delay tau [ns]",
            yaxis_title="T2 [ns]",
        )
        figures.append(t2_fig)

        fitting_report = table_html(
            table_dict(
                target,
                [f"T2 [ns] (tau={tau:.0f} ns)" for tau in valid_taus],
                [fit.t2[(target, tau)] for tau in valid_taus],
                display_error=True,
            )
        )

    return figures, fitting_report


cpmg_spectroscopy = Routine(_acquisition, _fit, _plot)
"""CpmgSpectroscopy Routine object."""
