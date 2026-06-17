from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.result import probability

from ..utils import table_dict, table_html
from .utils import dynamical_decoupling_sequence, exp_decay, single_exponential_fit

__all__ = ["cpmg_spectroscopy"]


@dataclass
class CpmgSpectroscopyParameters(Parameters):
    """CpmgSpectroscopy runcard inputs."""

    tau_min: int
    """Initial fixed delay between consecutive pulses [ns]."""
    tau_max: int
    """Final fixed delay between consecutive pulses [ns]."""
    tau_step: int
    """Step of the fixed delay between consecutive pulses [ns]."""
    min_number_pulses: int = 1
    """Minimum number of CPMG pulses swept for each ``tau``.

    ``1`` corresponds to a spin-echo sequence.
    """
    max_duration: int | None = None
    """Maximum total free evolution time ``n * tau`` [ns] explored for each ``tau``.

    If ``None`` it is set to ``min_number_pulses * tau_max``, so that even the
    largest ``tau`` is sampled with at least ``min_number_pulses`` points.
    """
    n_step: int = 1
    """Step in the number of CPMG pulses ``n`` swept for each ``tau``."""
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    Defaults to ``False``."""


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
    pulses ``n`` is swept (inner loop), so that the total free evolution time
    ``t = n * tau`` plays the role of the time axis of a coherence decay
    filtered at the CPMG passband centered at that ``tau``. Fitting this
    decay for every ``tau`` provides ``T2`` as a function of the CPMG filter
    frequency, i.e. a coherence "spectroscopy".
    """
    data = CpmgSpectroscopyData()

    max_duration = (
        params.max_duration
        if params.max_duration is not None
        else params.min_number_pulses * params.tau_max
    )

    tau_range = np.arange(params.tau_min, params.tau_max, params.tau_step)

    options = dict(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    for tau in tau_range:
        n_max = int(max_duration // tau)
        if n_max < params.min_number_pulses:
            n_values = np.array([params.min_number_pulses])
        else:
            n_values = np.arange(
                params.min_number_pulses, n_max + 1, params.n_step
            )

        sequences = []
        all_ro_pulses = []
        for n in n_values:
            sequence, _ = dynamical_decoupling_sequence(
                platform, targets, wait=tau / 2, n=int(n), kind="CPMG"
            )
            ro_pulses = {
                qubit: list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
                for qubit in targets
            }
            sequences.append(sequence)
            all_ro_pulses.append(ro_pulses)

        if params.unrolling:
            results = platform.execute(sequences, **options)
        else:
            results = {}
            for sequence in sequences:
                results.update(platform.execute([sequence], **options))

        for n, ro_pulses in zip(n_values, all_ro_pulses):
            for qubit in targets:
                prob = probability(results[ro_pulses[qubit].id], state=1)
                error = np.sqrt(prob * (1 - prob) / params.nshots)
                data.register_qubit(
                    CpmgSpectroscopyType,
                    (qubit, float(tau)),
                    dict(
                        n=np.array([n]),
                        wait=np.array([n * tau]),
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

        decay_fig.add_trace(
            go.Scatter(
                x=waits,
                y=probs,
                mode="markers+lines",
                name=f"tau = {tau:.0f} ns",
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
        xaxis_title="Total free evolution time n * tau [ns]",
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
