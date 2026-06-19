from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibocal.protocols.coherence.spin_echo import SpinEchoParameters
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper
from qibolab._core.native import SingleQubitNatives
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.result import probability

from ..utils import table_dict, table_html
from .utils import dynamical_decoupling_sequence, single_exponential_fit

MAX_GATES = 4096
"""Maximum number of total pulses/delays allowed across unrolled sequences."""


KAPPA_GUESS = 0.25
"""Initial guess for the cavity decay rate ``kappa`` [MHz] (typical readout linewidth ~250 kHz)."""


def noise_psd_model(f, a, alpha, b, c, kappa, epsilon):
    """Decay rate ``Gamma2 = 1 / T2`` as a function of the CPMG filter frequency.

    ``a * f ** -alpha`` captures the 1/f-like noise typically dominating at low
    frequency (``alpha`` close to 1); ``b`` is a frequency-independent floor (e.g.
    set by ``T1`` processes); ``c * kappa**2 / (f**2 + kappa**2)`` is the
    thermal-photon-induced dephasing contribution, which rolls off once the CPMG
    filter frequency exceeds the readout cavity decay rate ``kappa`` (lumping the
    dispersive shift, photon occupation and filtering factor into the amplitude
    ``c``, since only ``kappa`` itself is of physical interest here).
    
    ``epsilon * f`` is a phenomenological penalty term capturing the accumulation 
    of control errors (and potential heating effects) from the increasing number 
    of pi-pulses required at higher filter frequencies (shorter delays).
    """
    return (a * f**-alpha) + (c * kappa**2 / (f**2 + kappa**2)) + b + (epsilon * f)


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
    """Number of points (distinct sequences) for each ``tau``, to fit the ``T2`` decay.

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
    psd_fitted_parameters: dict[QubitId, list[float]] = field(default_factory=dict)
    """Noise PSD model (``noise_psd_model``) fitted parameters ``[a, alpha, b, c, kappa]``."""

    def __contains__(self, target: QubitId) -> bool:
        """Check if a qubit has been fitted for at least one ``tau``.
        """
        return all(target in key for key in self.t2)


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

    data: dict[tuple[QubitId, int]] = field(
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

    One sequence is built per number of CPMG pulses ``n``, and the inter-pulse
    delay ``tau`` is swept within each sequence using a hardware ``Sweeper``
    instead of rebuilding the sequences in a Python loop. The time axis used
    to fit ``T2`` is the actual elapsed time of the sequence

        ``t = n * (tau + RY.duration) + 2 * RX90.duration``,
    """
    data = CpmgSpectroscopyData()

    tau_range = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )
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

    # n_max shrinks as tau grows, so using the largest tau keeps n * tau
    # within max_duration for the whole swept range.
    n_max = int(params._max_duration // (tau_range.max() + ry_duration[targets[0]]))
    if n_max < params.min_number_pulses:
        n_values = np.array([params.min_number_pulses])
    else:
        span = n_max - params.min_number_pulses + 1
        step = max(1, int(np.ceil(span / params.max_points_per_tau)))
        n_values = np.arange(params.min_number_pulses, n_max + 1, step).astype(int)

    sequences = []
    ro_pulses = []
    delays_per_n = []
    for n in n_values:
        _sequence, _delays = dynamical_decoupling_sequence(
            platform, targets, wait = tau_range[0]//2, n=n, kind="CPMG"
        )
        _ro_pulses = {
            qubit: list(_sequence.channel(platform.qubits[qubit].acquisition))[-1]
            for qubit in targets
        }
        sequences.append(_sequence)
        ro_pulses.append(_ro_pulses)
        delays_per_n.append(_delays)

    if params.unrolling:
        n_gates = sum(len(sequence) for sequence in sequences)
        assert n_gates < MAX_GATES, (
            f"Unrolling {len(sequences)} sequences requires {n_gates} gates, "
            f"exceeding the maximum of {MAX_GATES}. Reduce ``max_points_per_tau`` "
            "or ``max_duration``."
        )
        sweeper = Sweeper(
            parameter=Parameter.duration,
            values=tau_range // 2,
            pulses=[delay for delays in delays_per_n for delay in delays],
        )
        results = platform.execute(sequences, [[sweeper]], **options)
    else:
        results = {}
        for sequence, delays in zip(sequences, delays_per_n):
            sweeper = Sweeper(
                parameter=Parameter.duration, values=tau_range // 2, pulses=delays
            )
            results.update(platform.execute([sequence], [[sweeper]], **options))

    for n, _ro_pulses in zip(n_values, ro_pulses):
        for qubit in targets:
            prob = probability(results[_ro_pulses[qubit].id], state=1)
            error = np.sqrt(prob * (1 - prob) / params.nshots)
            wait = n * (tau_range + ry_duration[qubit]) + 2 * rx90_duration[qubit]
            for i, tau in enumerate(tau_range):
                data.register_qubit(
                    CpmgSpectroscopyType,
                    (qubit, float(tau)),
                    dict(
                        n=np.array([n]),
                        wait=np.array([wait[i]]),
                        prob=np.array([prob[i]]),
                        error=np.array([error[i]]),
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
            log.warning(f"Exponential decay fit failed for {key}. {e}")
            t2s[key] = [np.nan, np.nan]
            fitted_parameters[key] = [np.nan, np.nan, np.nan]
            pcovs[key] = np.full((3, 3), np.nan).tolist()
            chi2[key] = (np.nan, np.nan)

    psd_fitted_parameters = {}
    for qubit in data.qubits:
        taus = np.array(
            [tau for (q, tau) in t2s if q == qubit and np.isfinite(t2s[(q, tau)][0])]
        )
        if len(taus) < 5:
            log.warning(f"Not enough points to fit noise PSD for qubit {qubit}.")
            psd_fitted_parameters[qubit] = [np.nan] * 5
            continue

        freq = 1 / (2*np.array(taus))*1e3  # MHz, CPMG filter peaks at f = 1 / (2 * tau)
        t2_values = np.array([t2s[(qubit, tau)][0] for tau in taus])
        t2_errors = np.array([t2s[(qubit, tau)][1] for tau in taus])
        gamma2 = 1 / t2_values
        gamma2_errors = t2_errors / t2_values**2

        reliable = np.isfinite(gamma2_errors) & (gamma2_errors <= gamma2)
        if np.sum(reliable) < 5:
            log.warning(f"Not enough reliable points to fit noise PSD for qubit {qubit}.")
            psd_fitted_parameters[qubit] = [np.nan] * 5
            continue

        try:
            popt, _ = curve_fit(
                noise_psd_model,
                freq[reliable],
                gamma2[reliable],
                p0=[1, 1, 0, np.max(gamma2[reliable]), KAPPA_GUESS, 0],
                bounds=(0, np.inf),
                sigma=gamma2_errors[reliable],
                maxfev=200000,
            )
            psd_fitted_parameters[qubit] = popt.tolist()
        except Exception as e:
            log.warning(f"Noise PSD fit failed for qubit {qubit}. {e}")
            psd_fitted_parameters[qubit] = [np.nan] * 5

    return CpmgSpectroscopyResults(
        t2s, fitted_parameters, pcovs, chi2, psd_fitted_parameters
    )


def _plot(
    data: CpmgSpectroscopyData, target: QubitId, fit: CpmgSpectroscopyResults = None
):
    """Plotting function for CpmgSpectroscopy."""

    fitting_report = ""
    taus = data.taus

    n_values = np.unique(data[target, taus[0]].n)
    prob_matrix = np.full((len(n_values), len(taus)), np.nan)
    for j, tau in enumerate(taus):
        qubit_data = data[target, tau]
        order = np.argsort(qubit_data.n)
        prob_matrix[:, j] = qubit_data.prob[order]

    decay_fig = go.Figure(
        go.Contour(
            x=taus,
            y=n_values,
            z=prob_matrix,
            colorscale="Viridis",
            colorbar=dict(title="P(1)"),
            hovertemplate=(
                "tau = %{x:.0f} ns<br>"
                "n = %{y:.0f}<br>"
                "P(1) = %{z:.3f}<extra></extra>"
            ),
        ),
    )

    decay_fig.update_layout(
        xaxis_title="Inter-pulse delay tau [ns]",
        yaxis_title="Number of pulses n",
    )

    duration_fig = go.Figure(
        go.Scatter(
            x=taus,
            y=[data[target, tau].wait.max() for tau in taus],
            mode="markers+lines",
            name="Total duration",
        )
    )
    duration_fig.update_layout(
        xaxis_title="Inter-pulse delay tau [ns]",
        yaxis_title="Total T2 experiment duration [ns]",
    )

    figures = [decay_fig, duration_fig]

    if fit is not None:
        valid_taus = [tau for tau in taus if (target, tau) in fit.t2]
        # CPMG filter function peaks at f = 1 / (2 * tau).
        filter_freq = 1 / (2*np.array(valid_taus))*1e3 # MHz, tau in ns
        order = np.argsort(filter_freq)
        filter_freq = filter_freq[order]
        t2_values = np.array([fit.t2[(target, tau)][0] for tau in valid_taus])[order]
        t2_errors = np.array([fit.t2[(target, tau)][1] for tau in valid_taus])[order]

        # Skip points whose fit error is too large to be informative.
        reliable = np.isfinite(t2_errors) & (t2_errors <= t2_values)
        filter_freq = filter_freq[reliable]
        t2_values = t2_values[reliable]
        t2_errors = t2_errors[reliable]

        gamma2 = 1 / t2_values
        gamma2_errors = t2_errors / t2_values**2

        t2_fig = go.Figure(
            go.Scatter(
                x=filter_freq,
                y=gamma2,
                error_y=dict(type="data", array=gamma2_errors),
                mode="markers",
                name="$\Gamma_2$ (filter frequency)",
            )
        )
        psd_params = fit.psd_fitted_parameters.get(target, [np.nan] * 5)
        if len(filter_freq) > 0 and np.all(np.isfinite(psd_params)):
            freq_fit = np.geomspace(filter_freq.min(), filter_freq.max(), 200)
            gamma2_fit = noise_psd_model(freq_fit, *psd_params)
            t2_fig.add_trace(
                go.Scatter(
                    x=freq_fit,
                    y=gamma2_fit,
                    mode="lines",
                    name="Noise PSD fit",
                    line=go.scatter.Line(dash="dash"),
                ),
            )

        t2_fig.update_layout(
            showlegend=True,
            xaxis_title="CPMG filter frequency [MHz]",
            yaxis_title="Gamma2 [1/ns]",
            xaxis_type="log",
            yaxis_type="log",
        )
        figures.append(t2_fig)

        fitting_report = table_html(
            table_dict(
                target,
                [f"T2 [ns] (tau={tau:.0f} ns)" for tau in [valid_taus[0], valid_taus[-1]]],
                [fit.t2[(target, tau)] for tau in [valid_taus[0], valid_taus[-1]]],
                display_error=True,
            )
        )
        if np.all(np.isfinite(psd_params)):
            fitting_report += table_html(
                table_dict(
                    target,
                    [
                        "Noise PSD: A",
                        "Noise PSD: alpha (1/f exponent)",
                        "Noise PSD: B (white-noise floor)",
                        "Noise PSD: C (thermal-photon amplitude)",
                        "Noise PSD: kappa (cavity decay rate) [MHz]",
                    ],
                    psd_params,
                )
            )

    return figures, fitting_report


cpmg_spectroscopy = Routine(_acquisition, _fit, _plot)
"""CpmgSpectroscopy Routine object."""
