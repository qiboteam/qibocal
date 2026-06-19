from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper
from qibolab._core.native import SingleQubitNatives
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.coherence.spin_echo import SpinEchoParameters

from ..utils import table_dict, table_html
from .utils import dynamical_decoupling_sequence, single_exponential_fit

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
    n_points: int = 20
    """Number of points used to sweep the number of CPMG pulses ``n``, log-spaced between
    ``1`` (a spin-echo sequence, which fits the full ``tau`` range within ``max_duration``)
    and the maximum number of flips required to span ``max_duration`` at the smallest
    swept ``tau``.
    """

    @property
    def _max_duration(self) -> int:
        """Coerced maximum total free evolution time ``n * tau`` [ns] explored for each ``tau``."""
        return (
            self.max_duration
            if self.max_duration is not None
            else self.delay_between_pulses_end
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
    """Noise PSD model (``noise_psd_model``) fitted parameters ``[a, alpha, b, c, kappa, epsilon]``."""

    def __contains__(self, target: QubitId) -> bool:
        """Check if a qubit has been fitted for at least one ``tau``."""
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

    data: dict[tuple[QubitId, int]] = field(default_factory=dict)
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

    One sequence is built per number of CPMG pulses ``N``, and the inter-pulse
    delay ``tau`` is swept within each sequence using a hardware ``Sweeper``.
    The time axis used to fit ``T2`` is the actual elapsed time of the sequence

        ``t = N * (tau + RY.duration) + 2 * RX90.duration``,

    which is increasing in both ``N`` and ``tau``. To keep every probed point
    within ``max_duration``, each ``N`` only sweeps the subset of ``tau``
    values for which ``t <= max_duration`` -- the longest sequences (largest
    ``N``) therefore only run for the shortest ``tau``, while ``N = 1`` (a
    spin-echo sequence) already fits the full ``tau`` range. ``N`` itself is
    swept between ``1`` and the number of flips needed to reach
    ``max_duration`` at the smallest ``tau``.

    Since each ``N`` sweeps a different number of ``tau`` points, sequences cannot
    share a single unrolled ``Sweeper`` call and are executed one at a time.
    TODO: Make sure this is true
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
        averaging_mode=AveragingMode.CYCLIC,
    )

    ry = ry_duration[targets[0]]
    rx90 = rx90_duration[targets[0]]
    # n = 1 (spin echo) always fits the full tau range within max_duration; the
    # smallest tau (hardest case) sets how many flips can be swept before even
    # it exceeds max_duration.
    n_max = max(1, int((params._max_duration - 2 * rx90) // (tau_range.min() + ry)))
    n_values = np.unique(np.round(np.geomspace(1, n_max, params.n_points)).astype(int))
    log.info(
        f"CPMG spectroscopy: sweeping {len(n_values)} n values between 1 and {n_max}."
    )

    results = {}
    ro_pulses = []
    taus_per_n = []
    for N in n_values:
        # Only the tau values for which this N still fits within max_duration
        # are swept; this shrinks as N grows, so larger N  probes fewer (and
        # shorter) delays.
        tau_cutoff = (params._max_duration - 2 * rx90) / N - ry
        taus_n = tau_range[tau_range <= tau_cutoff]
        if len(taus_n) == 0:
            taus_n = np.array(
                [tau_range[0]]
            )  # at least one tau value is needed to build a sequence

        log.info(f"Sweeping {len(taus_n)} taus for N={N} CPMG pulses.")

        single_tau = len(taus_n) == 1
        _sequence, _delays = dynamical_decoupling_sequence(
            platform,
            targets,
            wait=(taus_n[0] if single_tau else tau_range[0]) // 2,
            n=N,
            kind="CPMG",
        )
        _ro_pulses = {
            qubit: list(_sequence.channel(platform.qubits[qubit].acquisition))[-1]
            for qubit in targets
        }

        results.update(
            platform.execute(
                [_sequence],
                None
                if single_tau
                else [
                    [
                        Sweeper(
                            parameter=Parameter.duration,
                            values=taus_n // 2,
                            pulses=_delays,
                        )
                    ]
                ],
                **options,
            )
        )

        ro_pulses.append(_ro_pulses)
        taus_per_n.append(taus_n)

    for n, _ro_pulses, taus_n in zip(n_values, ro_pulses, taus_per_n):
        for qubit in targets:
            # A single-tau sequence is executed without a Sweeper, so its result
            # is a 0-d scalar rather than a 1-element array.
            prob = np.atleast_1d(results[_ro_pulses[qubit].id])
            error = np.sqrt(prob * (1 - prob) / params.nshots)
            wait = N * (taus_n + ry_duration[qubit]) + 2 * rx90_duration[qubit]
            for i, tau in enumerate(taus_n):
                data.register_qubit(
                    CpmgSpectroscopyType,
                    (qubit, float(tau)),
                    dict(
                        N=np.array([N]),
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
            psd_fitted_parameters[qubit] = [np.nan] * 6
            continue

        freq = (
            1 / (2 * np.array(taus)) * 1e3
        )  # MHz, CPMG filter peaks at f = 1 / (2 * tau) # Do I need to add here the pusle duration as well?
        t2_values = np.array([t2s[(qubit, tau)][0] for tau in taus])
        t2_errors = np.array([t2s[(qubit, tau)][1] for tau in taus])
        gamma2 = 1 / t2_values
        gamma2_errors = t2_errors / t2_values**2

        reliable = np.isfinite(gamma2_errors) & (gamma2_errors <= gamma2)
        if np.sum(reliable) < 5:
            log.warning(
                f"Not enough reliable points to fit noise PSD for qubit {qubit}."
            )
            psd_fitted_parameters[qubit] = [np.nan] * 6
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
            psd_fitted_parameters[qubit] = [np.nan] * 6

    return CpmgSpectroscopyResults(
        t2s, fitted_parameters, pcovs, chi2, psd_fitted_parameters
    )


def _plot(
    data: CpmgSpectroscopyData, target: QubitId, fit: CpmgSpectroscopyResults = None
):
    """Plotting function for CpmgSpectroscopy."""

    fitting_report = ""
    taus = data.taus

    # Each tau is probed at its own set of elapsed times (n * (tau + RY.duration) +
    # 2 * RX90.duration), so the (tau, time) grid is irregular. Interpolating the
    # scattered points onto a regular grid lets us still draw a contour.
    plot_tau, plot_time, plot_prob = [], [], []
    for tau in taus:
        qubit_data = data[target, tau]
        plot_tau.extend([tau] * len(qubit_data.wait))
        plot_time.extend(qubit_data.wait)
        plot_prob.extend(qubit_data.prob)

    tau_grid = np.linspace(min(plot_tau), max(plot_tau), 100)
    time_grid = np.linspace(0, max(plot_time), 100)
    tau_mesh, time_mesh = np.meshgrid(tau_grid, time_grid)
    prob_grid = griddata(
        (plot_tau, plot_time), plot_prob, (tau_mesh, time_mesh), method="linear"
    )

    decay_fig = go.Figure(
        go.Heatmap(
            x=tau_grid,
            y=time_grid,
            z=prob_grid,
            colorscale="Viridis",
            colorbar=dict(title="P(1)"),
            hovertemplate=(
                "tau = %{x:.0f} ns<br>time = %{y:.0f} ns<br>"
                "P(1) = %{z:.3f}<extra></extra>"
            ),
        ),
    )

    decay_fig.update_layout(
        xaxis_title="Inter-pulse delay tau [ns]",
        yaxis_title="Total free evolution time [ns]",
    )

    figures = [decay_fig, duration_fig]

    if fit is not None:
        valid_taus = [tau for tau in taus if (target, tau) in fit.t2]
        # CPMG filter function peaks at f = 1 / (2 * tau).
        filter_freq = 1 / (2 * np.array(valid_taus)) * 1e3  # MHz, tau in ns
        order = np.argsort(filter_freq)
        taus_ordered = np.array(valid_taus)[order]
        filter_freq = filter_freq[order]
        t2_values = np.array([fit.t2[(target, tau)][0] for tau in valid_taus])[order]
        t2_errors = np.array([fit.t2[(target, tau)][1] for tau in valid_taus])[order]

        # Skip points whose fit error is too large to be informative.
        reliable = np.isfinite(t2_errors) & (t2_errors <= t2_values)
        taus_ordered = taus_ordered[reliable]
        filter_freq = filter_freq[reliable]
        t2_values = t2_values[reliable]
        t2_errors = t2_errors[reliable]

        t2_tau_fig = go.Figure(
            go.Scatter(
                x=taus_ordered,
                y=t2_values,
                error_y=dict(type="data", array=t2_errors),
                mode="markers",
                name="T2",
            )
        )
        if len(t2_values) > 0:
            # Range set from the T2 values themselves, not stretched out by
            # the (sometimes much larger) error bars.
            margin = 0.1 * (t2_values.max() - t2_values.min())
            t2_tau_fig.update_layout(
                yaxis=dict(range=[t2_values.min() - margin, t2_values.max() + margin])
            )
        t2_tau_fig.update_layout(
            xaxis_title="Inter-pulse delay tau [ns]",
            yaxis_title="T2 [ns]",
        )
        figures.append(t2_tau_fig)

        gamma2 = 1 / t2_values
        gamma2_errors = t2_errors / t2_values**2

        t2_fig = go.Figure(
            go.Scatter(
                x=filter_freq,
                y=gamma2,
                error_y=dict(type="data", array=gamma2_errors),
                mode="markers",
                name=r"$\Gamma_2$ (filter frequency)",
            )
        )
        psd_params = fit.psd_fitted_parameters.get(target, [np.nan] * 6)
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
                [
                    f"T2 [ns] (tau={tau:.0f} ns)"
                    for tau in [valid_taus[0], valid_taus[-1]]
                ],
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
                        "Noise PSD: epsilon (pulse-count penalty)",
                    ],
                    psd_params,
                )
            )

    return figures, fitting_report


cpmg_spectroscopy = Routine(_acquisition, _fit, _plot)
"""CpmgSpectroscopy Routine object."""
