import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import leastsq, minimize

from qibocal.auto.operation import Results

from ..utils import (
    COLORBAND,
    COLORBAND_LINE,
    DELAY_FIT_PERCENTAGE,
    HZ_TO_GHZ,
    PowerLevel,
    lorentzian,
    table_dict,
    table_html,
)

PHASES_THRESHOLD_PERCENTAGE = 80
r"""Threshold percentage to ensure the phase data covers a significant portion of the full 2 :math:\pi circle."""
STD_DEV_GAUSSIAN_KERNEL = 30
"""Standard deviation for the Gaussian kernel."""
PHASE_ELEMENTS = 5
"""Number of values to better guess :math:`\theta` (in rad) in the phase fit function."""


def s21(
    frequencies: NDArray,
    resonance: float,
    q_loaded: float,
    q_coupling: float,
    phi: float = 0.0,
    amplitude: float = 1.0,
    alpha: float = 0.0,
    tau: float = 0.0,
) -> NDArray:
    """Full model of the S21 notch resonator based on eq. (1) described in:
    "Efficient and robust analysis of complex scattering data under noise in microwave resonators"
    (https://doi.org/10.1063/1.4907935) by S. Probst et al and on eq. (E.1) described in:
    "The Physics of Superconducting Microwave Resonators"
    (https://doi.org/10.7907/RAT0-VM75) by J. Gao.

    The equation is split into two parts describing the ideal resonator and the environment.

    Args:
        frequencies (NDArray[float]): frequencies (Hz) at which the measurement was taken.
        resonance (float): resonance frequency (Hz).
        q_loaded (float): loaded quality factor.
        q_coupling (float): coupling quality factor.
        phi (float): quantifies the impedance mismatch (Fano interference).
        amplitude (float): accounts for additional attenuation/amplification present in the setup.
        alpha (float): accounts for a additional phase shift.
        tau (float): cable delay caused by the length of the cable and finite speed of light.

    Returns:
        S21 resonance profile array (NDArray) of a notch resonator.
    """
    return (
        amplitude
        * np.exp(1j * alpha)
        * np.exp(-2 * np.pi * 1j * frequencies * tau)
        * (
            1
            - ((q_loaded / (np.abs(q_coupling))) * np.exp(1j * phi))
            / (1 + 2j * q_loaded * (frequencies / resonance - 1))
        )
    )


def s21_fit(
    data: NDArray, resonator_type=None, fit=None
) -> tuple[float, list[float], list[float]]:
    """
    Calibrates the S21 profile of a notch resonator, based on https://github.com/qkitgroup/qkit.

        Args:
            data (NDArray[complex]): S21 scattering matrix element.
        Returns:
            Model parameters

    """
    f_data = data.freq
    z_data = np.abs(data.signal) * np.exp(1j * data.phase)

    num_points = int(len(f_data) * DELAY_FIT_PERCENTAGE / 100)
    tau = cable_delay(f_data, data.phase, num_points)
    z_1 = remove_cable_delay(f_data, z_data, tau)

    z_c, r_0 = circle_fit(z_1)
    z_2 = z_1 - z_c

    phases = np.unwrap(np.angle(z_2))

    resonance, q_loaded, theta = phase_fit(f_data, phases)
    beta = periodic_boundary(theta - np.pi)
    off_resonant_point = z_c + r_0 * np.cos(beta) + 1j * r_0 * np.sin(beta)

    amplitude = np.abs(off_resonant_point)
    alpha = np.angle(off_resonant_point)
    phi = periodic_boundary(beta - alpha)
    r_0_norm = r_0 / amplitude
    q_coupling = q_loaded / (2 * r_0_norm) / np.cos(phi)

    model_parameters = [
        resonance,
        q_loaded,
        q_coupling,
        phi,
        amplitude,
        alpha,
        tau,
    ]
    perr = [0.0] * 7

    return model_parameters[0], model_parameters, perr


def spectroscopy_plot(data, qubit, fit: Results = None):
    figures = []
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )
    qubit_data = data[qubit]
    fitting_report = ""
    frequencies = qubit_data.freq * HZ_TO_GHZ
    signal = qubit_data.signal

    phase = qubit_data.phase
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=signal,
            opacity=1,
            name="Frequency",
            showlegend=True,
            legendgroup="Frequency",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=phase,
            opacity=1,
            name="Phase",
            showlegend=True,
            legendgroup="Phase",
        ),
        row=1,
        col=2,
    )

    show_error_bars = not np.isnan(qubit_data.error_signal).any()
    if show_error_bars:
        errors_signal = qubit_data.error_signal
        errors_phase = qubit_data.error_phase
        fig.add_trace(
            go.Scatter(
                x=np.concatenate((frequencies, frequencies[::-1])),
                y=np.concatenate(
                    (signal + errors_signal, (signal - errors_signal)[::-1])
                ),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Signal Errors",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate((frequencies, frequencies[::-1])),
                y=np.concatenate((phase + errors_phase, (phase - errors_phase)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Phase Errors",
            ),
            row=1,
            col=2,
        )

    freqrange = np.linspace(
        min(frequencies),
        max(frequencies),
        2 * len(frequencies),
    )

    if fit is not None:
        params = fit.fitted_parameters[qubit]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorentzian(freqrange, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
            row=1,
            col=1,
        )

        if data.power_level is PowerLevel.low:
            label = "Readout Frequency [Hz]"
            freq = fit.frequency
        elif data.power_level is PowerLevel.high:
            label = "Bare Resonator Frequency [Hz]"
            freq = fit.bare_frequency
        else:
            label = "Qubit Frequency [Hz]"
            freq = fit.frequency

        if data.amplitudes[qubit] is not None:
            if show_error_bars:
                labels = [label, "Amplitude", "Chi2 reduced"]
                values = [
                    (
                        freq[qubit],
                        fit.error_fit_pars[qubit][1],
                    ),
                    (data.amplitudes[qubit], 0),
                    fit.chi2_reduced[qubit],
                ]
            else:
                labels = [label, "Amplitude"]
                values = [freq[qubit], data.amplitudes[qubit]]

            fitting_report = table_html(
                table_dict(
                    qubit,
                    labels,
                    values,
                    display_error=show_error_bars,
                )
            )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHz]",
        yaxis_title="Signal [a.u.]",
        xaxis2_title="Frequency [GHz]",
        yaxis2_title="Phase [rad]",
    )
    figures.append(fig)

    return figures, fitting_report


def s21_spectroscopy_plot(data, qubit, fit: Results = None):
    figures = []
    fig_raw = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        specs=[
            [{"rowspan": 2}, {}],
            [None, {}],
        ],
    )
    qubit_data = data[qubit]
    fitting_report = ""
    frequencies = qubit_data.freq
    signal = qubit_data.signal
    phase = qubit_data.phase
    phase = (
        -phase if data.phase_sign else phase
    )  # TODO: tmp patch for the sign of the phase
    phase = np.unwrap(phase)  # TODO: move phase unwrapping in qibolab
    s21_raw = np.abs(signal) * np.exp(1j * phase)
    fig_raw.add_trace(
        go.Scatter(
            x=np.real(s21_raw),
            y=np.imag(s21_raw),
            mode="markers",
            marker=dict(
                size=4,
            ),
            opacity=1,
            name="S21",
            showlegend=True,
            legendgroup="S21",
        ),
        row=1,
        col=1,
    )

    fig_raw.add_trace(
        go.Scatter(
            x=frequencies * HZ_TO_GHZ,
            y=signal,
            mode="markers",
            marker=dict(
                size=4,
            ),
            opacity=1,
            name="Magnitude",
            showlegend=True,
            legendgroup="Magnitude",
        ),
        row=1,
        col=2,
    )

    fig_raw.add_trace(
        go.Scatter(
            x=frequencies * HZ_TO_GHZ,
            y=phase,
            mode="markers",
            marker=dict(
                size=4,
            ),
            opacity=1,
            name="Phase",
            showlegend=True,
            legendgroup="Phase",
        ),
        row=2,
        col=2,
    )

    show_error_bars = not np.isnan(qubit_data.error_signal).any()

    if show_error_bars:
        errors_signal = qubit_data.error_signal
        errors_phase = qubit_data.error_phase
        fig_raw.add_trace(
            go.Scatter(
                x=np.concatenate((frequencies, frequencies[::-1])) * HZ_TO_GHZ,
                y=np.concatenate(
                    (signal + errors_signal, (signal - errors_signal)[::-1])
                ),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Signal Errors",
            ),
            row=1,
            col=2,
        )
        fig_raw.add_trace(
            go.Scatter(
                x=np.concatenate((frequencies, frequencies[::-1])) * HZ_TO_GHZ,
                y=np.concatenate((phase + errors_phase, (phase - errors_phase)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Phase Errors",
            ),
            row=2,
            col=2,
        )

    freqrange = np.linspace(
        min(frequencies),
        max(frequencies),
        2 * len(frequencies),
    )

    if fit is not None:
        params = fit.fitted_parameters[qubit]
        s21_fitted = s21(freqrange, *params)

        fig_raw.add_trace(
            go.Scatter(
                x=np.real(s21_fitted),
                y=np.imag(s21_fitted),
                opacity=1,
                name="S21 Fit",
                line=go.scatter.Line(dash="solid"),
            ),
            row=1,
            col=1,
        )
        fig_raw.add_trace(
            go.Scatter(
                x=freqrange * HZ_TO_GHZ,
                y=np.abs(s21_fitted),
                name="Magnitude Fit",
                line=go.scatter.Line(dash="solid"),
            ),
            row=1,
            col=2,
        )
        fig_raw.add_trace(
            go.Scatter(
                x=freqrange * HZ_TO_GHZ,
                y=np.unwrap(np.angle(s21_fitted)),
                name="Phase Fit",
                line=go.scatter.Line(dash="solid"),
            ),
            row=2,
            col=2,
        )

        if data.power_level is PowerLevel.low:
            label = "Readout Frequency [Hz]"
            freq = fit.frequency
        elif data.power_level is PowerLevel.high:
            label = "Bare Resonator Frequency [Hz]"
            freq = fit.bare_frequency
        else:
            label = "Qubit Frequency [Hz]"
            freq = fit.frequency

        if data.amplitudes[qubit] is not None:
            if show_error_bars:
                labels = [label, "Amplitude", "Chi2 Reduced"]
                values = [
                    (
                        freq[qubit],
                        fit.error_fit_pars[qubit][1],
                    ),
                    (data.amplitudes[qubit], 0),
                    fit.chi2_reduced[qubit],
                ]
            else:
                labels = [
                    label,
                    "Loaded Quality Factor",
                    "Internal Quality Factor",
                    "Coupling Quality Factor",
                    "Fano Interference [rad]",
                    "Amplitude [a.u.]",
                    "Phase Shift [rad]",
                    "Electronic Delay [s]",
                ]
                values = [
                    freq[qubit],
                    params[1],
                    1.0 / (1.0 / params[1] - 1.0 / params[2]),
                    params[2],
                    params[3],
                    params[4],
                    params[5],
                    params[6],
                ]

            fitting_report = table_html(
                table_dict(
                    qubit,
                    labels,
                    values,
                    display_error=show_error_bars,
                )
            )
        s21_calibrated = (
            s21_raw
            / params[4]
            * np.exp(1j * (-params[5] + 2.0 * np.pi * params[6] * frequencies))
        )
        fig_calibrated = make_subplots(
            rows=2,
            cols=2,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            specs=[
                [{"rowspan": 2}, {}],
                [None, {}],
            ],
        )

        fig_calibrated.add_trace(
            go.Scatter(
                x=np.real(s21_calibrated),
                y=np.imag(s21_calibrated),
                mode="markers",
                marker=dict(
                    size=4,
                ),
                opacity=1,
                name="S21",
                showlegend=True,
                legendgroup="S21",
            ),
            row=1,
            col=1,
        )

        fig_calibrated.add_trace(
            go.Scatter(
                x=frequencies * HZ_TO_GHZ,
                y=np.abs(s21_calibrated),
                mode="markers",
                marker=dict(
                    size=4,
                ),
                opacity=1,
                name="Transmission",
                showlegend=True,
                legendgroup="Transmission",
            ),
            row=1,
            col=2,
        )

        fig_calibrated.add_trace(
            go.Scatter(
                x=frequencies * HZ_TO_GHZ,
                y=np.unwrap(np.angle(s21_calibrated)),
                mode="markers",
                marker=dict(
                    size=4,
                ),
                opacity=1,
                name="Phase",
                showlegend=True,
                legendgroup="Phase",
            ),
            row=2,
            col=2,
        )

        freqrange = np.linspace(
            min(frequencies),
            max(frequencies),
            2 * len(frequencies),
        )
        s21_calibrated_fitted = s21(
            freqrange, params[0], params[1], params[2], params[3]
        )
        fig_calibrated.add_trace(
            go.Scatter(
                x=np.real(s21_calibrated_fitted),
                y=np.imag(s21_calibrated_fitted),
                opacity=1,
                name="S21 Fit",
                line=go.scatter.Line(dash="solid"),
            ),
            row=1,
            col=1,
        )
        fig_calibrated.add_trace(
            go.Scatter(
                x=freqrange * HZ_TO_GHZ,
                y=np.abs(s21_calibrated_fitted),
                name="Transmission Fit",
                line=go.scatter.Line(dash="solid"),
            ),
            row=1,
            col=2,
        )
        fig_calibrated.add_trace(
            go.Scatter(
                x=freqrange * HZ_TO_GHZ,
                y=np.unwrap(np.angle(s21_calibrated_fitted)),
                name="Phase Fit",
                line=go.scatter.Line(dash="solid"),
            ),
            row=2,
            col=2,
        )

        fig_calibrated.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
        fig_calibrated.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        fig_calibrated.update_layout(
            title="Calibrated data",
            showlegend=True,
            xaxis_title="Real [a.u.]",
            yaxis_title="Imaginary [a.u.]",
            xaxis2_title="",
            yaxis2_title="Transmission [a.u.]",
            xaxis3_title="Frequency [GHz]",
            yaxis3_title="Phase [rad]",
        )
        figures.append(fig_calibrated)

    fig_raw.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    fig_raw.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig_raw.update_layout(
        title="Raw data",
        showlegend=True,
        xaxis_title="Real [a.u.]",
        yaxis_title="Imaginary [a.u.]",
        xaxis2_title="",
        yaxis2_title="Magnitude [a.u.]",
        xaxis3_title="Frequency [GHz]",
        yaxis3_title="Phase [rad]",
    )
    figures.append(fig_raw)
    figures.reverse()

    return figures, fitting_report


def cable_delay(frequencies: NDArray, phases: NDArray, num_points: int) -> float:
    """Evaluates the cable delay :math:`\tau` (in s).

    The cable delay :math:`\tau` (in s) is caused by the length of the cable and the finite speed of light.
    This is estimated fitting a first-grade polynomial fit of the `phases` (in rad) as a function of the
    `frequencies` (in Hz), and extracting the angular coefficient, which is then expressed
    in seconds.

    The `num_points` is used to select how many points should be fitted, from both the
    start and the end of the frequency range.
    """
    frequencies_selected = np.concatenate(
        (frequencies[:num_points], frequencies[-num_points:])
    )
    phase_selected = np.concatenate((phases[:num_points], phases[-num_points:]))

    pvals = np.polyfit(frequencies_selected, phase_selected, 1)

    return pvals[0] / (-2 * np.pi)


def remove_cable_delay(frequencies: NDArray, z: NDArray, tau: float) -> NDArray:
    """Corrects the cable delay :math:`\tau` (in s).

    The cable delay :math:`\tau` (in s) is removed from the scattering matrix element array `z` by performing
    an exponential product which also depends from the `frequencies` (in Hz).
    """

    return z * np.exp(2j * np.pi * frequencies * tau)


def circle_fit(z: NDArray) -> tuple[complex, float]:
    """Fits the circle of a scattering matrix element array.

    The circle fit exploits the algebraic fit described in
    "Efficient and robust analysis of complex scattering data under noise in microwave resonators"
    (https://doi.org/10.1063/1.4907935) by S. Probst et al and
    "The physics of superconducting microwave resonators"
    (https://doi.org/10.7907/RAT0-VM75) by J. Gao.

    The function, from the scattering matrix element array, evaluates the center coordinates
    `x_c` and `y_c` and the radius of the circle `r_0`.
    """

    z = z.copy()
    x_norm = 0.5 * (np.max(z.real) + np.min(z.real))
    y_norm = 0.5 * (np.max(z.imag) + np.min(z.imag))
    z -= x_norm + 1j * y_norm

    amplitude_norm = np.max(np.abs(z))
    z /= amplitude_norm

    coords = np.stack(
        [np.abs(z) ** 2, z.real, z.imag, np.ones_like(z, dtype=np.float64)]
    )
    m_matrix = np.einsum("in,jn->ij", coords, coords)

    b_matrix = np.array([[0, 0, 0, -2], [0, 1, 0, 0], [0, 0, 1, 0], [-2, 0, 0, 0]])

    coefficients = np.linalg.eigvals(np.linalg.inv(b_matrix).dot(m_matrix))

    eta = np.min(
        np.real(
            [
                coefficient
                for coefficient in coefficients
                if np.isreal(coefficient) and coefficient > 0
            ]
        )
    )

    def f_matrix(a_vector, m_matrix, b_matrix, eta):
        return a_vector.T @ m_matrix @ a_vector - eta * (
            a_vector.T @ b_matrix @ a_vector - 1
        )

    def constraint(a_vector, b_matrix):
        return a_vector.T @ b_matrix @ a_vector - 1

    constraints = [{"type": "eq", "fun": constraint, "args": (b_matrix,)}]

    a_vector = np.ones(4)
    result = minimize(
        f_matrix, a_vector, args=(m_matrix, b_matrix, eta), constraints=constraints
    )
    a_vector = result.x

    x_c = -a_vector[1] / (2 * a_vector[0])
    y_c = -a_vector[2] / (2 * a_vector[0])
    r_0 = 1 / (
        2
        * np.abs(a_vector[0])
        * np.sqrt(
            a_vector[1] * a_vector[1]
            + a_vector[2] * a_vector[2]
            - 4 * a_vector[0] * a_vector[3]
        )
    )

    return (
        complex(x_c * amplitude_norm + x_norm, y_c * amplitude_norm + y_norm),
        r_0 * amplitude_norm,
    )


def phase_fit(frequencies: NDArray, phases: NDArray) -> NDArray:
    r"""Fits the phase response of a resonator.

    The phase fit firstly ensure the phase data (in rad) covers a significant portion of the full 2 :math:`\pi`
    circle evaluating a `roll_off`. If the data do not cover a full circle it is possible to increase
    the frequency span around the resonance. Data are smoothed using a Gaussian filter and the
    derivative is evaluated while initial guesses for the parameters (`resonance_guess` (in Hz)),
    `q_loaded_guess`, `tau_guess` (in s) and `theta_guess` (in rad) are computed with `frequencies` (in Hz).

    The parameter estimation is done through an iterative least squares process to optimize the model
    parameters. The defined functions: `residuals_q_loaded`, `residuals_resonance_theta`
    `residuals_resonance_theta`, `residuals_tau`, `residuals_resonance_q_loaded`, `residuals_full`
    take the parameters to be fitted and return the residuals calculated by subtracting the phase
    centered model from the phase data (in rad).
    """

    if np.max(phases) - np.min(phases) <= PHASES_THRESHOLD_PERCENTAGE / 100 * 2 * np.pi:
        roll_off = np.max(phases) - np.min(phases)
    else:
        roll_off = 2 * np.pi

    phases_smooth = gaussian_filter1d(phases, STD_DEV_GAUSSIAN_KERNEL)
    phases_derivative = np.gradient(phases_smooth)
    resonance_guess = frequencies[np.argmax(np.abs(phases_derivative))]
    q_loaded_guess = 2 * resonance_guess / (frequencies[-1] - frequencies[0])
    slope = phases[-1] - phases[0] + roll_off
    tau_guess = -slope / (2 * np.pi * (frequencies[-1] - frequencies[0]))
    theta_guess = 0.5 * (
        np.mean(phases[:PHASE_ELEMENTS]) + np.mean(phases[-PHASE_ELEMENTS:])
    )

    def residuals_q_loaded(params):
        (q_loaded,) = params
        return residuals_full((resonance_guess, q_loaded, theta_guess, tau_guess))

    def residuals_resonance_theta(params):
        resonance, theta = params
        return residuals_full((resonance, q_loaded_guess, theta, tau_guess))

    def residuals_tau(params):
        (tau,) = params
        return residuals_full((resonance_guess, q_loaded_guess, theta_guess, tau))

    def residuals_resonance_q_loaded(params):
        resonance, q_loaded = params
        return residuals_full((resonance, q_loaded, theta_guess, tau_guess))

    def residuals_full(params):
        return phase_dist(phases - phase_centered(frequencies, *params))

    p_final = leastsq(residuals_q_loaded, [q_loaded_guess])
    (q_loaded_guess,) = p_final[0]
    p_final = leastsq(residuals_resonance_theta, [resonance_guess, theta_guess])
    resonance_guess, theta_guess = p_final[0]
    p_final = leastsq(residuals_tau, [tau_guess])
    (tau_guess,) = p_final[0]
    p_final = leastsq(residuals_resonance_q_loaded, [resonance_guess, q_loaded_guess])
    resonance_guess, q_loaded_guess = p_final[0]
    p_final = leastsq(
        residuals_full, [resonance_guess, q_loaded_guess, theta_guess, tau_guess]
    )

    return p_final[0][:-1]


def phase_dist(phases: NDArray) -> NDArray:
    """Maps `phases` (in rad) [-2pi, 2pi] to phase distance on circle [0, pi]."""
    return np.pi - np.abs(np.pi - np.abs(phases))


def phase_centered(
    frequencies: NDArray,
    resonance: float,
    q_loaded: float,
    theta: float,
    tau: float = 0.0,
) -> NDArray:
    """Evaluates the phase (in rad) response of a resonator.

    The phase centered evaluates the phase angle (in rad) of a circle centered around the origin accounting
    for a phase offset :math:`\theta` (in rad), a linear background slope
    :math: 2\\pi `\tau` (in s) (`frequencies` (in Hz) - `resonance` (in Hz)) (if needed) and a dependency on
    the `q_loaded`.
    """
    return (
        theta
        - 2 * np.pi * tau * (frequencies - resonance)
        + 2.0 * np.arctan(2.0 * q_loaded * (1.0 - frequencies / resonance))
    )


def periodic_boundary(angle: float) -> float:
    """Maps arbitrary `angle` (in rad) to interval [-np.pi, np.pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
