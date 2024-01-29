import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import HZ_TO_GHZ


def is_crosstalk(data):
    """Check if keys are tuple which corresponds to crosstalk data structure."""
    return all(isinstance(key, tuple) for key in data.data.keys())


def create_data_array(freq, bias, signal, phase, dtype):
    """Create custom dtype array for acquired data."""
    size = len(freq) * len(bias)
    ar = np.empty(size, dtype=dtype)
    frequency, biases = np.meshgrid(freq, bias)
    ar["freq"] = frequency.ravel()
    ar["bias"] = biases.ravel()
    ar["signal"] = signal.ravel()
    ar["phase"] = phase.ravel()
    return np.rec.array(ar)


def flux_dependence_plot(data, fit, qubit, fit_function=None):
    figures = []
    qubit_data = data[qubit]
    frequencies = qubit_data.freq * HZ_TO_GHZ

    subplot_titles = (
        "Signal [a.u.]",
        "Phase [rad]",
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=subplot_titles,
    )

    fig.add_trace(
        go.Heatmap(
            x=qubit_data.freq * HZ_TO_GHZ,
            y=qubit_data.bias,
            z=qubit_data.signal,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )

    # TODO: This fit is for frequency, can it be reused here, do we even want the fit ?
    if fit is not None and not data.__class__.__name__ == "CouplerSpectroscopyData":
        params = fit.fitted_parameters[qubit]
        bias = np.unique(qubit_data.bias)
        fig.add_trace(
            go.Scatter(
                x=fit_function(bias, *params),
                y=bias,
                showlegend=True,
                name="Fit",
                marker=dict(color="black"),
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(
        title_text=f"Frequency [GHz]",
        row=1,
        col=1,
    )

    fig.update_yaxes(title_text="Bias [V]", row=1, col=1)

    fig.add_trace(
        go.Heatmap(
            x=qubit_data.freq * HZ_TO_GHZ,
            y=qubit_data.bias,
            z=qubit_data.phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text=f"Frequency [GHz]",
        row=1,
        col=2,
    )

    fig.update_layout(xaxis1=dict(range=[np.min(frequencies), np.max(frequencies)]))

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h"),
    )

    figures.append(fig)

    return figures


def flux_crosstalk_plot(data, qubit, fit, fit_function):
    figures = []
    fitting_report = ""
    all_qubit_data = {
        index: data_qubit
        for index, data_qubit in data.data.items()
        if index[0] == qubit
    }
    fig = make_subplots(
        rows=1,
        cols=len(all_qubit_data),
        horizontal_spacing=0.3 / len(all_qubit_data),
        vertical_spacing=0.1,
        subplot_titles=len(all_qubit_data) * ("Signal [a.u.]",),
    )
    for col, (flux_qubit, qubit_data) in enumerate(all_qubit_data.items()):
        frequencies = qubit_data.freq * HZ_TO_GHZ
        fig.add_trace(
            go.Heatmap(
                x=frequencies, y=qubit_data.bias, z=qubit_data.signal, showscale=False
            ),
            row=1,
            col=col + 1,
        )
        if fit is not None:
            if flux_qubit[1] != qubit:
                fig.add_trace(
                    go.Scatter(
                        x=fit_function(
                            xj=qubit_data.bias, **fit.fitted_parameters[flux_qubit]
                        )
                        * HZ_TO_GHZ,
                        y=qubit_data.bias,
                        showlegend=not any(
                            isinstance(trace, go.Scatter) for trace in fig.data
                        ),
                        legendgroup="Fit",
                        name="Fit",
                        marker=dict(color="black"),
                    ),
                    row=1,
                    col=col + 1,
                )
            else:
                diagonal_params = dict(
                    w_max=data.qubit_frequency[flux_qubit[0]],
                    d=data.d[flux_qubit[0]],
                    matrix_element=data.matrix_element[flux_qubit[0]],
                    sweetspot=data.sweetspot[flux_qubit[0]],
                )
                if fit_function != transmon_frequency:
                    diagonal_params.update(
                        g=data.g[flux_qubit[0]],
                        resonator_freq=data.bare_resonator_frequency[flux_qubit[0]],
                    )
                fig.add_trace(
                    go.Scatter(
                        x=globals().get(fit_function.__name__ + "_diagonal")(
                            x=qubit_data.bias,
                            **diagonal_params,
                        )
                        * HZ_TO_GHZ,
                        y=qubit_data.bias,
                        showlegend=not any(
                            isinstance(trace, go.Scatter) for trace in fig.data
                        ),
                        legendgroup="Fit",
                        name="Fit",
                        marker=dict(color="black"),
                    ),
                    row=1,
                    col=col + 1,
                )

        fig.update_xaxes(
            title_text="Frequency [GHz]",
            row=1,
            col=col + 1,
        )

        fig.update_yaxes(
            title_text=f"Qubit {flux_qubit[1]}: Bias [V]", row=1, col=col + 1
        )

    fig.update_layout(xaxis1=dict(range=[np.min(frequencies), np.max(frequencies)]))
    fig.update_layout(
        showlegend=True,
    )
    figures.append(fig)

    return figures, fitting_report


def G_f_d(xi, xj, sweetspot, d, matrix_element, crosstalk_element):
    """Auxiliary function to calculate qubit frequency as a function of bias.

    It also determines the flux dependence of :math:`E_J`,:math:`E_J(\\phi)=E_J(0)G_f_d`.
    For more details see: https://arxiv.org/pdf/cond-mat/0703002.pdf

    Args:
        xi (float): bias of target qubit
        xj (float): bias of neighbor qubit
        sweetspot (float): sweetspot [V].
        matrix_element(float): diagonal crosstalk matrix element
        crosstalk_element(float): off-diagonal crosstalk matrix element
        d (float): asymmetry between the two junctions of the transmon.
                   Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.

    Returns:
        (float)
    """
    return (
        d**2
        + (1 - d**2)
        * np.cos(np.pi * ((xi - sweetspot) * matrix_element - xj * crosstalk_element))
        ** 2
    ) ** 0.25


def transmon_frequency(xi, xj, w_max, d, matrix_element, sweetspot, crosstalk_element):
    r"""Approximation to transmon frequency.

    The formula holds in the transmon regime Ej / Ec >> 1.

    See  https://arxiv.org/pdf/cond-mat/0703002.pdf for the complete formula.

    Args:
        xi (float): bias of target qubit
        xj (float): bias of neighbor qubit
        w_max (float): maximum frequency  :math:`w_{max} = \sqrt{8 E_j E_c}
        sweetspot (float): sweetspot [V].
        matrix_element(float): diagonal crosstalk matrix element
        crosstalk_element(float): off-diagonal crosstalk matrix element
        d (float): asymmetry between the two junctions of the transmon.
                   Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.

     Returns:
         (float): qubit frequency as a function of bias.
    """
    return w_max * G_f_d(
        xi,
        xj,
        sweetspot=sweetspot,
        d=d,
        matrix_element=matrix_element,
        crosstalk_element=crosstalk_element,
    )


def transmon_frequency_diagonal(x, w_max, d, matrix_element, sweetspot):
    r"""Approximation to transmon frequency without crosstalk.

    The formula holds in the transmon regime Ej / Ec >> 1.

    See  https://arxiv.org/pdf/cond-mat/0703002.pdf for the complete formula.

    Args:
        x (float): bias of target qubit
        w_max (float): maximum frequency  :math:`w_{max} = \sqrt{8 E_j E_c}
        sweetspot (float): sweetspot [V].
        matrix_element(float): diagonal crosstalk matrix element
        d (float): asymmetry between the two junctions of the transmon.
                   Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.

     Returns:
         (float): qubit frequency as a function of bias.
    """
    return transmon_frequency(
        xi=x,
        xj=0,
        w_max=w_max,
        d=d,
        matrix_element=matrix_element,
        sweetspot=sweetspot,
        crosstalk_element=0,
    )


def transmon_readout_frequency(
    xi, xj, w_max, d, matrix_element, crosstalk_element, sweetspot, resonator_freq, g
):
    r"""Approximation to flux dependent resonator frequency.

    The formula holds in the transmon regime Ej / Ec >> 1.

    See  https://arxiv.org/pdf/cond-mat/0703002.pdf for the complete formula.

    Args:
         xi (float): bias of target qubit
         xj (float): bias of neighbor qubit
         w_max (float): maximum frequency  :math:`w_{max} = \sqrt{8 E_j E_c}
         sweetspot (float): sweetspot [V].
         matrix_element(float): diagonal crosstalk matrix element
         crosstalk_element(float): off-diagonal crosstalk matrix element
         d (float): asymmetry between the two junctions of the transmon.
                    Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
         resonator_freq (float): bare resonator frequency [GHz]
         g (float): readout coupling.

     Returns:
         (float): resonator frequency as a function of bias.
    """
    return resonator_freq + g**2 * G_f_d(
        xi=xi,
        xj=xj,
        sweetspot=sweetspot,
        d=d,
        matrix_element=matrix_element,
        crosstalk_element=crosstalk_element,
    ) / (
        resonator_freq
        - transmon_frequency(
            xi=xi,
            xj=xj,
            w_max=w_max,
            d=d,
            matrix_element=matrix_element,
            sweetspot=sweetspot,
            crosstalk_element=crosstalk_element,
        )
    )


def transmon_readout_frequency_diagonal(
    x, w_max, d, matrix_element, sweetspot, resonator_freq, g
):
    r"""Approximation to flux dependent resonator frequency without crosstalk.

    The formula holds in the transmon regime Ej / Ec >> 1.

    See  https://arxiv.org/pdf/cond-mat/0703002.pdf for the complete formula.

    Args:
         x (float): bias value
         w_max (float): maximum frequency  :math:`w_{max} = \sqrt{8 E_j E_c}
         d (float):  d (float): asymmetry between the two junctions of the transmon.
         matrix element (float): diagonal crosstalk matrix element
         phase (float): bias corresponding to zero flux (sweetspot).
         resonator_freq (float): bare resonator frequency [GHz]
         g (float): readout coupling.

     Returns:
         (float): resonator frequency as a function of bias.
    """
    return transmon_readout_frequency(
        xi=x,
        xj=0,
        w_max=w_max,
        sweetspot=sweetspot,
        d=d,
        matrix_element=matrix_element,
        crosstalk_element=0,
        g=g,
        resonator_freq=resonator_freq,
    )


def extract_min_feature(freq, bias, signal, threshold=1.5):
    """Extract min feature using SNR."""
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    snr_map = (signal - mean_signal) / std_signal
    binary_mask = snr_map < -threshold
    return freq[binary_mask], bias[binary_mask]


def extract_max_feature(freq, bias, signal, threshold=1.5):
    """Extract max feature using SNR."""
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    snr_map = (signal - mean_signal) / std_signal
    binary_mask = snr_map > threshold
    return freq[binary_mask], bias[binary_mask]


def qubit_flux_dependence_fit_bounds(qubit_frequency: float, bias: np.array):
    """Returns bounds for qubit flux fit."""
    return (
        [
            qubit_frequency * HZ_TO_GHZ - 1,
            0,
            0,
            np.mean(bias) - 0.5,
        ],
        [
            qubit_frequency * HZ_TO_GHZ + 1,
            1,
            np.inf,
            np.mean(bias) + 0.5,
        ],
    )


def resonator_flux_dependence_fit_bounds(
    qubit_frequency: float, bias: np.array, bare_resonator_frequency: float
):
    """Returns bounds for resonator flux fit."""
    left_bound, right_bound = qubit_flux_dependence_fit_bounds(
        qubit_frequency=qubit_frequency, bias=bias
    )
    left_bound += [bare_resonator_frequency - 0.5, 0]
    right_bound += [bare_resonator_frequency + 0.5, 1]
    return (left_bound, right_bound)
