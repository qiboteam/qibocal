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


def flux_crosstalk_plot(data, qubit, fit):
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
        if flux_qubit[1] != qubit:
            fig.add_trace(
                go.Scatter(
                    x=transmon_frequency(
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
            fig.add_trace(
                go.Scatter(
                    x=transmon_frequency_diagonal(
                        x=qubit_data.bias,
                        w_max=data.drive_frequency[qubit],
                        d=data.d[qubit],
                        matrix_element=data.matrix_element[qubit],
                        sweetspot=data.sweetspot[qubit],
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


# TODO: restore second order approximation
# def freq_q_transmon(x, p0, p1, p2, p3):
#     """
#     Qubit frequency in the boson description. Close to the half-flux quantum (:math:'\\phi=0.5`), :math:`E_J/E_C = E_J(\\phi=0)*d/E_C` can be too small for a quasi-symmetric split-transmon to apply this expression. We assume that the qubit frequencty :math:`\\gg E_C`.

#     Args:
#         p[0] (float): bias offset.
#         p[1] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
#         p[2] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
#         p[3] (float): qubit frequency at the sweetspot.

#     Returns:
#         (float)
#     """
#     return p3 * G_f_d(x, p0, p1, p2)


# def freq_r_transmon(x, p0, p1, p2, p3, p4, p5):
#     """
#     Flux dependent resonator frequency in the transmon limit.

#     Args:
#         p[0] (float): bias offset.
#         p[1] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
#         p[2] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
#         p[3] (float): qubit frequency at the sweetspot / high power resonator frequency,
#         p[4] (float): readout coupling at the sweetspot. Typically denoted as :math:`g`.
#         p[5] (float): high power resonator frequency.

#     Returns:
#         (float)
#     """
#     return p5 + p4**2 * G_f_d(x, p0, p1, p2) / (p5 - p3 * p5 * G_f_d(x, p0, p1, p2))


# def kordering(m, ng=0.4999):
#     """
#     Auxilliary function to compute the qubit frequency in the CPB model (useful when the boson description fails). It sorts the eigenvalues :math:`|m,ng\\rangle` for the Schrodinger equation for the
#     Cooper pair box circuit in the phase basis.

#     Args:
#         m (integer): index denoting the m eigenvector.
#         ng (float): effective offset charge. The sorting does not work for ng integer or half-integer. To study the sweet spot at :math:`ng = 0.5` for instance, one should insert an approximation like :math:`ng = 0.4999`.

#     Returns:
#         (float)
#     """

#     a1 = (round(2 * ng + 1 / 2) % 2) * (round(ng) + 1 * (-1) ** m * divmod(m + 1, 2)[0])
#     a2 = (round(2 * ng - 1 / 2) % 2) * (round(ng) - 1 * (-1) ** m * divmod(m + 1, 2)[0])
#     return a1 + a2


# def mathieu(index, x):
#     """
#     Mathieu's characteristic value. Auxilliary function to compute the qubit frequency in the CPB model.

#     Args:
#         index (integer): index to specify the Mathieu's characteristic value.

#     Returns:def freq_q_transmon(x, p0, p1, p2, p3):
#     """
#     Qubit frequency in the boson description. Close to the half-flux quantum (:math:'\\phi=0.5`), :math:`E_J/E_C = E_J(\\phi=0)*d/E_C` can be too small for a quasi-symmetric split-transmon to apply this expression. We assume that the qubit frequencty :math:`\\gg E_C`.

#     Args:
#         p[0] (float): bias offset.
#         p[1] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
#         p[2] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
#         p[3] (float): qubit frequency at the sweetspot.

#     Returns:
#         (float)
#     """
#     return p3 * G_f_d(x, p0, p1, p2)


# def freq_r_transmon(x, p0, p1, p2, p3, p4, p5):
#     """
#     Flux dependent resonator frequency in the transmon limit.

#     Args:
#         p[0] (float): bias offset.
#         p[1] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
#         p[2] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
#         p[3] (float): qubit frequency at the sweetspot / high power resonator frequency,
#         p[4] (float): readout coupling at the sweetspot. Typically denoted as :math:`g`.
#         p[5] (float): high power resonator frequency.

#     Returns:
#         (float)
#     """
#     return p5 + p4**2 * G_f_d(x, p0, p1, p2) / (p5 - p3 * p5 * G_f_d(x, p0, p1, p2))


# def kordering(m, ng=0.4999):
#     """
#     Auxilliary function to compute the qubit frequency in the CPB model (useful when the boson description fails). It sorts the eigenvalues :math:`|m,ng\\rangle` for the Schrodinger equation for the
#     Cooper pair box circuit in the phase basis.

#     Args:
#         m (integer): index denoting the m eigenvector.
#         ng (float): effective offset charge. The sorting does not work for ng integer or half-integer. To study the sweet spot at :math:`ng = 0.5` for instance, one should insert an approximation like :math:`ng = 0.4999`.

#     Returns:
#         (float)
#     """

#     a1 = (round(2 * ng + 1 / 2) % 2) * (round(ng) + 1 * (-1) ** m * divmod(m + 1, 2)[0])
#     a2 = (round(2 * ng - 1 / 2) % 2) * (round(ng) - 1 * (-1) ** m * divmod(m + 1, 2)[0])
#     return a1 + a2


# def mathieu(index, x):
#     """
#     Mathieu's characteristic value. Auxilliary function to compute the qubit frequency in the CPB model.

#     Args:
#         index (integer): index to specify the Mathieu's characteristic value.

#     Returns:
#         (float)
#     """
#     if index < 0:
#         return mathieu_b(-index, x)
#     else:
#         return mathieu_a(index, x)


# def freq_q_mathieu(x, p0, p1, p2, p3, p4, p5=0.499):
#     """
#     Qubit frequency in the CPB model. It is useful when the boson description fails and to determine :math:`E_C` and :math:`E_J`.

#     Args:
#         p[0] (float): bias offset.
#         p[1] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
#         p[2] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
#         p[3] (float): charge energy at the sweetspot, :math:`E_C`.
#         p[4] (float): Josephson energy, :math:`E_J`.
#         p[5] (float): effective offset charge, :math:`ng`.

#     Returns:
#         (float)
#     """
#     index1 = int(2 * (p5 + kordering(1, p5)))
#     index0 = int(2 * (p5 + kordering(0, p5)))
#     p4 = p4 * G_f_d(x, p0, p1, p2)
#     return p3 * (mathieu(index1, -p4 / (2 * p3)) - mathieu(index0, -p4 / (2 * p3)))


# def freq_r_mathieu(x, p0, p1, p2, p3, p4, p5, p6, p7=0.499):
#     """
#     Resonator frequency in the CPB model.

#     Args:
#         p[0] (float): high power resonator frequency.
#         p[1] (float): readout coupling at the sweetspot.
#         p[2] (float): bias offset.
#         p[3] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
#         p[4] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
#         p[5] (float): charge energy at the sweetspot, :math:`E_C`.
#         p[6] (float): Josephson energy, :math:`E_J`.
#         p[7] (float): effective offset charge, :math:`ng`.

#     Returns:
#         (float)
#     """
#     G = G_f_d(x, p2, p3, p4)
#     f_q = freq_q_mathieu(x, p2, p3, p4, p5, p6, p7)
#     f_r = p0 + p1**2 * G / (p0 - f_q)
#     return f_r

#         (float)
#     """
#     if index < 0:
#         return mathieu_b(-index, x)
#     else:
#         return mathieu_a(index, x)


# def freq_q_mathieu(x, p0, p1, p2, p3, p4, p5=0.499):
#     """
#     Qubit frequency in the CPB model. It is useful when the boson description fails and to determine :math:`E_C` and :math:`E_J`.

#     Args:
#         p[0] (float): bias offset.
#         p[1] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
#         p[2] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
#         p[3] (float): charge energy at the sweetspot, :math:`E_C`.
#         p[4] (float): Josephson energy, :math:`E_J`.
#         p[5] (float): effective offset charge, :math:`ng`.

#     Returns:
#         (float)
#     """
#     index1 = int(2 * (p5 + kordering(1, p5)))
#     index0 = int(2 * (p5 + kordering(0, p5)))
#     p4 = p4 * G_f_d(x, p0, p1, p2)
#     return p3 * (mathieu(index1, -p4 / (2 * p3)) - mathieu(index0, -p4 / (2 * p3)))


# def freq_r_mathieu(x, p0, p1, p2, p3, p4, p5, p6, p7=0.499):
#     """
#     Resonator frequency in the CPB model.

#     Args:
#         p[0] (float): high power resonator frequency.
#         p[1] (float): readout coupling at the sweetspot.
#         p[2] (float): bias offset.
#         p[3] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
#         p[4] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
#         p[5] (float): charge energy at the sweetspot, :math:`E_C`.
#         p[6] (float): Josephson energy, :math:`E_J`.
#         p[7] (float): effective offset charge, :math:`ng`.

#     Returns:
#         (float)
#     """
#     G = G_f_d(x, p2, p3, p4)
#     f_q = freq_q_mathieu(x, p2, p3, p4, p5, p6, p7)
#     f_r = p0 + p1**2 * G / (p0 - f_q)
#     return f_r


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
