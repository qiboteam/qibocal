import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platform import Platform
from qibolab.qubits import QubitId

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
    if (
        fit is not None
        and not data.__class__.__name__ == "CouplerSpectroscopyData"
        and qubit in fit.fitted_parameters
    ):
        params = fit.fitted_parameters[qubit]
        bias = np.unique(qubit_data.bias)
        fig.add_trace(
            go.Scatter(
                x=fit_function(bias, **params),
                y=bias,
                showlegend=True,
                name="Fit",
                marker=dict(color="green"),
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
                x=frequencies,
                y=qubit_data.bias,
                z=qubit_data.signal,
                showscale=False,
            ),
            row=1,
            col=col + 1,
        )
        if fit is not None:

            if flux_qubit[1] != qubit and flux_qubit in fit.fitted_parameters:
                fig.add_trace(
                    go.Scatter(
                        x=fit_function(
                            xj=qubit_data.bias, **fit.fitted_parameters[flux_qubit]
                        ),
                        y=qubit_data.bias,
                        showlegend=not any(
                            isinstance(trace, go.Scatter) for trace in fig.data
                        ),
                        legendgroup="Fit",
                        name="Fit",
                        marker=dict(color="green"),
                    ),
                    row=1,
                    col=col + 1,
                )
            elif flux_qubit in fit.fitted_parameters:
                diagonal_params = fit.fitted_parameters[qubit, qubit]
                fig.add_trace(
                    go.Scatter(
                        x=fit_function(
                            qubit_data.bias,
                            **diagonal_params,
                        ),
                        y=qubit_data.bias,
                        showlegend=not any(
                            isinstance(trace, go.Scatter) for trace in fig.data
                        ),
                        legendgroup="Fit",
                        name="Fit",
                        marker=dict(color="green"),
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
    fig.update_layout(xaxis2=dict(range=[np.min(frequencies), np.max(frequencies)]))
    fig.update_layout(xaxis3=dict(range=[np.min(frequencies), np.max(frequencies)]))
    fig.update_layout(
        showlegend=True,
    )
    figures.append(fig)

    return figures, fitting_report


def G_f_d(xi, xj, offset, d, crosstalk_element, normalization):
    """Auxiliary function to calculate qubit frequency as a function of bias.

    It also determines the flux dependence of :math:`E_J`,:math:`E_J(\\phi)=E_J(0)G_f_d`.
    For more details see: https://arxiv.org/pdf/cond-mat/0703002.pdf

    Args:
        xi (float): bias of target qubit
        xj (float): bias of neighbor qubit
        offset (float): phase_offset [V].
        matrix_element(float): diagonal crosstalk matrix element
        crosstalk_element(float): off-diagonal crosstalk matrix element
        d (float): asymmetry between the two junctions of the transmon.
                   Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
        normalization (float): Normalize diagonal element to 1
    Returns:
        (float)
    """
    return (
        d**2
        + (1 - d**2)
        * np.cos(
            np.pi
            * (xi * normalization + normalization * xj * crosstalk_element + offset)
        )
        ** 2
    ) ** 0.25


def transmon_frequency(
    xi, xj, w_max, d, normalization, offset, crosstalk_element, charging_energy
):
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
        charging_energy (float): Ec / h (GHz)

     Returns:
         (float): qubit frequency as a function of bias.
    """
    return (w_max + charging_energy) * G_f_d(
        xi,
        xj,
        offset=offset,
        d=d,
        normalization=normalization,
        crosstalk_element=crosstalk_element,
    ) - charging_energy


def transmon_readout_frequency(
    xi,
    xj,
    w_max,
    d,
    normalization,
    crosstalk_element,
    offset,
    resonator_freq,
    g,
    charging_energy,
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
        xi,
        xj,
        offset=offset,
        d=d,
        normalization=normalization,
        crosstalk_element=crosstalk_element,
    ) / (
        resonator_freq
        - transmon_frequency(
            xi=xi,
            xj=xj,
            w_max=w_max,
            d=d,
            normalization=normalization,
            offset=offset,
            crosstalk_element=crosstalk_element,
            charging_energy=charging_energy,
        )
    )


def qubit_flux_dependence_fit_bounds(qubit_frequency: float):
    """Returns bounds for qubit flux fit."""
    return (
        [
            qubit_frequency * HZ_TO_GHZ - 1,
            0,
            -1,
        ],
        [
            qubit_frequency * HZ_TO_GHZ + 1,
            np.inf,
            1,
        ],
    )


def crosstalk_matrix(platform: Platform, qubits: list[QubitId]) -> np.ndarray:
    """Computing crosstalk matrix for number of qubits selected.
    The matrix returns has the following matrix element:
    (M)ij = qubits[i].crosstalk_matrix[qubits[j]]
    """
    size = len(qubits)
    matrix = np.ones((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i, j] = platform.qubits[qubits[i]].crosstalk_matrix[qubits[j]]

    return matrix


def compensation_matrix(platform: Platform, qubits: list[QubitId]) -> np.ndarray:
    """Compensation matrix C computed as M C = diag(M') where M is the
    crosstalk matrix.
    For more details check: https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf
    8.2.3
    """
    size = len(qubits)
    matrix = np.ones((size, size))
    crosstalk = crosstalk_matrix(platform, qubits)
    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i, j] = 1
            else:
                matrix[i, j] = -crosstalk[i, j] / crosstalk[i, i]

    return matrix


def invert_transmon_freq(target_freq: float, platform: Platform, qubit: QubitId):
    """Return right side of equation matrix * total_flux = f(target_freq).
    Target frequency shoudl be expressed in GHz.
    """
    charging_energy = -platform.qubits[qubit].anharmonicity * HZ_TO_GHZ
    offset = (
        -platform.qubits[qubit].sweetspot
        * platform.qubits[qubit].crosstalk_matrix[qubit]
    )
    w_max = platform.qubits[qubit].drive_frequency * HZ_TO_GHZ
    d = platform.qubits[qubit].asymmetry
    angle = np.sqrt(
        1
        / (1 - d**2)
        * (((target_freq + charging_energy) / (w_max + charging_energy)) ** 4 - d**2)
    )
    return 1 / np.pi * np.arccos(angle) - offset


def frequency_to_bias(
    target_freqs: dict[QubitId, float], platform: Platform
) -> np.ndarray:
    """Starting from set of target_freqs computes bias points using the compensation matrix."""
    qubits = list(target_freqs)
    inverted_crosstalk_matrix = np.linalg.inv(
        crosstalk_matrix(platform, qubits) @ compensation_matrix(platform, qubits)
    )
    transmon_freq = np.array(
        [
            invert_transmon_freq(freq, platform, qubit)
            for qubit, freq in target_freqs.items()
        ]
    )
    bias_array = inverted_crosstalk_matrix @ transmon_freq
    return {qubit: bias_array[index] for index, qubit in enumerate(qubits)}
