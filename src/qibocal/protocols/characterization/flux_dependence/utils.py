import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import mathieu_a, mathieu_b
from sklearn.linear_model import Ridge

from ..utils import HZ_TO_GHZ, table_dict, table_html

FLUX_PARAMETERS = {
    "Xi": "Constant to map flux to bias [V]",
    "d": "Junction asymmetry",
    "Ec": "Charge energy Ec [Hz]",
    "Ej": "Josephson energy Ej [Hz]",
    "f_q_offset": "Qubit frequency offset [Hz]",
    "C_ii": "Flux matrix element C_ii [Hz/V]",
    "g": "Readout coupling",
    "bare_resonator_frequency": "Bare resonator frequency [Hz]",
    "f_qs": "Qubit frequency [Hz]",
    "f_r_offset": "Resonator frequency offset [Hz]",
}


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


def flux_dependence_plot(data, fit, qubit):
    figures = []
    fitting_report = ""

    qubit_data = data[qubit]

    if not data.__class__.__name__ == "CouplerSpectroscopyData":
        subplot_titles = (
            "Signal [a.u.]",
            "Phase [rad]",
        )
    else:
        subplot_titles = (
            "Signal [a.u.] Qubit" + str(qubit),
            "Phase [rad] Qubit" + str(qubit),
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=subplot_titles,
    )
    frequencies = qubit_data.freq * HZ_TO_GHZ
    signal = qubit_data.signal
    if data.__class__.__name__ == "ResonatorFluxData":
        signal_mask = 0.5
        if data.resonator_type == "3D":
            signal = -signal
    elif (
        data.__class__.__name__ == "QubitFluxData"
        or data.__class__.__name__ == "CouplerSpectroscopyData"
    ):
        signal_mask = 0.3
        if data.resonator_type == "2D":
            signal = -signal

    frequencies1, biases1 = image_to_curve(
        frequencies, qubit_data.bias, signal, signal_mask
    )

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=qubit_data.bias,
            z=qubit_data.signal,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )

    if not data.__class__.__name__ == "CouplerSpectroscopyData":
        fig.add_trace(
            go.Scatter(
                x=frequencies1,
                y=biases1,
                mode="markers",
                marker_color="green",
                showlegend=True,
                name="Curve estimation",
            ),
            row=1,
            col=1,
        )

    # TODO: This fit is for frequency, can it be reused here, do we even want the fit ?
    if fit is not None and not data.__class__.__name__ == "CouplerSpectroscopyData":
        fitting_report = ""
        params = fit.fitted_parameters[qubit]
        fitting_report_label = "Frequency"
        if fit.frequency[qubit] != 0:
            if data.__class__.__name__ == "ResonatorFluxData":
                fitting_report_label = "Resonator Frequency [Hz]"
                if all(param in params for param in ["Ec", "Ej"]):
                    popt = [
                        params["bare_resonator_frequency"],
                        params["g"],
                        fit.sweetspot[qubit],
                        params["Xi"],
                        params["d"],
                        params["Ec"],
                        params["Ej"],
                    ]
                    freq_fit = freq_r_mathieu(biases1, *popt)
                else:
                    popt = [
                        fit.sweetspot[qubit],
                        params["Xi"],
                        params["d"],
                        params["f_q/bare_resonator_frequency"],
                        params["g"],
                        params["bare_resonator_frequency"],
                    ]
                    freq_fit = freq_r_transmon(biases1, *popt)
            elif data.__class__.__name__ == "QubitFluxData":
                fitting_report_label = "Qubit Frequency [Hz]"
                if all(param in params for param in ["Ec", "Ej"]):
                    popt = [
                        fit.sweetspot[qubit],
                        params["Xi"],
                        params["d"],
                        params["Ec"],
                        params["Ej"],
                    ]
                    freq_fit = freq_q_mathieu(biases1, *popt)
                else:
                    popt = [
                        fit.sweetspot[qubit],
                        params["Xi"],
                        params["d"],
                        fit.frequency[qubit],
                    ]
                    freq_fit = freq_q_transmon(biases1, *popt)

            fig.add_trace(
                go.Scatter(
                    x=freq_fit,
                    y=biases1,
                    showlegend=True,
                    name="Fit",
                ),
                row=1,
                col=1,
            )

            parameters = []
            values = []

            for key, value in fit.fitted_parameters[qubit].items():
                values.append(np.round(value, 5))
                parameters.append(FLUX_PARAMETERS[key])

            parameters.extend([fitting_report_label, "Sweetspot"])
            values.extend(
                [np.round(fit.frequency[qubit], 5), np.round(fit.sweetspot[qubit], 3)]
            )

            fitting_report = table_html(table_dict(qubit, parameters, values))

    fig.update_xaxes(
        title_text=f"Frequency [GHz]",
        row=1,
        col=1,
    )
    if not data.__class__.__name__ == "CouplerSpectroscopyData":
        fig.update_yaxes(title_text="Bias [V]", row=1, col=1)
    else:
        fig.update_yaxes(title_text="Pulse Amplitude [a.u.]", row=1, col=1)

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
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

    if not data.__class__.__name__ == "CouplerSpectroscopyData":
        fig.update_yaxes(title_text="Bias [V]", row=1, col=2)
    else:
        fig.update_yaxes(title_text="Pulse Amplitude [a.u.]", row=1, col=2)

    fig.update_layout(xaxis1=dict(range=[np.min(frequencies), np.max(frequencies)]))

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h"),
    )

    figures.append(fig)

    return figures, fitting_report


def flux_crosstalk_plot(data, qubit):
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
        signal = qubit_data.signal
        if data.resonator_type == "2D":
            signal = -signal

        fig.add_trace(
            go.Heatmap(
                x=frequencies,
                y=qubit_data.bias,
                z=qubit_data.signal,
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
    fig.update_traces(showscale=False)  # disable colorbar
    fig.update_layout(
        showlegend=False,
    )

    figures.append(fig)

    return figures, fitting_report


def G_f_d(x, p0, p1, p2):
    """
    Auxiliary function to calculate the qubit frequency as a function of bias for the qubit flux spectroscopy. It also determines the flux dependence of :math:`E_J`, :math:`E_J(\\phi)=E_J(0)G_f_d^2`.

    Args:
        p[0] (float): bias offset.
        p[1] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
        p[2] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.

    Returns:
        (float)
    """
    G = np.sqrt(
        np.cos(np.pi * (x - p0) * p1) ** 2
        + p2**2 * np.sin(np.pi * (x - p0) * p1) ** 2
    )
    return np.sqrt(G)


def freq_q_transmon(x, p0, p1, p2, p3):
    """
    Qubit frequency in the boson description. Close to the half-flux quantum (:math:'\\phi=0.5`), :math:`E_J/E_C = E_J(\\phi=0)*d/E_C` can be too small for a quasi-symmetric split-transmon to apply this expression. We assume that the qubit frequencty :math:`\\gg E_C`.

    Args:
        p[0] (float): bias offset.
        p[1] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
        p[2] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
        p[3] (float): qubit frequency at the sweetspot.

    Returns:
        (float)
    """
    return p3 * G_f_d(x, p0, p1, p2)


def freq_r_transmon(x, p0, p1, p2, p3, p4, p5):
    """
    Flux dependent resonator frequency in the transmon limit.

    Args:
        p[0] (float): bias offset.
        p[1] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
        p[2] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
        p[3] (float): qubit frequency at the sweetspot / high power resonator frequency,
        p[4] (float): readout coupling at the sweetspot. Typically denoted as :math:`g`.
        p[5] (float): high power resonator frequency.

    Returns:
        (float)
    """
    return p5 + p4**2 * G_f_d(x, p0, p1, p2) / (p5 - p3 * p5 * G_f_d(x, p0, p1, p2))


def kordering(m, ng=0.4999):
    """
    Auxilliary function to compute the qubit frequency in the CPB model (useful when the boson description fails). It sorts the eigenvalues :math:`|m,ng\\rangle` for the Schrodinger equation for the
    Cooper pair box circuit in the phase basis.

    Args:
        m (integer): index denoting the m eigenvector.
        ng (float): effective offset charge. The sorting does not work for ng integer or half-integer. To study the sweet spot at :math:`ng = 0.5` for instance, one should insert an approximation like :math:`ng = 0.4999`.

    Returns:
        (float)
    """

    a1 = (round(2 * ng + 1 / 2) % 2) * (round(ng) + 1 * (-1) ** m * divmod(m + 1, 2)[0])
    a2 = (round(2 * ng - 1 / 2) % 2) * (round(ng) - 1 * (-1) ** m * divmod(m + 1, 2)[0])
    return a1 + a2


def mathieu(index, x):
    """
    Mathieu's characteristic value. Auxilliary function to compute the qubit frequency in the CPB model.

    Args:
        index (integer): index to specify the Mathieu's characteristic value.

    Returns:
        (float)
    """
    if index < 0:
        return mathieu_b(-index, x)
    else:
        return mathieu_a(index, x)


def freq_q_mathieu(x, p0, p1, p2, p3, p4, p5=0.499):
    """
    Qubit frequency in the CPB model. It is useful when the boson description fails and to determine :math:`E_C` and :math:`E_J`.

    Args:
        p[0] (float): bias offset.
        p[1] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
        p[2] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
        p[3] (float): charge energy at the sweetspot, :math:`E_C`.
        p[4] (float): Josephson energy, :math:`E_J`.
        p[5] (float): effective offset charge, :math:`ng`.

    Returns:
        (float)
    """
    index1 = int(2 * (p5 + kordering(1, p5)))
    index0 = int(2 * (p5 + kordering(0, p5)))
    p4 = p4 * G_f_d(x, p0, p1, p2)
    return p3 * (mathieu(index1, -p4 / (2 * p3)) - mathieu(index0, -p4 / (2 * p3)))


def freq_r_mathieu(x, p0, p1, p2, p3, p4, p5, p6, p7=0.499):
    """
    Resonator frequency in the CPB model.

    Args:
        p[0] (float): high power resonator frequency.
        p[1] (float): readout coupling at the sweetspot.
        p[2] (float): bias offset.
        p[3] (float): constant to convert flux (:math:`\\phi_0`) to bias (:math:`v_0`). Typically denoted as :math:`\\Xi`. :math:`v_0 = \\Xi \\phi_0`.
        p[4] (float): asymmetry between the two junctions of the transmon. Typically denoted as :math:`d`. :math:`d = (E_J^1 - E_J^2) / (E_J^1 + E_J^2)`.
        p[5] (float): charge energy at the sweetspot, :math:`E_C`.
        p[6] (float): Josephson energy, :math:`E_J`.
        p[7] (float): effective offset charge, :math:`ng`.

    Returns:
        (float)
    """
    G = G_f_d(x, p2, p3, p4)
    f_q = freq_q_mathieu(x, p2, p3, p4, p5, p6, p7)
    f_r = p0 + p1**2 * G / (p0 - f_q)
    return f_r


def line(x, p0, p1):
    """
    Linear fit.

    Args:
        p[0] (float): slope.
        p[1] (float): intercept.

    Returns:
        (float)
    """
    return p0 * x + p1


def feature(x, order=3):
    """
    Auxilliary function for the function image_to_curve(). It generates a polynomial feature of the form [1, x, x^2, ..., x^order].

    Args:
        x (ndarray) column vector.

    Returns:
        (ndarray)
    """
    x = x.reshape(-1, 1)
    return np.power(x, np.arange(order + 1).reshape(1, -1))


def image_to_curve(x, y, z, signal_mask=0.5, alpha=1e-5, order=50):
    """
    Extracts a feature characterized by min(z(x, y)). It considers all the data and applies Ridge regression on a polynomial ansatz in x. This allows obtaining a set of points describing the feature as y vs x.

    Args:
        x (ndarray) frequencies
        y (ndarray) bias
        z (ndarray) signal

    Returns:
        y_pred (ndarray) frequencies
        x_pred (ndarray) bias
    """
    max_x = np.max(x)
    min_x = np.min(x)
    lenx = int((max_x - min_x) / (x[1] - x[0])) + 1
    max_y = np.max(y)
    min_y = np.min(y)
    leny = int(len(y) / (lenx))
    z = np.array(z, float)
    if len(z) != leny * lenx:
        lenx += 1
        leny = int(len(y) / (lenx))
    x = np.linspace(min_x, max_x, lenx)
    y = np.linspace(min_y, max_y, leny)
    z = np.reshape(z, (leny, lenx))
    zmax, zmin = z.max(), z.min()
    znorm = (z - zmin) / (zmax - zmin)

    # Mask out region
    mask = znorm < signal_mask
    z = np.argwhere(mask)
    weights = znorm[mask] / float(znorm.max())
    # Column indices
    x_fit = y[z[:, 0].reshape(-1, 1)]
    # Row indices to predict.
    y_fit = x[z[:, 1]]

    # Ridge regression, i.e., least squares with l2 regularization
    A = feature(x_fit, order)
    model = Ridge(alpha=alpha)
    model.fit(A, y_fit, sample_weight=weights)
    x_pred = y
    X_pred = feature(x_pred, order)
    y_pred = model.predict(X_pred)
    return y_pred, x_pred


def get_resonator_freq_flux(
    bias, sweetspot, flux_to_bias, asymmetry, g, brf, ssf_brf, Ec, Ej
):
    """
    Estimate the resonator frequency for a give "bias".

    Args:
        bias (float): Bias value where the resonator frequency should be estimated
        sweetspot (float): qubit sweetspot
        flux_to_bias (float): Resonator value to convert from flux to bias
        asymmetry (float): Resonator asymmetry
        g (float): Readout coupling
        brf (float): Bare resonator frequency at sweetspot
        ssf_brf (float): Estimated sweetspot qubit frequency divided by the bare_resonator_frequency
        Ec (float): Readout Charge Energy
        Ej (float): Readout Josephson Energy

    Returns:
        freq_resonator (float): Estimated Resonator frequency at the provided bias point
    """
    if (
        flux_to_bias == 0.0
        or asymmetry == 0.0
        or g == 0.0
        or brf == 0.0
        or Ec == 0.0
        or Ej == 0.0
    ):
        raise ValueError(
            "Not enough parameters to estimate the resonator frequency for the given bias"
        )

    if ssf_brf == 0:
        # First order approximation used during resonator flux fitting
        #   'sweetspot_0':p0,
        #   'flux_to_bias':p1,
        #   'asymmetry':p2,
        #   'readout_coupling':p4,
        #   'bare_resonator_frequency_0':p5
        #   'sweetspot_qubit_frequency/bare_resonator_frequency':p3,
        freq_resonator = freq_r_transmon(
            bias,
            sweetspot,
            flux_to_bias,
            asymmetry,
            ssf_brf,
            g,
            brf,
        )
    else:
        # Second order approximation used during resonator flux fitting
        #   'sweetspot_0':p2,
        #   'flux_to_bias':p3,
        #   'asymmetry':p4,
        #   'readout_coupling':p1,
        #   'bare_resonator_frequency_0':p0,
        #   'Ec':p5,
        #   'Ej:p6'
        freq_resonator = freq_r_mathieu(
            bias,
            brf,
            g,
            sweetspot,
            flux_to_bias,
            asymmetry,
            Ec,
            Ej,
        )

    return freq_resonator
