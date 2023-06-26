import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import mathieu_a, mathieu_b
from sklearn.linear_model import Ridge


def flux_dependence_plot(data, fit, qubit):
    figures = []
    fitting_report = "No fitting data"

    qubit_data = data.df[data.df["qubit"] == qubit]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR [V]",
            "Phase [rad]",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=qubit_data["frequency"].pint.to("GHz").pint.magnitude,
            y=qubit_data["bias"].pint.to("V").pint.magnitude,
            z=qubit_data["MSR"].pint.to("uV").pint.magnitude,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text=f"{qubit}: Frequency (Hz)",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Bias (V)", row=1, col=1)

    fig.add_trace(
        go.Heatmap(
            x=qubit_data["frequency"].pint.to("GHz").pint.magnitude,
            y=qubit_data["bias"].pint.to("V").pint.magnitude,
            z=qubit_data["phase"].pint.to("rad").pint.magnitude,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text=f"{qubit}: Frequency (Hz)",
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Bias (V)", row=1, col=2)

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


def G_f_d(x, p0, p1, p2):
    # Current offset:          : p[0]
    # 1/I_0, Phi0=Xi*I_0       : p[1]
    # Junction asymmetry d     : p[2]
    G = np.sqrt(
        np.cos(np.pi * (x - p0) * p1) ** 2
        + p2**2 * np.sin(np.pi * (x - p0) * p1) ** 2
    )
    return np.sqrt(G)


def freq_q_transmon(x, p0, p1, p2, p3):
    # Current offset:                                      : p[0]
    # 1/I_0, Phi0=Xi*I_0                                   : p[1]
    # Junction asymmetry d                                 : p[2]
    # f_q0 Qubit frequency at zero flux                    : p[3]
    return p3 * G_f_d(x, p0, p1, p2)


def freq_r_transmon(x, p0, p1, p2, p3, p4, p5):
    # Current offset:                                      : p[0]
    # 1/I_0, Phi0=Xi*I_0                                   : p[1]
    # Junction asymmetry d                                 : p[2]
    # f_q0/f_rh, f_q0 = Qubit frequency at zero flux       : p[3]
    # Qubit-resonator coupling at zero magnetic flux, g_0  : p[4]
    # High power resonator frequency, f_rh                 : p[5]
    return p5 + p4**2 * G_f_d(x, p0, p1, p2) / (p5 - p3 * p5 * G_f_d(x, p0, p1, p2))


def kordering(m, ng=0.4999):
    # Ordering function sorting the eigenvalues |m,ng> for the Schrodinger equation for the
    # Cooper pair box circuit in the phase basis.
    a1 = (round(2 * ng + 1 / 2) % 2) * (round(ng) + 1 * (-1) ** m * divmod(m + 1, 2)[0])
    a2 = (round(2 * ng - 1 / 2) % 2) * (round(ng) - 1 * (-1) ** m * divmod(m + 1, 2)[0])
    return a1 + a2


def mathieu(index, x):
    # Mathieu's characteristic value a_index(x).
    if index < 0:
        dummy = mathieu_b(-index, x)
    else:
        dummy = mathieu_a(index, x)
    return dummy


def freq_q_mathieu(x, p0, p1, p2, p3, p4, p5=0.499):
    # Current offset:                                      : p[0]
    # 1/I_0, Phi0=Xi*I_0                                   : p[1]
    # Junction asymmetry d                                 : p[2]
    # Charging energy E_C                                  : p[3]
    # Josephson energy E_J                                 : p[4]
    # Effective offset charge ng                           : p[5]
    index1 = int(2 * (p5 + kordering(1, p5)))
    index0 = int(2 * (p5 + kordering(0, p5)))
    p4 = p4 * G_f_d(x, p0, p1, p2)
    return p3 * (mathieu(index1, -p4 / (2 * p3)) - mathieu(index0, -p4 / (2 * p3)))


def freq_r_mathieu(x, p0, p1, p2, p3, p4, p5, p6, p7=0.499):
    # High power resonator frequency, f_rh                 : p[0]
    # Qubit-resonator coupling at zero magnetic flux, g_0  : p[1]
    # Current offset:                                      : p[2]
    # 1/I_0, Phi0=Xi*I_0                                   : p[3]
    # Junction asymmetry d                                 : p[4]
    # Charging energy E_C                                  : p[5]
    # Josephson energy E_J                                 : p[6]
    # Effective offset charge ng                           : p[7]
    G = G_f_d(x, p2, p3, p4)
    f_q = freq_q_mathieu(x, p2, p3, p4, p5, p6, p7)
    f_r = p0 + p1**2 * G / (p0 - f_q)
    return f_r


def line(x, p0, p1):
    # Slope                   : p[0]
    # Intercept               : p[1]
    return p0 * x + p1


def feature(x, order=3):
    """Generate polynomial feature of the form
    [1, x, x^2, ..., x^order] where x is the column of x-coordinates
    and 1 is the column of ones for the intercept.
    """
    x = x.reshape(-1, 1)
    return np.power(x, np.arange(order + 1).reshape(1, -1))


def image_to_curve(x, y, z, alpha=0.0001, order=50):
    max_x = np.max(x)
    min_x = np.min(x)
    lenx = int((max_x - min_x) / (x[1] - x[0])) + 1
    max_y = np.max(y)
    min_y = np.min(y)
    leny = int(len(y) / (lenx))
    x = np.linspace(min_x, max_x, lenx)
    y = np.linspace(min_y, max_y, leny)
    z = np.array(z, float)
    z = np.reshape(z, (lenx, leny))
    zmax, zmin = z.max(), z.min()
    znorm = (z - zmin) / (zmax - zmin)

    # Mask out region
    mask = znorm < 0.5
    z = np.argwhere(mask)
    weights = znorm[mask] / float(znorm.max())
    # Column indices
    x_fit = y[z[:, 1].reshape(-1, 1)]
    # Row indices to predict.
    y_fit = x[z[:, 0]]

    # Ridge regression, i.e., least squares with l2 regularization
    A = feature(x_fit, order)
    model = Ridge(alpha=alpha)
    model.fit(A, y_fit, sample_weight=weights)
    x_pred = y
    X_pred = feature(x_pred, order)
    y_pred = model.predict(X_pred)
    return y_pred, x_pred
