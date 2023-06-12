import re
from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import pint
from scipy.special import mathieu_a, mathieu_b
from sklearn.linear_model import Ridge


def lorenzian(frequency, amplitude, center, sigma, offset):
    # http://openafox.com/science/peak-function-derivations.html
    return (amplitude / np.pi) * (
        sigma / ((frequency - center) ** 2 + sigma**2)
    ) + offset


def rabi(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   Period    T                  : 1/p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(2 * np.pi * x * p2 + p3) * np.exp(-x * p4)


def ramsey(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   DeltaFreq                    : p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(x * p2 + p3) * np.exp(-x * p4)


def exp(x, *p):
    return p[0] - p[1] * np.exp(-1 * x * p[2])


def flipping(x, p0, p1, p2, p3):
    # A fit to Flipping Qubit oscillation
    # Epsilon?? shoule be Amplitude : p[0]
    # Offset                        : p[1]
    # Period of oscillation         : p[2]
    # phase for the first point corresponding to pi/2 rotation   : p[3]
    return np.sin(x * 2 * np.pi / p2 + p3) * p0 + p1


def cos(x, p0, p1, p2, p3):
    # Offset                  : p[0]
    # Amplitude               : p[1]
    # Period                  : p[2]
    # Phase                   : p[3]
    return p0 + p1 * np.cos(2 * np.pi * x / p2 + p3)


def line(x, p0, p1):
    # Slope                   : p[0]
    # Intercept               : p[1]
    return p0 * x + p1


def parse(key):
    name = key.split("[")[0]
    unit = re.search(r"\[([A-Za-z0-9_]+)\]", key).group(1)
    return name, unit


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


def feature(x, order=3):
    """Generate polynomial feature of the form
    [1, x, x^2, ..., x^order] where x is the column of x-coordinates
    and 1 is the column of ones for the intercept.
    """
    x = x.reshape(-1, 1)
    return np.power(x, np.arange(order + 1).reshape(1, -1))


def image_to_curve(x, y, z, alpha=0.0001, order=50):
    max_y = np.max(y)
    min_y = np.min(y)
    leny = int((max_y - min_y) / (y[1] - y[0])) + 1
    max_x = np.max(x)
    min_x = np.min(x)
    lenx = int(len(x) / (leny))
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


def pint_to_float(x):
    if isinstance(x, pd.Series):
        return x.apply(pint_to_float)
    elif isinstance(x, pint.Quantity):
        return x.to(x.units).magnitude
    else:
        return x


def cumulative(input_data, points):
    r"""Evaluates in `input_data` the cumulative distribution
    function of `points`.
    WARNING: `input_data` and `points` should be sorted data.
    """
    input_data_sort = np.sort(input_data)
    points_sort = np.sort(points)

    prob = []
    app = 0
    for val in input_data_sort:
        app += np.max(np.searchsorted(points_sort[app::], val), 0)
        prob.append(app)

    return np.array(prob)


def data_errors(data, method=None, symmetric=False, data_median=None):
    """Compute the errors of the median (or given) values for the given ``data``.

    Args:
        data (list or np.ndarray): 2d array with rows containing data points
            from which the median value is extracted.
        method (str or int or float, optional): method of computing the method. If `"std"`, computes the
            standard deviation. If type `float` or `int` between 0 and 100, computes the corresponding
            confidence interval using `np.percentile`. Otherwise, returns `None`. Defaults to `None`.
        symmetric (bool): If `False` and `method` is of type `float`, returns 2d array
            with 2 rows contanining lower and higher errors. If `True`, returns a list of errors
            corresponding to each mean value. Defaults to `False`.
        data_median (list or np.ndarray, optional): 1d array used to get the errors from the confidence interval.
            If `None`, the median values are computed from `data`.

    Returns:
        np.ndarray: errors of the data.
    """

    if method == "std":
        return np.std(data, axis=1)
    if isinstance(method, (int, float)) and 0 <= method <= 100:
        percentiles = [
            (100 - method) / 2,
            (100 + method) / 2,
        ]
        data_loc = data_median if data_median is not None else np.median(data, axis=1)
        data_errors = np.abs(
            np.vstack([data_loc, data_loc]) - np.percentile(data, percentiles, axis=1)
        )
        if symmetric:
            return np.max(data_errors, axis=0)
        return data_errors
    return None


def bootstrap(
    x_data: Union[np.ndarray, list],
    y_data: Union[np.ndarray, list],
    fit_func: Callable,
    sigma_method: Optional[Union[str, float]] = None,
    n_bootstrap: Optional[int] = 0,
    resample_func: Optional[Callable] = None,
    filter_estimates: Optional[Callable] = None,
    **kwargs,
):
    """Semiparametric bootstrap resampling.

    Args:
        x_data (list or np.ndarray): 1d array of x values.
        y_data (list or np.ndarray): 2d array with rows containing data points
            from which the mean values for y are computed.
        fit_func (callable): fitting function that returns parameters and errors, given
            `x`, `y`, `sigma` (if `sigma_method` is not `None`) and `**kwargs`.
        sigma_method (str or float, optional): method of computing `sigma` for the `fit_func`
            when `sigma` is not given in `kwargs`. If `std`, computes the standard deviation.
            If type `float` between 0 and 1, computes the maximum of low and high errors from
            the corresponding confidence interval. Otherwise, does not compute `sigma`. Defaults to `None`.
        n_bootstrap (int): number of bootstrap iterations. If `0`,
            returns `y_data` for y estimates and an empty list for fitting parameters estimates.
        resample_func (callable, optional): function that preforms resampling for a list of y values.
            (see :func:`qibocal.protocols.characterization.randomized_benchmarking.standard_rb.resample_p0`)
            If `None`, only non-parametric resampling is performed. Defaults to `None`.
        filter_estimates (callable, optional): if given, maps bootstrap estimates -> bool. Defaults to `None`.


    Returns:
        Tuple[list, list]: y data estimates and fitting parameters estimates.
    """

    if n_bootstrap == 0:
        return y_data, []

    if resample_func is None:
        resample_func = lambda data: data

    # Extract sigma for fit_func if given
    init_kwargs = deepcopy(kwargs)
    init_sigma = init_kwargs.pop("sigma", None)

    popt_estimates = []
    y_estimates = []

    for _ in range(n_bootstrap):
        bootstrap_y_scatter = []
        for y in y_data:
            # Non-parametric bootstrap: resample data points with replacement
            non_parametric_y = np.random.choice(y, size=len(y), replace=True)

            # Parametrically resample the new data
            bootstrap_y_scatter.append(resample_func(non_parametric_y))
        bootstrap_y = np.mean(bootstrap_y_scatter, axis=1)

        # Fit the resampled data to get parameters estimates
        bootstrap_sigma = (
            data_errors(
                bootstrap_y_scatter,
                sigma_method,
                symmetric=True,
                data_median=bootstrap_y,
            )
            if init_sigma is None
            else init_sigma
        )
        popt, _ = fit_func(x_data, bootstrap_y, sigma=bootstrap_sigma, **kwargs)

        # Filter obtained estimates
        if filter_estimates is None or filter_estimates(bootstrap_y, popt):
            y_estimates.append(bootstrap_y)
            popt_estimates.append(popt)

    return np.array(y_estimates).T, np.array(popt_estimates).T
