from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np


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
    n_bootstrap: int,
    sigma_method: Optional[Union[str, float]] = None,
    resample_func: Optional[Callable] = None,
    filter_estimates: Optional[Callable] = None,
    **kwargs,
):
    """Semiparametric bootstrap resampling.

    Steps:
        1. (Non-parametric resampling) For each row in `y_data`, draw the same number of samples with replacement.
        2. (Semi-parametric resampling) Parametrically resample obtained values if `resample_func` is given.
        3. Compute the mean of the samples.
        4. Repeat for each row in `y_data`.
        5. Fit the bootstrap estimates of `y` with `fit_func` and record new fitting parameters.
        6. If `filter_estimates` is given, check whether the obtained estimates should be recorded.

    Args:
        x_data (list or np.ndarray): 1d array of x values.
        y_data (list or np.ndarray): 2d array with rows containing data points
            from which the mean values for y are computed.
        fit_func (callable): fitting function that returns parameters and errors, given
            `x`, `y`, `sigma` (if `sigma_method` is not `None`) and `**kwargs`.
        n_bootstrap (int): number of bootstrap iterations. If `0`, returns `(y_data, [])`.
        sigma_method (str or float, optional): method of computing `sigma` for the `fit_func`
            when `sigma` is not given in `kwargs`. If `std`, computes the standard deviation.
            If type `float` between 0 and 1, computes the maximum of low and high errors from
            the corresponding confidence interval. Otherwise, does not compute `sigma`. Defaults to `None`.
        resample_func (callable, optional): function that maps a list of non-parametrically resmapled `y` values to a new list.
            (see :func:`qibocal.protocols.characterization.randomized_benchmarking.standard_rb.resample_p0`)
            If `None`, only non-parametric resampling is performed. Defaults to `None`.
        filter_estimates (callable, optional): function that returns `False` if bootstrap estimates should not be added.
            Maps `list, list` to `bool`. Defaults to `None`.
        kwargs: parameters passed to the `fit_func`.

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
