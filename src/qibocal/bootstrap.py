from typing import Optional, Union

import numpy as np


def data_errors(data, method=None, symmetric=False, data_median=None):
    """Compute the errors of the median (or given) values for the given ``data``.

    Args:
        data (list or np.ndarray): 2d array with rows containing data points
            from which the median value is extracted.
        method (str or int or float, optional): method of computing the method. If ``"std"``, computes the
            standard deviation. If type ``float`` or ``int`` between 0 and 100, computes the corresponding
            confidence interval using ``np.percentile``. Otherwise, returns ``None``. Defaults to ``None``.
        symmetric (bool): If ``False`` and ``method`` is of type ``float``, returns 2d array
            with 2 rows contanining lower and higher errors. If ``True``, returns a list of errors
            corresponding to each mean value. Defaults to ``False``.
        data_median (list or np.ndarray, optional): 1d array for computing the errors from the confidence interval.
            If ``None``, the median values are computed from ``data``.

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
    data: Union[np.ndarray, list],
    n_bootstrap: int,
    homogeneous: bool = True,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
):
    """Non-parametric bootstrap resampling.

    Args:
        data (list or np.ndarray): 2d array with rows containing samples.
        n_bootstrap (int): number of bootstrap iterations. If ``0``, returns ``(data, [])``.
        homogeneous (bool): if ``True``, assumes that all rows in ``data`` are of the same size. Default is ``True``.
        sample_size (int, optional): number of samples per row in ``data``. If ``None``, defaults to ``len(data[0])``.
        seed (int, optional): A fixed seed to initialize ``np.random.Generator``. If ``None``,
            initializes a generator with a random seed. Defaults is ``None``.

    Returns:
        list or np.ndarray: resampled data of shape (len(data), sample_size, n_bootstrap)
    """

    local_state = np.random.default_rng(seed)

    if homogeneous:
        sample_size = sample_size or len(data[0])
        random_inds = local_state.integers(
            0, sample_size, size=(sample_size, n_bootstrap)
        )
        return np.array(data)[:, random_inds]

    bootstrap_y_scatter = []
    for row in data:
        bootstrap_y_scatter.append(
            local_state.choice(row, size=((sample_size or len(row)), n_bootstrap))
        )
    return bootstrap_y_scatter
