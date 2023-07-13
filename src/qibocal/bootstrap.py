from typing import Optional, Union

import numpy as np


def data_uncertainties(data, method=None, data_median=None, homogeneous=True):
    """Compute the uncertainties of the median (or specified) values.

    Args:
        data (list or np.ndarray): 2d array with rows containing data points
            from which the median value is extracted.
        method (str or int or float, optional): method of computing the method.
            If ``"std"``, computes the standard deviation. If number between 0 and 100,
            computes the corresponding confidence interval using ``np.percentile``.
            Otherwise, returns ``None``. Defaults to ``None``.
        data_median (list or np.ndarray, optional): 1d array for computing the errors from the
            confidence interval. If ``None``, the median values are computed from ``data``.
        homogeneous (bool): if ``True``, assumes that all rows in ``data`` are of the same size
            and returns ``np.ndarray``. Default is ``True``.

    Returns:
        np.ndarray: uncertainties of the data.
    """

    if method == "std":
        return np.std(data, axis=1) if homogeneous else [np.std(row) for row in data]
    if isinstance(method, (int, float)) and 0 <= method <= 100:
        percentiles = [
            (100 - method) / 2,
            (100 + method) / 2,
        ]
        if data_median is None:
            data_median = (
                np.median(data, axis=1)
                if homogeneous
                else [np.meadian(row) for row in data]
            )
        percentile_inteval = (
            np.percentile(data, percentiles, axis=1)
            if homogeneous
            else np.array([np.percentile(row, percentiles) for row in data]).T
        )
        uncertainties = np.abs(
            np.vstack([data_median, data_median]) - percentile_inteval
        )
        return uncertainties
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
        n_bootstrap (int): number of bootstrap iterations.
        homogeneous (bool): if ``True``, assumes that all rows in ``data`` are of the same size
            and returns ``np.ndarray``. If ``False``, returns a list of lists. Default is ``True``.
        sample_size (int, optional): number of samples per row in ``data``. If ``None``,
            defaults to ``len(row)`` from ``data``.
        seed (int, optional): A fixed seed to initialize ``np.random.Generator``. If ``None``,
            initializes a generator with a random seed. Defaults is ``None``.

    Returns:
        list or np.ndarray: resampled data of shape (len(data), sample_size, n_bootstrap)
    """

    random_generator = np.random.default_rng(seed)

    if homogeneous:
        sample_size = sample_size or len(data[0])
        random_inds = random_generator.integers(
            0, sample_size, size=(sample_size, n_bootstrap)
        )
        return np.array(data)[:, random_inds]

    bootstrap_y_scatter = []
    for row in data:
        bootstrap_y_scatter.append(
            random_generator.choice(row, size=((sample_size or len(row)), n_bootstrap))
        )
    return bootstrap_y_scatter
