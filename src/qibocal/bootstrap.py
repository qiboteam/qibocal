import numpy as np


def data_uncertainties(data, method=None, data_median=None, homogeneous=True):
    """Compute the uncertainties of the median (or specified) values.

    Args:
        data (list or np.ndarray): 2d array with rows containing data points
            from which the median value is extracted.
        method (float, optional): method of computing the method.
            If it is `None`, computes the standard deviation, otherwise it
            computes the corresponding confidence interval using ``np.percentile``.
            Defaults to ``None``.
        data_median (list or np.ndarray, optional): 1d array for computing the errors from the
            confidence interval. If ``None``, the median values are computed from ``data``.
        homogeneous (bool): if ``True``, assumes that all rows in ``data`` are of the same size
            and returns ``np.ndarray``. Default is ``True``.

    Returns:
        np.ndarray: uncertainties of the data.
    """

    if method is None:
        return np.std(data, axis=1) if homogeneous else [np.std(row) for row in data]

    percentiles = [
        (100 - method) / 2,
        (100 + method) / 2,
    ]
    percentile_interval = np.percentile(data, percentiles, axis=1)

    uncertainties = np.abs(np.vstack([data_median, data_median]) - percentile_interval)

    return uncertainties
