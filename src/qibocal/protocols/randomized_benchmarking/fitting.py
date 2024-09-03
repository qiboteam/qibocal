"""In this python script the fitting methods for the gate set protocols are defined.
They consist mostly of exponential decay fitting.
"""

from typing import Optional, Union

import numpy as np
from scipy.linalg import hankel, svd
from scipy.optimize import curve_fit

from qibocal.config import log, raise_error


def exp1_func(x: np.ndarray, A: float, f: float) -> np.ndarray:
    """Return :math:`A\\cdot f^x` where ``x`` is an ``np.ndarray`` and
    ``A``, ``f`` are floats
    """
    return A * f**x


def exp1B_func(x: np.ndarray, A: float, f: float, B: float) -> np.ndarray:
    """Return :math:`A\\cdot f^x+B` where ``x`` is an ``np.ndarray`` and
    ``A``, ``f``, ``B`` are floats
    """
    return A * f**x + B


def exp2_func(x: np.ndarray, A1: float, A2: float, f1: float, f2: float) -> np.ndarray:
    """Return :math:`A_1\\cdot f_1^x+A_2\\cdot f_2^x` where ``x`` is an ``np.ndarray`` and
    ``A1``, ``f1``, ``A2``, ``f2`` are floats. There is no linear offsett B.
    """
    x = np.array(x, dtype=complex)
    return A1 * f1**x + A2 * f2**x


def esprit(
    xdata: np.ndarray,
    ydata: np.ndarray,
    num_decays: int,
    hankel_dim: Optional[int] = None,
) -> np.ndarray:
    """Implements the ESPRIT algorithm for peak detection.

    Args:
        xdata (np.ndarray): Labels of data. Has to be equally spaced.
        ydata (np.ndarray): The data where multiple decays are fitted in.
        num_decays (int): How many decays should be fitted.
        hankel_dim (int | None, optional): The Hankel dimension. Defaults to None.

    Returns:
        np.ndarray: The decay parameters.

    Raises:
        ValueError: When the x-labels are not equally spaced the algorithm does not work.
    """
    # Check for equal spacing.
    if np.any(xdata[1:] - xdata[:-1] != xdata[1] - xdata[0]):
        raise_error(ValueError, "xdata has to be equally spaced.")
    sampleRate = 1 / (xdata[1] - xdata[0])
    # xdata has to be an array.
    xdata = np.array(xdata)
    # Define the Hankel dimension if not given.
    if hankel_dim is None:
        hankel_dim = int(np.round(0.5 * xdata.size))
    # Fine tune the dimension of the hankel matrix such that the mulitplication
    # processes don't break.
    hankel_dim = max(num_decays + 1, hankel_dim)
    hankel_dim = min(hankel_dim, xdata.size - num_decays + 1)
    hankelMatrix = hankel(ydata[:hankel_dim], ydata[(hankel_dim - 1) :])
    # Calculate nontrivial (nonzero) singular vectors of the hankel matrix.
    U, _, _ = svd(hankelMatrix, full_matrices=False)
    # Cut off the columns to the amount which is needed.
    U_signal = U[:, :num_decays]
    # Calculte the solution.
    spectralMatrix = np.linalg.pinv(U_signal[:-1,]) @ U_signal[1:,]
    # Calculate the poles/eigenvectors and space them right. Return them.
    decays = np.linalg.eigvals(spectralMatrix)
    decays = np.array(decays, dtype=complex)
    return decays**sampleRate


def fit_exp1B_func(
    xdata: Union[np.ndarray, list], ydata: Union[np.ndarray, list], **kwargs
) -> tuple[tuple, tuple]:
    """Calculate an single exponential A*p^m+B fit to the given ydata.

    Args:
        xdata (Union[np.ndarray, list]): The x-labels.
        ydata (Union[np.ndarray, list]): The data to be fitted.

    Returns:
        tuple[tuple, tuple]: The fitting parameters (A, p, B) and the estimated error
                             (A_err, p_err, B_err)
    """

    # Check if all the values in ``ydata``are the same. That would make the
    # exponential fit unnecessary.
    if np.all(ydata == ydata[0]):
        popt, perr = (ydata[0], 1.0, 0), (0, 0, 0)
    else:
        kwargs.setdefault("p0", (np.max(ydata) - np.min(ydata), 0.9, np.min(ydata)))
        # If the search for fitting parameters does not work just return
        # fixed parameters where one can see that the fit did not work
        try:
            popt, pcov = curve_fit(
                exp1B_func,
                xdata,
                ydata,
                **kwargs,
            )
            popt = tuple(popt)
            perr = tuple(np.sqrt(np.diag(pcov)))
        except Exception as e:
            log.warning("Ap^x+B fit: the fitting was not succesful. %s", e)
            popt, perr = (0, 0, 0), (0, 0, 0)
    return popt, perr


def fit_exp1_func(
    xdata: Union[np.ndarray, list], ydata: Union[np.ndarray, list], **kwargs
) -> tuple[tuple, tuple]:
    """Calculate an single exponential  A*p^m fit to the given ydata, no linear offset.

    Args:
        xdata (Union[np.ndarray, list]): The x-labels.
        ydata (Union[np.ndarray, list]): The data to be fitted.

    Returns:
        tuple[tuple, tuple]: The fitting parameters (A, p) and the estimated error (A_err, p_err).
    """

    # Check if all the values in ``ydata``are the same. That would make the
    # exponential fit unnecessary.
    if np.all(ydata == ydata[0]):
        popt, perr = (ydata[0], 1.0), (0, 0)
    else:
        # If the search for fitting parameters does not work just return
        # fixed parameters where one can see that the fit did not work
        try:
            kwargs.setdefault("p0", (np.max(ydata) - np.min(ydata), 0.9))
            # Build a new function such that the linear offset is zero.
            popt, pcov = curve_fit(exp1_func, xdata, ydata, **kwargs)
            perr = tuple(np.sqrt(np.diag(pcov)))
        except Exception as e:
            log.warning("Ap^x fit: the fitting was not succesful. %s", e)
            popt, perr = (0, 0), (0, 0)

    return popt, perr


def fit_expn_func(
    xdata: Union[np.ndarray, list],
    ydata: Union[np.ndarray, list],
    n: int = 2,
) -> tuple[tuple, tuple]:
    """Calculate n exponentials on top of each other, fit to the given ydata.
    No linear offset, the ESPRIT algorithm is used to identify ``n`` exponential decays.

    Args:
        xdata (Union[np.ndarray, list]): The x-labels.
        ydata (Union[np.ndarray, list]): The data to be fitted.
        n (int): number of decays to fit. Default is 2.

    Returns:
        tuple[tuple, tuple]: (A1, ..., An, f1, ..., fn) with f* the decay parameters.
    """

    # TODO how are the errors estimated?
    # TODO the data has to have a sufficiently big size, check that.
    decays = esprit(np.array(xdata), np.array(ydata), n)
    vandermonde = np.array([decays**x for x in xdata])
    alphas = np.linalg.pinv(vandermonde) @ np.array(ydata).reshape(-1, 1).flatten()
    return tuple([*alphas, *decays]), (0,) * (len(alphas) + len(decays))


def fit_exp2_func(
    xdata: Union[np.ndarray, list],
    ydata: Union[np.ndarray, list],
) -> tuple[tuple, tuple]:
    """Calculate 2 exponentials on top of each other, fit to the given ydata.

    No linear offset, the ESPRIT algorithm is used to identify the two exponential decays.

    Args:
        xdata (Union[np.ndarray, list]): The x-labels.
        ydata (Union[np.ndarray, list]): The data to be fitted

    Returns:
        tuple[tuple, tuple]: (A1, A2, f1, f2) with f* the decay parameters.
    """

    return fit_expn_func(xdata, ydata, 2)
