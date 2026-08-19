"""Frequency/period estimation and chi2 statistics for qibocal protocols."""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import TypeAdapter

from .constants import EXTREME_CHI


def chi2_reduced(
    observed: NDArray,
    estimated: NDArray,
    errors: NDArray,
    dof: float | None = None,
) -> float:
    if np.count_nonzero(errors) < len(errors):
        return EXTREME_CHI

    if dof is None:
        dof = len(observed) - 1

    chi2 = np.sum(np.square((observed - estimated) / errors)) / dof

    return float(chi2)


def chi2_reduced_complex(
    observed: tuple[NDArray, NDArray],
    estimated: NDArray,
    errors: tuple[NDArray, NDArray],
    dof: float | None = None,
) -> float:
    observed_complex = np.abs(observed[0] * np.exp(1j * observed[1]))

    observed_real = np.real(observed_complex)
    observed_imag = np.imag(observed_complex)

    estimated_real = np.real(estimated)
    estimated_imag = np.imag(estimated)

    observed_error_real = np.sqrt(
        (np.cos(observed[1]) * errors[0]) ** 2
        + (observed[0] * np.sin(observed[1]) * errors[1]) ** 2
    )
    observed_error_imag = np.sqrt(
        (np.sin(observed[1]) * errors[0]) ** 2
        + (observed[0] * np.cos(observed[1]) * errors[1]) ** 2
    )

    chi2_real = chi2_reduced(observed_real, estimated_real, observed_error_real, dof)
    chi2_imag = chi2_reduced(observed_imag, estimated_imag, observed_error_imag, dof)

    return chi2_real + chi2_imag


def guess_frequency(x: NDArray, y: NDArray, axis: int = -1) -> NDArray:
    """Return FFT frequency estimation given a sinusoidal plot.

    Computes the dominant frequency component from the FFT of the signal.
    axis represents the dimension along which to compute the FFT.
    This parameter specifies which dimension of y contains the sinusoidal
    data to analyze. For multi-dimensional arrays, the FFT is computed
    along this axis, and the dominant frequency is determined for each
    slice along other dimensions.
    """
    assert x.ndim == 1, f"Expected 1D array, got array with shape {x.shape}"

    y = np.moveaxis(y, axis, -1)
    fft = np.fft.rfft(y, axis=-1)
    fft_freqs = np.fft.rfftfreq(y.shape[-1], d=(x[1] - x[0]))
    mags = np.abs(fft)
    mags[:, 0] = 0

    selected_freqs = fft_freqs[np.argmax(mags, axis=-1)]

    return np.moveaxis(selected_freqs, -1, axis)


def fallback_frequency(frequency: NDArray) -> NDArray:
    """Return a numeric frequency array with NaNs replaced by a fallback value.

    This helper is used to ensure a valid numeric frequency is available for
    downstream processing when the frequency estimation returns missing values.
    """
    assert frequency.ndim <= 1, (
        f"Expected 1D array or scalar, got array with shape {frequency.shape}"
    )

    return np.where(np.isnan(frequency), 0.25, frequency)


def quinn_fernandes_algorithm(
    signal: Any,
    x: Any,
    axis: int = -1,
    speedup_flag: bool = False,
    iterations: int = 100,
    tol: int = 1e-8,
) -> NDArray:
    """This is a custom implementation of the Quinn-Fernandes algorithm.
    We compute the signal sampling rate from :param:x, hence this function assumes x to be
    ordered.

    The Quinn–Fernandes method is a high-accuracy frequency estimator based on
    phase interpolation of the discrete Fourier transform (DFT). It refines the
    peak frequency obtained from the FFT by analyzing the phase evolution of the
    complex spectrum, achieving super-resolution beyond the FFT bin spacing.
    If :const:`speedup_flag` is set to `True`, the algorithm will change the updating rule,
    can lead to faster convergence, especially when the initial guess is close to the true frequency.
    Link for the original paper: https://www.jstor.org/stable/2337018?seq=3
    """

    fs = 1 / abs(x[0] - x[1])

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    omegas = 2 * np.pi * fallback_frequency(guess_frequency(x, signal, axis=axis))
    alpha = 2 * np.cos(omegas)

    signal = signal - np.mean(signal, axis=axis, keepdims=True)

    sig_shape = list(signal.shape)
    sig_shape[axis] += 2
    buffer_beta = []
    for _ in range(iterations):
        xi = np.zeros(sig_shape)
        for t in range(2, xi.shape[axis]):
            xi[..., t] = signal[..., t - 2] + alpha * xi[..., t - 1] - xi[..., t - 2]

        beta = np.sum((xi[..., 2:] + xi[..., :-2]) * xi[..., 1:-1], axis=axis) / np.sum(
            xi[..., :-1] ** 2, axis=axis
        )
        beta[np.isnan(beta)] = 0
        if len(buffer_beta) >= 5:
            buffer_beta.pop(0)
        buffer_beta.append(beta)
        if np.all(np.abs(np.mean(buffer_beta, axis=0) - alpha) < tol):
            alpha = beta
            break

        if speedup_flag:
            alpha = beta
        else:
            alpha = 2 * beta - alpha

    alpha = np.clip(alpha, -2, 2)
    omega_est = np.arccos(alpha / 2)
    med_omega = np.median(omega_est)

    return med_omega * fs


def guess_period(
    x: NDArray, y: NDArray, axis: int = -1, speedup_flag: bool = True
) -> NDArray:
    """Estimate the period(s) of a (set of) sinusoidal signal(s).

    This is a thin wrapper around :func:`guess_frequency` that returns the
    reciprocal of the estimated dominant frequency. For multi-dimensional
    signals, the period is computed along ``axis`` of ``y`` and returned with
    the same shape semantics as :func:`guess_frequency`.
    """

    return (
        2
        * np.pi
        / quinn_fernandes_algorithm(
            signal=y,
            x=x,
            axis=axis,
            speedup_flag=speedup_flag,
        )
    )


def fallback_period(period: NDArray) -> NDArray:
    """Return a numeric period array with NaNs replaced by a fallback value.

    This helper is used to ensure a valid numeric period is available for
    downstream processing when the period estimation returns missing values.
    """
    assert period.ndim <= 1, (
        f"Expected 1D array or scalar, got array with shape {period.shape}"
    )

    return np.where(np.isnan(period), 4, period)


Range = tuple[float, float, float]
"""Value range, corresponding to ``(start, stop, step)``."""

RangeLike = (
    tuple[float, float, float]
    | tuple[Literal["linspace"], float, float, int]
    | tuple[Literal["window"], float, float, float]
    | tuple[Literal["linwindow"], float, float, int]
    | tuple[Literal["center"], float, float]
    | tuple[Literal["lincenter"], float, int]
    | tuple[Literal["asym"], tuple[float, float], float]
    | tuple[Literal["linasym"], tuple[float, float], int]
)
"""Range specification.

The alternative options allow for multiple representations of a range, which could all
be mapped to a sequence of evenly spaced values.

The semantics of the default one is equivalent to that of the built-in :func:`range`, in
which the three values correspond to ``(start, stop, step)``. Unlike :func:`range`, all
values are mandatory, since the usual defaults are often unsuitable.

The other variants are unambiguously discriminated by a starting label, e.g.
``("linspace", 1.2, 1.5, 70)``.

In the ``linspace`` variant, the ``step`` element is replaced with the number of steps.
Since the step specification is always mandatory, for each of the other variants a
further ``lin<...>`` version is also available, making the same substitution.

The other variants are the following:

- ``window``, which allows to specify the center and width of the window, instead of its
  extremes, i.e. ``("window", center, width, step)``
- ``center``, similar to the former, but assuming a default for the center, which
  depends the chosen protocol, i.e. ``("center", width, step)``
- ``asym``, which is also a way to build upon an implicit center, but with specifying an
  asymmetric interval around it, i.e. ``("asim", (left-shift, right-shift), step)``
"""

_RangeLike = TypeAdapter(RangeLike)


def to_range(spec: RangeLike, center: float | None = None) -> Range:
    """Convert any range specification into the default representation."""

    spec_ = _RangeLike.validate_python(spec)
    mode = spec_[0]

    if not isinstance(mode, str):
        # Default case: assume it's a tuple of (start, stop, step)
        return spec_

    if any(lab in mode for lab in ("center", "asym")) and center is None:
        raise ValueError(f"Center must be provided for '{mode}' range specification.")

    if mode == "linspace":
        start, stop = spec_[1:3]
    elif mode.endswith("window"):
        center_, width = spec_[1:3]
        start, stop = center_ - width / 2, center_ + width / 2
    elif mode.endswith("center"):
        width = spec_[1]
        start, stop = center - width / 2, center + width / 2
    elif mode.endswith("asym"):
        left_shift, right_shift = spec_[1]
        start, stop = center - left_shift, center + right_shift

    if mode.startswith("lin"):
        n = spec_[-1]
        if n <= 1:
            raise ValueError(f"At least 2 steps required for {mode}, passed {n}")
        step = (stop - start) / (n - 1)
        return start, stop, step
    return start, stop, spec_[-1]
