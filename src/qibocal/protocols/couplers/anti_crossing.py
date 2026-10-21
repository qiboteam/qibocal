"""Coupler flux dependence from qubit-coupler avoided crossings.

Experiment
    Each qubit of a pair is probed in its own acquisition: a long spectroscopy pulse on
    the qubit drive plays together with a coupler flux pulse, then the qubit is read
    out. The amplitude of the flux pulse (the *bias*, applied on top of the coupler DC
    offset) and the drive frequency are swept.

Physics
    While the coupler is far detuned, the qubit line is flat. Where the coupler
    frequency ``f_c(x)`` crosses the qubit frequency ``fq`` they hybridise with coupling
    ``g``; with ``dx = x - xc`` and ``sc = f_c'(xc)`` the bare coupler slope, the two
    eigenbranches are

        f(x) = fq + sc*dx/2 + s*sqrt((sc*dx/2)**2 + g**2),   s = +1 or -1.

    The bright, qubit-like line follows ``s = -sign(sc*dx)``: it jumps by about
    ``-2 g sign(sc)`` at ``xc`` (jump sign opposite to ``sc``) and its tangent there is
    ``sc/2``. Hence each crossing gives a point ``f_c(xc) = fq`` of the coupler curve
    and, when the bend is resolved, its tangent ``f_c'(xc) = sc``.

Analysis
    1. Trace the qubit line with the ``qubit_flux`` peak extractor.
    2. Find its (at most two) steps and fit the qubit-like branch around each one.
    3. Fit the coupler curve, a parabola ``c2*x**2 + c1*x + c0`` valid near its
       sweetspot, to all points and tangents.

Units: frequencies in Hz, biases in a.u., slopes in Hz/a.u.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    GaussianSquare,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitPairId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.calibration.calibration import QubitCalibration
from qibocal.config import log
from qibocal.protocols.flux_dependence.qubit_flux_dependence import (
    _extract_peak_coordinates,
)
from qibocal.protocols.utils import (
    HZ_TO_GHZ,
    RangeLike,
    readout_frequency,
    reshaping_raw_signal,
    table_dict,
    table_html,
    to_range,
)
from qibocal.result import magnitude

__all__ = ["coupler_anticrossing"]

CouplerAnticrossingType = np.dtype(
    [("bias", float), ("frequency", float), ("signal", float)]
)
"""One point of a spectroscopy map: coupler bias, drive frequency, signal magnitude."""

WINDOW = 10
"""Half-width of the local fit around each crossing, in bias steps."""


@dataclass
class CouplerAnticrossingParameters(Parameters):
    """Coupler anticrossing runcard inputs."""

    bias: RangeLike
    """Coupler flux pulse amplitude range [a.u.], on top of the coupler DC offset."""
    frequency_0: RangeLike
    """Drive frequency range [Hz] of the first qubit, around its drive frequency."""
    frequency_1: RangeLike
    """Drive frequency range [Hz] of the second qubit, around its drive frequency."""
    pulse_duration: float
    """Duration of the drive and coupler flux pulses [ns]."""
    pulse_amplitude: float
    """Amplitude of the drive pulse [a.u.]."""


@dataclass
class CouplerAnticrossingData(Data):
    """Spectroscopy maps keyed by ``(q0, q1, probed qubit)``."""

    data: dict[tuple, npt.NDArray] = field(default_factory=dict)


@dataclass
class CouplerAnticrossingResults(Results):
    """Fitted crossings and coupler parabola of each pair."""

    crossings: dict[QubitPairId, list[dict]] = field(default_factory=dict)
    """Crossings with their errors: ``bias``, ``frequency``, ``gap`` (``2 g``), the
    locally fitted coupler slope ``fitted_slope`` and the tangent ``slope`` used in the
    coupler fit (``None`` when the bend is not traced), plus the probed ``qubit``."""
    coupler: dict[QubitPairId, dict] = field(default_factory=dict)
    """Parabola ``coefficients`` ``[c2, c1, c0]`` with their ``errors``, the vertex
    ``sweetspot`` [a.u.] and ``frequency`` [Hz], and the fit ``chi2`` and ``dof``."""

    def __contains__(self, key):
        # Crossings may exist without a coupler fit, and are still worth reporting.
        return tuple(key) in self.crossings


def spectroscopy(platform, pair, qubit, frequency, params, updates=None) -> dict:
    """Acquire the map of ``qubit`` versus coupler bias and drive frequency.

    ``frequency`` is a range around the qubit drive frequency and ``updates`` extra
    platform updates (e.g. a qubit flux offset). Returns flat ``bias``, ``frequency``
    and ``signal`` arrays, ready for :meth:`Data.register_qubit`.
    """
    coupler = platform.calibration.two_qubits[pair].coupler
    natives = platform.natives.single_qubit[qubit]
    qd_channel, _ = natives.RX()[0]
    ro_channel, ro_pulse = natives.MZ()[0]
    drive = Pulse(
        duration=params.pulse_duration,
        amplitude=params.pulse_amplitude,
        envelope=Rectangular(),
    )
    flux = Pulse(
        duration=params.pulse_duration,
        amplitude=to_range(params.bias)[0],
        envelope=GaussianSquare(sigma=0.2, risefall=0),
    )
    # Drive and coupler pulse play together; the readout waits for both to end.
    sequence = PulseSequence(
        [
            (qd_channel, drive),
            (platform.couplers[coupler].flux, flux),
            (ro_channel, Delay(duration=drive.duration)),
            (ro_channel, ro_pulse),
        ]
    )
    center = platform.config(qd_channel).frequency
    sweepers = [
        Sweeper(
            parameter=Parameter.amplitude, range=to_range(params.bias), pulses=[flux]
        ),
        Sweeper(
            parameter=Parameter.frequency,
            range=to_range(frequency, center),
            channels=[qd_channel],
        ),
    ]
    probe = {
        platform.qubits[qubit].probe: {"frequency": readout_frequency(qubit, platform)}
    }
    results = platform.execute(
        [sequence],
        [[sweeper] for sweeper in sweepers],
        updates=[probe | (updates or {})],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    # Results have shape (bias, frequency, I/Q): bias is the outer loop.
    bias, freq = np.meshgrid(sweepers[0].values, sweepers[1].values, indexing="ij")
    signal = magnitude(results[ro_pulse.id])
    return {"bias": bias.ravel(), "frequency": freq.ravel(), "signal": signal.ravel()}


def _acquisition(
    params: CouplerAnticrossingParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerAnticrossingData:
    """Probe each qubit of each pair in a separate acquisition."""
    data = CouplerAnticrossingData()
    for pair in map(tuple, targets):
        for qubit, frequency in zip(pair, (params.frequency_0, params.frequency_1)):
            record = spectroscopy(platform, pair, qubit, frequency, params)
            data.register_qubit(CouplerAnticrossingType, (*pair, qubit), record)
    return data


def branch(x, xc, fq, sc, g, s):
    """Eigenbranch ``s`` (+1 upper, -1 lower) of a flat qubit at ``fq`` coupled by
    ``g`` to a coupler of bare slope ``sc`` crossing it at ``xc``."""
    half = sc * (x - xc) / 2
    return fq + half + s * np.hypot(half, g)


def steps(x, y, resolution) -> list[tuple[int, int]]:
    """Steps of the trace ``(x, y)``, as ``(left, right)`` sample indices, largest first.

    Returns nothing for a scattered trace (e.g. noisy data), where steps are not
    distinguishable from noise.
    """
    jumps = np.diff(y)
    # Robust standard deviation of the jumps: most of them are just noise.
    noise = 1.4826 * np.median(np.abs(jumps - np.median(jumps)))
    significant = np.flatnonzero(np.abs(jumps) > max(5 * noise, 3 * resolution))
    if len(significant) > max(6, 0.05 * len(y)):
        return []
    # The extractor may report a few off-branch samples inside a step, splitting it
    # into several jumps: jumps closer than 5 samples form one step.
    found = []
    for jump in significant:
        if found and jump - found[-1][1] < 5:
            found[-1][1] = jump + 1
        else:
            found.append([jump, jump + 1])
    # Excursions (e.g. spikes) return to the same level and are not steps.
    found = [(l, r) for l, r in found if abs(y[r] - y[l]) > 3 * resolution]
    return sorted(found, key=lambda step: -abs(y[step[1]] - y[step[0]]))


def local_fit(x, y, left, right, window, resolution) -> dict | None:
    """Fit the qubit-like branch around the step from sample ``left`` to ``right``.

    Only samples within ``window`` of the step, and outside it, are used. Returns
    ``None`` if the fit fails or is degenerate.
    """
    step = y[right] - y[left]
    sign = -np.sign(step)  # sign of the bare coupler slope
    center = (x[left] + x[right]) / 2
    near = (np.abs(x - center) <= window) & ((x <= x[left]) | (x >= x[right]))
    xs, ys = x[near], y[near]
    # The branch of each sample is fixed by its side of the observed step, so that
    # it cannot jump during the optimisation (s = -sign(sc*dx), see module docs).
    s = np.where(xs <= x[left], sign, -sign)

    def model(x, *p):
        return branch(x, *p, s)

    # Bounds: crossing within the window, slope with the sign set by the step.
    reach = np.ptp(ys) + abs(step)
    max_slope = 1e3 * reach * WINDOW / window
    low, high = (0, max_slope) if sign > 0 else (-max_slope, 0)
    bounds = (
        [center - window, ys.min() - reach, low, 0],
        [center + window, ys.max() + reach, high, reach],
    )
    scale = [window / WINDOW, resolution, resolution * WINDOW / window, resolution]
    best, cost = None, np.inf
    # The bare slope cannot be guessed from the trace: try a few magnitudes.
    for ratio in (1, 0.3, 0.1, 0.03):
        p0 = [center, (y[left] + y[right]) / 2, sign * abs(step) / window / ratio]
        try:
            # A robust loss limits the weight of samples off the qubit-like branch.
            popt, pcov = curve_fit(
                model,
                xs,
                ys,
                p0 + [abs(step) / 2],
                bounds=bounds,
                loss="soft_l1",
                f_scale=resolution,
                x_scale=scale,
            )
        except (RuntimeError, ValueError):
            continue
        r = (model(xs, *popt) - ys) / resolution
        if (new_cost := np.sum(np.sqrt(1 + r**2) - 1)) < cost:
            best, cost = (popt, pcov), new_cost
    if best is None:
        return None
    (xc, fq, sc, g), pcov = best
    errors = np.sqrt(np.diag(pcov))
    if g == 0 or sc == 0 or abs(xc - center) > 0.99 * window:
        return None  # collapsed onto a bound
    if not np.all(np.isfinite(errors)):
        return None
    # The slope is only measured where the branch bends, |dx| < 2 g / |sc|. It is
    # used as tangent only if the step is a single jump (no off-branch samples
    # inside) and the bend holds at least two well-fitted samples on each side.
    inliers = np.abs(model(xs, xc, fq, sc, g) - ys) < 3 * resolution
    bend = inliers & (np.abs(xs - xc) * abs(sc) < 2 * g)
    sides = min(np.sum(bend & (xs < xc)), np.sum(bend & (xs > xc)))
    resolved = right - left == 1 and sides >= 2 and abs(sc) > 3 * errors[2]
    return {
        "bias": float(xc),
        "bias_error": float(errors[0]),
        "frequency": float(fq),
        "frequency_error": float(errors[1]),
        "gap": float(2 * g),
        "fitted_slope": float(sc),
        "slope": float(sc) if resolved else None,
        "slope_error": float(errors[2]) if resolved else None,
    }


def crossings(bias, frequency, signal) -> list[dict]:
    """Fit up to two crossings in a ``(bias, frequency)`` signal map."""
    x, y = _extract_peak_coordinates(frequencies=frequency, biases=bias, signal=signal)
    if len(y) < 5:
        return []
    resolution = np.median(np.diff(frequency))
    window = WINDOW * np.median(np.diff(bias))
    # A qubit crosses the coupler at most twice: keep the two largest steps that are
    # farther apart than the fit window.
    seeds = []
    for left, right in steps(x, y, resolution):
        if all(abs(x[left] - x[other]) > window for other, _ in seeds):
            seeds.append((left, right))
    fits = (local_fit(x, y, *seed, window, resolution) for seed in seeds[:2])
    return [fit for fit in fits if fit is not None]


def parabola_fit(crossings: list[dict]) -> dict | None:
    """Fit ``f_c(x) = c2*x**2 + c1*x + c0`` to crossing points and tangents.

    Each point gives ``f_c(x) = frequency`` and each tangent ``f_c'(x) = slope``: both
    are linear in the coefficients, so this is a weighted linear least squares.
    Returns ``None`` if the constraints cannot fix the three coefficients.
    """
    rows, values, errors = [], [], []
    for c in crossings:
        x = c["bias"]
        rows.append([x**2, x, 1])
        values.append(c["frequency"])
        # The bias error propagates to frequency through the coupler slope.
        errors.append(
            np.hypot(c["frequency_error"], c["fitted_slope"] * c["bias_error"])
        )
        if c["slope"] is not None:
            rows.append([2 * x, 1, 0])
            values.append(c["slope"])
            errors.append(c["slope_error"])
    if len(rows) < 3:
        return None
    a = np.array(rows) / np.array(errors)[:, None]
    b = np.array(values) / np.array(errors)
    coefficients, _, rank, _ = np.linalg.lstsq(a, b)
    if rank < 3:  # e.g. both points of a single qubit, without tangents
        return None
    dof = len(b) - 3
    chi2 = float(np.sum((a @ coefficients - b) ** 2))
    # Inflate the errors when the constraints scatter more than their errors.
    covariance = np.linalg.inv(a.T @ a) * (max(1.0, chi2 / dof) if dof else 1.0)
    c2, c1, c0 = coefficients
    return {
        "coefficients": coefficients.tolist(),
        "errors": np.sqrt(np.diag(covariance)).tolist(),
        "sweetspot": float(-c1 / (2 * c2)),
        "frequency": float(c0 - c1**2 / (4 * c2)),
        "chi2": chi2,
        "dof": dof,
    }


def _fit(data: CouplerAnticrossingData) -> CouplerAnticrossingResults:
    """Fit the crossings of each probed qubit, then the coupler of each pair."""
    results = CouplerAnticrossingResults()
    for key, record in data.data.items():
        frequency, bias, signal = reshaping_raw_signal(
            record["frequency"], record["bias"], record["signal"]
        )
        found = [c | {"qubit": key[2]} for c in crossings(bias, frequency, signal)]
        results.crossings.setdefault(tuple(key[:2]), []).extend(found)
    for pair, found in results.crossings.items():
        if (coupler := parabola_fit(found)) is not None:
            results.coupler[pair] = coupler
    return results


def plot_crossings(crossings, coupler, bias):
    """Figure with the crossings, their tangents and the coupler parabola over the
    ``bias`` range, and a report table of the coupler fit."""
    figure = go.Figure()
    for c in crossings:
        label = f"qubit {c['qubit']}"
        if "qubit_bias" in c:
            label += f", qubit bias {c['qubit_bias']:.4g}"
        figure.add_scatter(
            x=[c["bias"]],
            y=[c["frequency"] * HZ_TO_GHZ],
            error_x={"array": [c["bias_error"]]},
            mode="markers",
            marker={"size": 10},
            name=f"Crossing, {label}",
        )
        if c["slope"] is not None:
            x = c["bias"] + np.array([-0.05, 0.05]) * np.ptp(bias)
            y = c["frequency"] + c["slope"] * (x - c["bias"])
            figure.add_scatter(
                x=x, y=y * HZ_TO_GHZ, mode="lines", name=f"Tangent, {label}"
            )
    report = ""
    if coupler is not None:
        x = np.linspace(np.min(bias), np.max(bias), 200)
        y = np.polyval(coupler["coefficients"], x)
        figure.add_scatter(x=x, y=y * HZ_TO_GHZ, mode="lines", name="Coupler parabola")
        report = table_html(
            table_dict(
                "coupler",
                ["Sweetspot [a.u.]", "Frequency at sweetspot [GHz]", "chi2 / dof"],
                [
                    round(coupler["sweetspot"], 4),
                    round(coupler["frequency"] * HZ_TO_GHZ, 4),
                    f"{coupler['chi2']:.3g} / {coupler['dof']}",
                ],
            )
        )
    figure.update_layout(
        xaxis_title="Coupler flux pulse amplitude [a.u.]",
        yaxis_title="Frequency [GHz]",
    )
    return [figure], report


def _plot(
    data: CouplerAnticrossingData,
    target: QubitPairId,
    fit: CouplerAnticrossingResults | None = None,
):
    """Crossings of both qubits, their tangents and the coupler parabola."""
    pair = tuple(target)
    if fit is None:
        return [], ""
    bias = np.concatenate([r["bias"] for k, r in data.data.items() if k[:2] == pair])
    return plot_crossings(fit.crossings[pair], fit.coupler.get(pair), bias)


def _update(
    results: CouplerAnticrossingResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    """Store the coupler maximum frequency and sweetspot.

    The sweetspot is the coupler DC offset plus the vertex bias, which assumes that
    flux pulse amplitude and DC offset share the same scale.
    """
    pair = tuple(target)
    coupler = results.coupler[pair]
    if coupler["coefficients"][0] >= 0:
        # A transmon coupler has a frequency maximum at its sweetspot.
        log.warning(f"Coupler parabola of {pair} has no maximum: no update.")
        return
    coupler_id = platform.calibration.two_qubits[pair].coupler
    offset = platform.config(platform.couplers[coupler_id].flux).offset
    calibration = platform.calibration.couplers.setdefault(
        coupler_id, QubitCalibration()
    )
    update.sweetspot(offset + coupler["sweetspot"], platform, coupler_id, coupler=True)
    calibration.qubit.maximum_frequency = coupler["frequency"]


coupler_anticrossing = Protocol(_acquisition, _fit, _plot, _update)
"""CouplerAnticrossing Routine object."""
