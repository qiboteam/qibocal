"""ZZ-interaction strength versus coupler flux-pulse amplitude.

Conditional Ramsey experiment on the target qubit, repeated with the control
qubit in :math:`|0\\rangle` and :math:`|1\\rangle`, while applying a rectangular
coupler flux pulse during the Ramsey wait. The pulsed ZZ coupling is extracted as

.. math::

    \\zeta(A) = f_{\\text{target}}^{(c=1)}(A)
               - f_{\\text{target}}^{(c=0)}(A),

where :math:`f_{\\text{target}}` is the physical conditional target frequency,
not the unsigned detuned fringe frequency. The
minimum-coupling pulse amplitude is located with a local quadratic fit to
:math:`\\zeta(A)`. Population-only Ramsey measures unsigned fringe frequencies:
both conditional detunings must stay on the branch selected by ``detuning``.

References:
    - Collodo et al., "Implementation of Conditional-Phase Gates based on
      tunable ZZ-Interactions", PRL 125, 240502 (2020).
    - Sung et al., "Realization of High-Fidelity CZ and ZZ-Free iSWAP Gates
      with a Tunable Coupler", PRX 11, 021058 (2021).
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    Rectangular,
    Sweeper,
)
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from qibocal.auto.operation import Data, Parameters, Protocol, QubitPairId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.ramsey.acquisition import single_qubit_ramsey_sequence
from qibocal.protocols.utils import (
    RangeLike,
    table_dict,
    table_html,
    to_range,
)

__all__ = ["coupler_zz"]

SETUPS = ["I", "X"]
"""Control qubit preparations: identity (ground state) and X (excited state)."""

HZ_TO_KHZ = 1e-3
"""Conversion factor from Hz to kHz."""


def _sweep_values(spec: RangeLike, center: float = 0) -> npt.NDArray:
    """Expand a range specification, retaining the endpoint for linspace variants."""
    start, stop, step = to_range(spec, center=center)
    if not np.all(np.isfinite([start, stop, step])) or step <= 0 or stop <= start:
        raise ValueError(
            "Sweep ranges require an increasing interval and positive step."
        )
    if isinstance(spec[0], str) and spec[0].startswith("lin"):
        return np.linspace(start, stop, spec[-1])
    return np.arange(start, stop, step)


@dataclass
class CouplerZZParameters(Parameters):
    """CouplerZZ runcard inputs."""

    delay_range: RangeLike
    """Ramsey wait and coupler pulse duration sweep [ns]."""
    amplitude_range: RangeLike
    """Coupler flux-pulse amplitude sweep [a.u.], relative to the idle bias."""
    detuning: float = 1e6
    """Artificial detuning of the target drive frequency [Hz], to produce
    Ramsey fringes."""


@dataclass
class CouplerZZResults(Results):
    """CouplerZZ outputs."""

    amplitudes: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """Coupler flux-pulse amplitudes at which ZZ was evaluated."""
    zz: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """ZZ coupling [Hz] for each pulse amplitude (NaN where fitting failed)."""
    zz_free_amplitude: dict[QubitPairId, float] = field(default_factory=dict)
    """Pulse amplitude minimizing fitted |ZZ| within the measured interval.

    This is a minimum, not necessarily a zero. No extrapolation is performed."""
    residual_zz: dict[QubitPairId, float] = field(default_factory=dict)
    """Residual ZZ [Hz] at the amplitude closest to the ZZ-free point."""
    zz_error: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """One-sigma ZZ fit uncertainties [Hz], NaN for failed fits."""
    minimum_zz: dict[QubitPairId, float] = field(default_factory=dict)
    """ZZ [Hz] predicted at the fitted minimum."""
    polynomial: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """Local polynomial coefficients in ascending powers of pulse amplitude."""
    fit_amplitudes: dict[QubitPairId, list[float]] = field(default_factory=dict)
    """Interval over which the local polynomial is valid."""
    fitted_parameters: dict[QubitPairId, dict[str, list[list[float]]]] = field(
        default_factory=dict
    )
    """Per-amplitude [offset, cosine, sine, frequency Hz, decay 1/ns, time origin ns]."""

    def __contains__(self, pair: QubitPairId) -> bool:
        return pair in self.zz


CouplerZZType = np.dtype(
    [
        ("wait", np.float64),
        ("amplitude", np.float64),
        ("prob", np.float64),
    ]
)
"""Custom dtype for coupler ZZ acquisition."""


@dataclass
class CouplerZZData(Data):
    """CouplerZZ acquisition outputs."""

    detuning: float = 1e6
    """Artificial detuning of the target drive frequency [Hz]."""
    flux_offsets: dict[QubitPairId, float] = field(default_factory=dict)
    """Acquisition DC coupler offsets, for absolute-bias calibration."""
    data: dict[tuple[QubitPairId, str], npt.NDArray[CouplerZZType]] = field(
        default_factory=dict
    )
    """Raw data acquired, keyed by (pair, control setup)."""

    @property
    def frequency_scale(self) -> float:
        """Convert the unsigned conditional fringe difference to physical ZZ."""
        return -float(np.sign(self.detuning))

    @property
    def pairs(self) -> list[QubitPairId]:
        """Qubit pairs in the data (keys are (pair, setup) tuples)."""
        return list({key[0] for key in self.data})

    def waits(self, pair: QubitPairId) -> npt.NDArray:
        """Ramsey wait times for a pair."""
        return np.unique(
            np.concatenate(
                [
                    records["wait"]
                    for (key, _), records in self.data.items()
                    if key == pair
                ]
            )
        )

    def amplitudes(self, pair: QubitPairId) -> npt.NDArray:
        """Coupler flux-pulse amplitudes for a pair."""
        return np.unique(
            np.concatenate(
                [
                    records["amplitude"]
                    for (key, _), records in self.data.items()
                    if key == pair
                ]
            )
        )


def _acquisition(
    params: CouplerZZParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> CouplerZZData:
    """Conditional Ramsey on the target qubit versus coupler pulse amplitude.

    For each control state the sequence is::

        control: [RX if state == 1]
        target:               RX90 -- wait -- RX90 -- MZ
        coupler:                     -- flux pulse --

    The inner sweeper varies both the Ramsey wait and the flux-pulse duration;
    the outer sweeper varies the pulse amplitude, leaving the DC bias unchanged.
    """

    if len(targets) != 1:
        raise ValueError("coupler_zz requires exactly one qubit pair.")
    if (
        params.detuning is None
        or not np.isfinite(params.detuning)
        or params.detuning == 0
    ):
        raise ValueError("A finite nonzero detuning is required to resolve ZZ.")
    pair = targets[0]
    target_qubit, control_qubit = pair
    if target_qubit == control_qubit:
        raise ValueError("Target and control must be different qubits.")

    coupler_id = platform.calibration.two_qubits[pair].coupler
    if coupler_id is None:
        raise ValueError("Coupler ID should not be None.")
    coupler_channel = platform.couplers[coupler_id].flux
    if coupler_channel is None:
        raise ValueError("Coupler flux channel should not be None.")
    waits = _sweep_values(params.delay_range)
    amplitudes = _sweep_values(params.amplitude_range)
    if waits[0] <= 0:
        raise ValueError("Ramsey waits and coupler pulse durations must be positive.")
    if len(waits) < 8:
        raise ValueError("At least eight Ramsey waits are required for joint fitting.")
    if abs(params.detuning) >= 0.5e9 / np.max(np.diff(waits)):
        raise ValueError("Detuning must be below the wait sweep's Nyquist frequency.")
    data = CouplerZZData(
        detuning=params.detuning,
        flux_offsets={pair: platform.config(coupler_channel).offset},
    )

    updates = []
    if params.detuning is not None:
        drive_channel = platform.qubits[target_qubit].drive
        f0 = platform.config(drive_channel).frequency
        updates.append({drive_channel: {"frequency": f0 + params.detuning}})

    for setup in SETUPS:
        sequence, ro_sequence, delays = single_qubit_ramsey_sequence(
            platform=platform, target=target_qubit, wait=waits[0]
        )
        rx90_duration = (
            platform.natives.single_qubit[target_qubit].R(theta=np.pi / 2).duration
        )
        coupler_pulse = Pulse(
            duration=waits[0],
            amplitude=amplitudes[0],
            envelope=Rectangular(),
        )
        sequence.extend(
            [
                (coupler_channel, Delay(duration=rx90_duration)),
                (coupler_channel, coupler_pulse),
            ]
        )
        ramsey_sequence = sequence + ro_sequence
        if setup == "X":
            # prepare the control qubit in the excited state
            experiment_sequence = (
                platform.natives.single_qubit[control_qubit].RX() | ramsey_sequence
            )
        else:
            experiment_sequence = ramsey_sequence

        wait_sweeper = Sweeper(
            parameter=Parameter.duration,
            values=waits,
            pulses=[*delays, coupler_pulse],
        )
        amplitude_sweeper = Sweeper(
            parameter=Parameter.amplitude,
            values=amplitudes,
            pulses=[coupler_pulse],
        )
        results = platform.execute(
            [experiment_sequence],
            [[amplitude_sweeper], [wait_sweeper]],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            updates=updates,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        ro_pulse = list(
            experiment_sequence.channel(platform.qubits[target_qubit].acquisition)
        )[-1]
        # probs has shape (len(amplitudes), len(waits))
        probs = np.asarray(results[ro_pulse.id])
        waits = wait_sweeper.values
        amplitudes = amplitude_sweeper.values

        data.register_qubit(
            CouplerZZType,
            (pair, setup),
            {
                "wait": np.tile(waits, len(amplitudes)),
                "amplitude": np.repeat(amplitudes, len(waits)),
                "prob": probs.ravel(),
            },
        )

    return data


def _conditional_fit(
    waits: npt.NDArray, probs: npt.NDArray, detuning: float
) -> tuple[float, float, list[list[float]]]:
    """Joint damped-fringe fit, with a shared target dephasing rate.

    Variable projection solves each trace's offset and two quadratures exactly
    at every nonlinear step. Multiple spectral seeds avoid harmonic/local minima.
    Frequencies are positive; their physical branch is set by the acquisition.
    """
    waits = np.asarray(waits, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if (
        waits.ndim != 1
        or len(waits) < 8
        or probs.shape != (2, len(waits))
        or not np.all(np.isfinite(waits))
        or not np.all(np.isfinite(probs))
        or np.any((probs < 0) | (probs > 1))
        or np.any(np.diff(waits) <= 0)
        or np.any(np.ptp(probs, axis=1) < 1e-6)
    ):
        raise ValueError(
            "Need two finite, nonconstant probability traces and >=8 waits."
        )
    span = np.ptp(waits)
    x = (waits - waits[0]) / span
    # Conservative band also applies to nonuniform sampling.
    upper_frequency = span / (2 * np.max(np.diff(waits)))
    grid = np.linspace(0.05, upper_frequency * 0.999, max(64, 8 * len(waits)))
    if upper_frequency <= 0.1:
        raise ValueError("Insufficient sampling bandwidth.")

    def design(frequency, decay):
        angle = 2 * np.pi * frequency * x
        envelope = np.exp(-decay * x)
        return np.column_stack(
            (np.ones_like(x), envelope * np.cos(angle), envelope * np.sin(angle))
        )

    def residual(parameters):
        f0, f1, decay = parameters
        residuals = []
        for frequency, trace in zip((f0, f1), probs):
            matrix = design(frequency, decay)
            coefficients = np.linalg.lstsq(matrix, trace, rcond=None)[0]
            residuals.append(matrix @ coefficients - trace)
        return np.concatenate(residuals)

    seeds = []
    for trace in probs:
        power = np.array(
            [
                np.sum(
                    (
                        trace
                        - design(frequency, 0)
                        @ np.linalg.lstsq(design(frequency, 0), trace, rcond=None)[0]
                    )
                    ** 2
                )
                for frequency in grid
            ]
        )
        peaks, _ = find_peaks(-power)
        candidates = list(peaks[np.argsort(power[peaks])[:3]])
        candidates.append(int(np.argmin(power)))
        frequencies = [grid[index] for index in dict.fromkeys(candidates)]
        carrier = abs(detuning) * span * 1e-9
        if 0.01 < carrier < upper_frequency:
            frequencies.append(carrier)
        seeds.append(frequencies)

    best = None
    for f0 in seeds[0]:
        for f1 in seeds[1]:
            result = least_squares(
                residual,
                [f0, f1, 0.2],
                bounds=([0.001, 0.001, 0], [upper_frequency, upper_frequency, 20]),
                max_nfev=1000,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            if result.success and (best is None or result.cost < best.cost):
                best = result
    if best is None:
        raise RuntimeError("Joint conditional Ramsey fit did not converge.")
    f0, f1, decay = best.x
    if (
        min(f0, f1) < 0.1
        or max(f0, f1) > 0.99 * upper_frequency
        or decay > 19.9
        or np.linalg.matrix_rank(best.jac) < 3
    ):
        raise ValueError("Unresolved, aliased or ill-conditioned conditional fringes.")
    parameters = []
    noise = np.sqrt(2 * best.cost / (probs.size - 9))
    for frequency, trace in zip((f0, f1), probs):
        offset, cosine, sine = np.linalg.lstsq(
            design(frequency, decay), trace, rcond=None
        )[0]
        oscillation = design(frequency, decay)[:, 1:] @ [cosine, sine]
        if (
            np.hypot(cosine, sine) < 1e-4
            or np.linalg.norm(oscillation - np.mean(oscillation)) < 5 * noise
        ):
            raise ValueError("Conditional Ramsey contrast is below the noise floor.")
        parameters.append(
            [
                float(offset),
                float(cosine),
                float(sine),
                float(frequency / span * 1e9),
                float(decay / span),
                float(waits[0]),
            ]
        )
    covariance = np.linalg.inv(best.jac.T @ best.jac) * noise**2
    if np.any(np.sqrt(np.maximum(0, np.diag(covariance)[:2])) > 0.5 * best.x[:2]):
        raise ValueError("Conditional frequencies are not statistically resolved.")
    difference = np.array([-1.0, 1.0, 0.0])
    error = np.sqrt(max(0, difference @ covariance @ difference)) / span * 1e9
    return float((f1 - f0) / span * 1e9), float(error), parameters


def _zero_crossing(amplitudes: npt.NDArray, zz: npt.NDArray) -> float | None:
    """Linearly interpolate the first zero crossing of ZZ versus pulse amplitude."""
    signs = np.sign(zz)
    for i in range(len(zz) - 1):
        if not np.isfinite(zz[i]):
            continue
        if signs[i] == 0:
            return float(amplitudes[i])
        if signs[i] * signs[i + 1] < 0:
            x0, x1 = amplitudes[i], amplitudes[i + 1]
            y0, y1 = zz[i], zz[i + 1]
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    if len(signs) and signs[-1] == 0:
        return float(amplitudes[-1])
    return None


def _minimum(
    amplitudes: npt.NDArray, zz: npt.NDArray, errors: npt.NDArray
) -> tuple[float, float, list[float], list[float]]:
    """Minimize a local ZZ polynomial without extrapolating or bridging failed fits."""
    index = int(np.nanargmin(np.abs(zz)))
    left = right = index
    while left > 0 and np.isfinite(zz[left - 1]) and index - left < 2:
        left -= 1
    while right + 1 < len(zz) and np.isfinite(zz[right + 1]) and right - index < 2:
        right += 1
    x, y = amplitudes[left : right + 1], zz[left : right + 1]
    if len(x) < 3:
        crossing = _zero_crossing(x, y)
        if crossing is not None:
            return crossing, 0.0, [], []
        return float(amplitudes[index]), float(zz[index]), [], []
    uncertainty = errors[left : right + 1]
    weights = None
    if np.all(np.isfinite(uncertainty)) and np.all(uncertainty > 0):
        weights = 1 / np.maximum(uncertainty, np.max(uncertainty) * 1e-3)
    polynomial = np.polynomial.Polynomial.fit(x, y, 2, w=weights).convert()
    candidates = [x[0], x[-1]]
    for root in [*polynomial.roots(), *polynomial.deriv().roots()]:
        if np.isreal(root) and x[0] <= np.real(root) <= x[-1]:
            candidates.append(float(np.real(root)))
    point = min(candidates, key=lambda value: abs(polynomial(value)))
    # Never replace a better measured point with a worse polynomial prediction.
    if abs(polynomial(point)) > abs(zz[index]):
        return float(amplitudes[index]), float(zz[index]), [], []
    return (
        float(point),
        float(polynomial(point)),
        polynomial.coef.tolist(),
        [float(x[0]), float(x[-1])],
    )


def _fit(data: CouplerZZData) -> CouplerZZResults:
    """Extract physical ZZ from both conditional fringes and minimize its magnitude."""

    results = CouplerZZResults()

    for pair in data.pairs:
        amplitudes = data.amplitudes(pair)
        zz = np.full(len(amplitudes), np.nan)
        errors = np.full(len(amplitudes), np.nan)
        parameters = {setup: [[] for _ in amplitudes] for setup in SETUPS}
        if (
            data.detuning is None
            or not np.isfinite(data.detuning)
            or data.detuning == 0
        ):
            log.warning(
                f"ZZ extraction failed for pair {pair}: nonzero detuning required."
            )
            continue
        scale = data.frequency_scale

        for i, amplitude in enumerate(amplitudes):
            try:
                traces = []
                times = []
                for setup in SETUPS:
                    setup_data = data[pair, setup]
                    trace = setup_data[setup_data["amplitude"] == amplitude]
                    trace = np.sort(trace, order="wait")
                    times.append(trace["wait"])
                    traces.append(trace["prob"])
                if not np.array_equal(times[0], times[1]):
                    raise ValueError("Control-state traces have different wait grids.")
                difference, error, fitted = _conditional_fit(
                    times[0], np.asarray(traces), data.detuning
                )
                zz[i], errors[i] = scale * difference, abs(scale) * error
                for setup, values in zip(SETUPS, fitted):
                    parameters[setup][i] = values
            except (ValueError, RuntimeError, KeyError, np.linalg.LinAlgError) as e:
                log.warning(
                    f"ZZ fitting failed for pair {pair}, "
                    f"coupler pulse amplitude {amplitude:.6f}: {e}."
                )

        if np.all(np.isnan(zz)):
            log.warning(f"ZZ extraction failed for pair {pair}: no valid fits.")
            continue

        results.amplitudes[pair] = amplitudes.tolist()
        results.zz[pair] = zz.tolist()
        results.zz_error[pair] = errors.tolist()
        results.fitted_parameters[pair] = parameters
        point, minimum, polynomial, interval = _minimum(amplitudes, zz, errors)
        results.zz_free_amplitude[pair] = point
        results.minimum_zz[pair] = minimum
        results.polynomial[pair] = polynomial
        results.fit_amplitudes[pair] = interval
        valid = np.isfinite(zz)
        closest = int(np.argmin(np.abs(amplitudes[valid] - point)))
        results.residual_zz[pair] = float(zz[valid][closest])
        if point in (amplitudes[0], amplitudes[-1]):
            log.warning(
                f"ZZ minimum for pair {pair} is at the sweep boundary; "
                "extend the amplitude range before treating it as a null."
            )
    return results


def _plot(
    data: CouplerZZData,
    target: QubitPairId,
    fit: CouplerZZResults | None = None,
):
    """Heatmaps of the conditional Ramsey fringes and ZZ versus pulse amplitude."""

    figures = []
    waits = data.waits(target)
    amplitudes = data.amplitudes(target)

    heatmap_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Control in |0⟩",
            "Control in |1⟩",
        ),
        shared_yaxes=True,
    )

    for col, setup in enumerate(SETUPS, start=1):
        setup_data = data.data.get((target, setup), [])
        z = np.full((len(amplitudes), len(waits)), np.nan)
        for row in setup_data:
            z[
                np.searchsorted(amplitudes, row["amplitude"]),
                np.searchsorted(waits, row["wait"]),
            ] = row["prob"]
        heatmap_fig.add_trace(
            go.Heatmap(
                x=waits,
                y=amplitudes,
                z=z,
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                showscale=(col == 2),
                colorbar={
                    "title": {"text": "Excited State Probability", "side": "right"}
                },
            ),
            row=1,
            col=col,
        )
        heatmap_fig.update_xaxes(title_text="Wait time [ns]", row=1, col=col)

    heatmap_fig.update_yaxes(title_text="Coupler pulse amplitude [a.u.]", row=1, col=1)
    figures.append(heatmap_fig)

    fitting_report = ""
    if fit is not None and target in fit.zz:
        zz_fig = go.Figure(
            go.Scatter(
                x=fit.amplitudes[target],
                y=np.array(fit.zz[target]) * HZ_TO_KHZ,
                mode="lines+markers",
                name="ZZ",
                error_y={"array": np.asarray(fit.zz_error.get(target, [])) * HZ_TO_KHZ},
            )
        )
        zz_fig.add_hline(y=0, line={"color": "gray", "dash": "dot"})
        if fit.polynomial.get(target):
            x = np.linspace(*fit.fit_amplitudes[target], 200)
            zz_fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.polynomial.polynomial.polyval(x, fit.polynomial[target])
                    * HZ_TO_KHZ,
                    name="Local ZZ fit",
                )
            )

        if target in fit.zz_free_amplitude:
            zz_fig.add_vline(
                x=fit.zz_free_amplitude[target],
                line={"color": "black", "dash": "dash", "width": 2},
                annotation_text=f"Minimum |ZZ|: {fit.zz_free_amplitude[target]:.4f}",
                annotation_position="top right",
            )
            fitting_report = table_html(
                table_dict(
                    str(target),
                    [
                        "Minimum-|ZZ| coupler pulse amplitude [a.u.]",
                        "Fitted minimum ZZ [Hz]",
                        "Residual ZZ at closest amplitude [Hz]",
                    ],
                    [
                        np.round(fit.zz_free_amplitude[target], 6),
                        np.round(fit.minimum_zz.get(target, np.nan), 1),
                        np.round(fit.residual_zz[target], 1),
                    ],
                )
            )

        zz_fig.update_layout(
            xaxis_title="Coupler pulse amplitude [a.u.]",
            yaxis_title="ZZ coupling [kHz]",
            showlegend=True,
        )
        figures.append(zz_fig)

    return figures, fitting_report


coupler_zz = Protocol(_acquisition, _fit, _plot, two_qubit_gates=True)
"""CouplerZZ Protocol object."""
