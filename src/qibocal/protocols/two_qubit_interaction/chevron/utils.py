from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from qibolab import Delay, PulseSequence, VirtualZ
from scipy.ndimage import uniform_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from qibocal.auto.operation import QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import minmax_scaling
from qibocal.update import replace

COUPLER_PULSE_START = 0
"""Start of coupler pulse."""
COUPLER_PULSE_DURATION = 100
"""Duration of coupler pulse."""


def chevron_sequence(
    platform: CalibrationPlatform,
    ordered_pair: QubitPairId,
    duration_max: int | None = None,
    parking: bool = False,
    native: str = "CZ",
    dt: int = 0,
):
    """Chevron pulse sequence."""
    sequence = PulseSequence()
    low_natives = platform.natives.single_qubit[ordered_pair[0]]
    high_natives = platform.natives.single_qubit[ordered_pair[1]]
    if native == "CZ":
        sequence += low_natives.RX()
    sequence += high_natives.RX()

    drive_duration = sequence.duration
    raw_flux_sequence = getattr(platform.natives.two_qubit[ordered_pair], native)()
    flux_channel, flux_pulse = next(
        (ch, pulse)
        for ch, pulse in raw_flux_sequence
        if ch == platform.qubits[ordered_pair[1]].flux
    )

    if duration_max is not None:
        flux_pulse = replace(flux_pulse, duration=duration_max)

    sequence.append((flux_channel, Delay(duration=drive_duration)))
    sequence.append((flux_channel, flux_pulse))

    parking_pulses = []
    if parking:
        for ch, pulse in raw_flux_sequence:
            if not isinstance(pulse, VirtualZ) and ch != flux_channel:
                sequence.append((ch, Delay(duration=drive_duration)))
                sequence.append((ch, pulse))
                parking_pulses.append(pulse)

    flux_duration = max(flux_pulse.duration, raw_flux_sequence.duration)

    ro_low_channel, ro_high_channel = (
        platform.qubits[ordered_pair[0]].acquisition,
        platform.qubits[ordered_pair[1]].acquisition,
    )
    ro_low_delay = ro_high_delay = drive_delay = Delay(duration=flux_duration)
    dt_delay = Delay(duration=dt)
    drive_channel, second_rx = high_natives.RX()[0]
    sequence += [
        (ro_low_channel, Delay(duration=drive_duration)),
        (ro_high_channel, Delay(duration=drive_duration)),
        (ro_low_channel, ro_low_delay),
        (ro_high_channel, ro_high_delay),
        (ro_low_channel, dt_delay),
        (ro_high_channel, dt_delay),
        (drive_channel, drive_delay),
        (drive_channel, dt_delay),
    ]

    if native == "CZ":
        sequence += [
            (ro_low_channel, Delay(duration=second_rx.duration)),
            (ro_high_channel, Delay(duration=second_rx.duration)),
            (drive_channel, second_rx),
        ]

    # add readout
    sequence += low_natives.MZ() + high_natives.MZ()

    return (
        sequence,
        flux_pulse,
        parking_pulses,
        [ro_low_delay, ro_high_delay, drive_delay],
    )


SNR_THRESHOLD = 2.0
"""Minimum readout SNR to contribute to the fit."""
FIT_FAILED = "no chevron found."
"""Error message for every failed check, since the plot shows the cause better."""


def cosine(
    t: NDArray[np.floating], offset: float, amplitude: float, omega: float, t0: float
) -> NDArray[np.floating]:
    """Population along the resonance column."""
    return offset + amplitude * np.cos(omega * (t - t0))


def readout_snr(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Contrast-to-noise ratio of each qubit's readout.

    ``data`` has shape ``(n_qubits, n_duration, n_amplitude)``.
    """
    snr = []
    for qubit_data in data:
        n_duration = len(qubit_data)
        smoothed = (qubit_data[:-2] + qubit_data[1:-1] + qubit_data[2:]) / 3
        contrast = (smoothed.max(axis=0) - smoothed.min(axis=0)).max()
        # x[i+1] - 2x[i] + x[i-1] cancels the locally linear signal. Its noise variance
        # is (1^2 + (-2)^2 + 1^2) sigma^2 = 6 sigma^2.
        second_diff = np.diff(qubit_data, n=2, axis=0)
        # Median is robust to outlier columns.
        noise = np.median(np.std(second_diff, axis=0)) / np.sqrt(6)
        # Normalised by the expected range of N Gaussian samples, 2 sigma sqrt(2 ln N).
        snr.append(
            contrast / (2 * noise * np.sqrt(2 * np.log(n_duration)))
            if noise > 0
            else 0.0
        )
    return np.array(snr)


def find_population_exchanges(
    data: NDArray[np.floating], duration: NDArray[np.floating]
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Locate the population exchanges of the first oscillation in each amplitude
    column.

    Gives rough, grid-level estimates used to find the resonance and seed the
    cosine fit. Per column, it returns the duration index of the first full exchange
    and of the following return, or -1 where absent or outside the first oscillation.
    """
    n_amplitude = data.shape[1]
    # Drop from the background to the deepest exchange, the scale for all thresholds.
    depth = np.median(data) - data.min()
    threshold = data.min() + 0.3 * depth

    exchange = np.full(n_amplitude, -1)
    revival = np.full(n_amplitude, -1)
    for amp_idx in range(n_amplitude):
        amplitude_data = data[:, amp_idx]
        # The exchange is the first minimum within 30 % of the depth from the bottom.
        minima, _ = find_peaks(-amplitude_data, height=-threshold)
        if len(minima) == 0:
            continue
        exchange_idx = minima[0]
        exchange[amp_idx] = exchange_idx
        # The revival is the next maximum, if it climbs back by 30 % of the depth.
        maxima, _ = find_peaks(amplitude_data)
        maxima = maxima[maxima > exchange_idx]
        if len(maxima):
            revival_idx = maxima[0]
            if (
                amplitude_data[revival_idx] - amplitude_data[exchange_idx]
                >= 0.3 * depth
            ):
                revival[amp_idx] = revival_idx

    # Exchanges later than 1.5x the earliest belong to a later oscillation.
    if (exchange >= 0).any():
        earliest = duration[exchange[exchange >= 0]].min()
        later = (exchange >= 0) & (duration[np.maximum(exchange, 0)] > 1.5 * earliest)
        exchange[later] = revival[later] = -1
    return exchange, revival


def clusters(columns: NDArray[np.intp], max_gap: int) -> list[NDArray[np.intp]]:
    """Split sorted column indices at gaps larger than ``max_gap``."""
    return np.split(columns, np.flatnonzero(np.diff(columns) > max_gap) + 1)


def has_second_crossing(mask: NDArray[np.bool_], resonance_idx: int) -> bool:
    """Whether the exchanging columns in ``mask`` form a cluster away from the
    resonance column."""
    columns = np.flatnonzero(mask)
    if len(columns) == 0:
        return False

    def touches(cluster: NDArray[np.intp]) -> bool:
        return cluster[0] - 1 <= resonance_idx <= cluster[-1] + 1

    # Gaps narrower than the resonance cluster are noise.
    width = max((np.ptp(c) + 1 for c in clusters(columns, 3) if touches(c)), default=0)
    return any(not touches(c) for c in clusters(columns, max(3, width)))


def fit_gate_parameters(
    data: NDArray[np.floating],
    amplitude: Sequence[float] | NDArray[np.floating],
    duration: Sequence[float] | NDArray[np.floating],
    native: str,
    signal: bool = False,
) -> tuple[float, float]:
    """Gate amplitude and duration from the first oscillation of the chevron.

    With ``signal``, ``data`` is readout magnitude, whose direction is unknown, and each
    readout is oriented from the data instead.

    Returns:
        ``(amplitude0, duration0)``, the resonance amplitude and the gate duration [ns].
        For CZ this is the first return of the population, for iSWAP the first full
        exchange.
    """
    amplitude, duration = np.asarray(amplitude, float), np.asarray(duration, float)
    n_duration, n_amplitude = len(duration), len(amplitude)
    step = duration[1] - duration[0]
    assert np.allclose(np.diff(duration), step)

    # 1. combine the two chevrons
    # Oriented so the exchange is a minimum. For iSWAP the low qubit starts at 0 and is
    # inverted.
    scaled_data = minmax_scaling(data, axis=(1, 2))
    if native == "iSWAP":
        scaled_data[0] = 1 - scaled_data[0]
    if signal:
        # Whether |1> reads higher or lower depends on the readout point. Most of the
        # scan is off resonance, so the median is the background, which must be high.
        flip = np.median(scaled_data, axis=(1, 2)) < 0.5
        scaled_data[flip] = 1 - scaled_data[flip]
    snr = readout_snr(scaled_data)
    # Chevrons measurements with low SNR get zero weight in the combination.
    weights = np.where(snr >= SNR_THRESHOLD, snr, 0.0)
    if not weights.any():
        raise ValueError(FIT_FAILED)
    smooth = [uniform_filter(y, size=3, mode="nearest") for y in scaled_data]
    combined = np.average(smooth, axis=0, weights=weights)

    # 2. locate the resonance amplitude
    exchange_idx_guess, revival_idx_guess = find_population_exchanges(
        combined, duration
    )
    # Lowest at resonance, where the swap is most complete. Inf without a revival.
    population_at_exchange = np.where(
        revival_idx_guess >= 0,
        combined[np.maximum(exchange_idx_guess, 0), np.arange(n_amplitude)],
        np.inf,
    )
    # The deepest column is noisy, so the resonance is the centre of the contiguous
    # columns within 5 % of its depth, assuming a symmetric chevron.
    deepest_idx = int(np.argmin(population_at_exchange))
    depth = np.median(combined) - combined.min()
    depth_limit = population_at_exchange[deepest_idx] + 0.05 * depth
    near_deepest = population_at_exchange <= depth_limit
    # The cumsum is constant where near_deepest is True
    runs = np.cumsum(~near_deepest)
    resonance_band = np.flatnonzero(near_deepest & (runs == runs[deepest_idx]))
    resonance_amplitude = float(amplitude[resonance_band].mean())
    resonance_amplitude_idx = int(np.abs(amplitude - resonance_amplitude).argmin())

    # 3. reject an unclear chevron
    # A resonance band touching the edge of the amplitude window may extend beyond it,
    # so its centre is unreliable.
    if resonance_band[0] == 0 or resonance_band[-1] == n_amplitude - 1:
        raise ValueError(FIT_FAILED)
    # A second crossing is another group of exchanging columns. Readouts are also
    # checked individually, since the combination can dilute it.
    masks = [exchange_idx_guess >= 0] + [
        find_population_exchanges(s, duration)[0] >= 0
        for s, w in zip(smooth, weights)
        if w
    ]
    if any(has_second_crossing(mask, resonance_amplitude_idx) for mask in masks):
        raise ValueError(FIT_FAILED)

    # 4. fit a cosine on the resonance column
    # Over the first oscillation, seeded with the exchange at half a period.
    resonance_exchange_idx = int(exchange_idx_guess[resonance_amplitude_idx])
    resonance_revival_idx = int(revival_idx_guess[resonance_amplitude_idx])
    exchange_time = float(duration[resonance_exchange_idx])
    half_oscillation = (resonance_revival_idx - resonance_exchange_idx) // 2
    window = slice(0, min(n_duration, resonance_revival_idx + half_oscillation + 2))
    trace = np.average(
        scaled_data[:, window, resonance_amplitude_idx], axis=0, weights=weights
    )
    t = duration[window]
    omega0 = np.pi / max(exchange_time, step)
    lower = [0, 0, np.pi / np.ptp(duration), -0.25 * np.pi / omega0]
    upper = [1, 1, np.pi / (2 * step), 0.25 * np.pi / omega0]
    p0 = np.clip(np.array([trace.mean(), np.ptp(trace) / 2, omega0, 0.0]), lower, upper)
    result = least_squares(
        lambda p: cosine(t, *p) - trace, p0, bounds=(lower, upper), max_nfev=2000
    )
    params = result.x.tolist()
    period, t0 = 2 * np.pi / params[2], params[3]

    # 5. gate duration
    # CZ lasts until the return (full period), iSWAP until the exchange (half).
    periods = 1 if native == "CZ" else 0.5
    return resonance_amplitude, t0 + periods * period
