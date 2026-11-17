import numpy as np
import plotly.graph_objects as go
import scipy.constants
from qibolab import Platform
from scipy.optimize import curve_fit

from qibocal.auto.operation import QubitId, QubitPairId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from ..utils import fallback_period, guess_period


def order_pair(pair: QubitPairId, platform: Platform) -> tuple[QubitId, QubitId]:
    """Order a pair of qubits by drive frequency."""
    q0, q1 = pair

    drive0 = platform.config(platform.qubits[q0].drive)
    drive1 = platform.config(platform.qubits[q1].drive)
    return (q1, q0) if drive0.frequency > drive1.frequency else (q0, q1)


def determine_cz_hybridization_state(
    calibration_platform: CalibrationPlatform, pair: QubitPairId
) -> tuple[dict[int, QubitId], list[float]]:
    """Determine which qubit occupies each level in the CZ hybridized states.

    Uses calibrated frequencies, anharmonicities, and flux coefficients to
    find the flux roots of the 11–02 and 11–20 resonances. The nearest roots
    to zero are compared to infer which qubit occupies levels 0 and 2.
    This function returns a tuple containing the mapping from level (0 or 2) to qubit
    identifier and the flux values where resonance occurs.
    """

    calibration = calibration_platform.calibration
    if calibration is None:
        raise ValueError("Calibration data not available.")
    # pairs may be stored in either order, independently of the frequency ordering
    if (
        pair not in calibration.two_qubits
        and (pair[1], pair[0]) not in calibration.two_qubits
    ):
        raise ValueError(f"Qubit pair {pair} not found in the two-qubit system.")

    low_qubit, high_qubit = order_pair(pair, calibration_platform)

    low = calibration.single_qubits[low_qubit].qubit
    high = calibration.single_qubits[high_qubit].qubit

    high_qubit_flux = high.flux_coefficients
    if not high_qubit_flux or len(high_qubit_flux) != 3:
        raise ValueError(
            f"Qubit {high_qubit} must be flux tunable, with quadratic flux coefficients."
        )

    low_qubit_f01 = low.frequency_01
    high_qubit_f01 = high.frequency_01
    if not (low_qubit_f01 and high_qubit_f01):
        raise ValueError("Qubits must have frequency 01 defined.")

    # anharmonicities are derived from frequency_01, hence checked afterwards
    low_qubit_eta = low.anharmonicity
    high_qubit_eta = high.anharmonicity
    if not (low_qubit_eta and high_qubit_eta):
        raise ValueError("Qubits must have anharmonicity defined.")

    # in this convention the first qubit is always the low frequency qubit.
    # The high frequency qubit follows f_high(x) = f01_high + a x^2 + b x + c,
    # with all frequencies in Hz, so that
    #   E_11(x) = f_high(x) + f01_low
    #   E_02(x) = 2 f_high(x) + eta_high
    #   E_20    = 2 f01_low + eta_low
    # Both resonance conditions reduce to f_high(x) = target, i.e. to a
    # quadratic in x that is solved in closed form.
    # Flux coefficients are fitted in GHz, hence they are converted to Hz.
    a, b, c = (coeff * scipy.constants.giga for coeff in high_qubit_flux)
    offset = c + high_qubit_f01
    # E_11 = E_02  <=>  f_high(x) = f01_low - eta_high
    flux_resonance_11_02 = solve_quadratic(
        a, b, offset - (low_qubit_f01 - high_qubit_eta)
    )
    # E_11 = E_20  <=>  f_high(x) = f01_low + eta_low
    flux_resonance_11_20 = solve_quadratic(
        a, b, offset - (low_qubit_f01 + low_qubit_eta)
    )

    # If Omega(high)_12 < Omega(low)_01, the 11 and 02 levels never cross
    # as flux is varied. In that case there is no real 11-02 resonance point
    # to compare with the 11-20 crossing, which always occurs.
    # In fact, since Omega(20) = 2 * Omega(low)_01 + anharm(low)
    # and Omega(11) = Omega(low)_01 + Omega(high)_01, but Omega(high)_12 < Omega(low)_01 so for
    # transmons (negative anharmonicity), then is also true that
    # Omega(high)_01 > Omega(low)_01 + anharm(low).
    if len(flux_resonance_11_02) > 0:
        smallest_11_02 = min(flux_resonance_11_02, key=abs)
        if len(flux_resonance_11_20) == 0 or abs(smallest_11_02) <= abs(
            min(flux_resonance_11_20, key=abs)
        ):
            return {0: low_qubit, 2: high_qubit}, flux_resonance_11_02

    return {0: high_qubit, 2: low_qubit}, flux_resonance_11_20


def plot_cz_state_energies(
    calibration_platform: CalibrationPlatform,
    pair: QubitPairId,
    amplitudes: np.ndarray,
) -> go.Figure:
    """Plot the energies of the 11, 02 and 20 states as a function of flux amplitude.

    Only the high frequency qubit is flux pulsed, following the same model of
    :func:`determine_cz_hybridization_state`. States are labelled as
    |low, high⟩, with qubits ordered by frequency. Energies are shown as frequencies
    (E / h) in GHz, and the 11-02 and 11-20 resonances found in the plotted
    range are marked with vertical lines.
    """
    low_qubit, high_qubit = order_pair(pair, calibration_platform)
    low = calibration_platform.calibration.single_qubits[low_qubit].qubit
    high = calibration_platform.calibration.single_qubits[high_qubit].qubit
    if not high.flux_coefficients or len(high.flux_coefficients) != 3:
        raise ValueError(
            f"Qubit {high_qubit} must be flux tunable, with quadratic flux coefficients."
        )

    # frequencies in Hz, rescaled to GHz only when plotting
    low_f01 = low.frequency_01
    low_eta = low.anharmonicity
    high_f01 = high.frequency_01
    high_eta = high.anharmonicity

    def high_frequency(x):
        # detuning is fitted in GHz, hence it is converted to Hz
        return high_f01 + high.detuning(x) * scipy.constants.giga

    energies = {
        "11": high_frequency(amplitudes) + low_f01,
        "02": 2 * high_frequency(amplitudes) + high_eta,
        "20": np.full_like(amplitudes, 2 * low_f01 + low_eta, dtype=float),
    }

    # E_11 - E_n = 0 reduces to f_high(x) = target, as in the hybridization analysis
    a, b, c = (coeff * scipy.constants.giga for coeff in high.flux_coefficients)
    resonances = {
        "02": solve_quadratic(a, b, c + high_f01 - (low_f01 - high_eta)),
        "20": solve_quadratic(a, b, c + high_f01 - (low_f01 + low_eta)),
    }

    fig = go.Figure()
    for state, energy in energies.items():
        fig.add_trace(
            go.Scatter(
                x=amplitudes,
                y=energy * scipy.constants.nano,
                mode="lines",
                name=f"|{state}⟩",
            )
        )
    for state, roots in resonances.items():
        roots = np.array(
            [r for r in roots if amplitudes.min() <= r <= amplitudes.max()]
        )
        if len(roots) == 0:
            continue
        for root in roots:
            fig.add_vline(x=root, line_dash="dash", line_color="gray")
        fig.add_trace(
            go.Scatter(
                x=roots,
                y=(high_frequency(roots) + low_f01) * scipy.constants.nano,
                mode="markers",
                marker={"size": 10, "symbol": "x"},
                name=f"11-{state} resonance",
                hovertemplate="amplitude: %{x:.4f}<br>E / h: %{y:.4f} GHz",
            )
        )

    fig.update_layout(
        title=f"State energies for qubits (low, high) = ({low_qubit}, {high_qubit})",
        xaxis_title="Flux pulse amplitude [a.u.]",
        yaxis_title="Energy / h [GHz]",
        showlegend=True,
    )
    return fig


def solve_quadratic(a: float, b: float, c: float) -> list[float]:
    """Real roots of ``a x^2 + b x + c = 0``, sorted in ascending order.

    The closed-form solution uses the numerically stable variant of the
    quadratic formula, which avoids cancellation when ``b^2 >> |4 a c|``.
    A vanishing ``a`` falls back to the linear solution. Complex roots, and
    the degenerate case in which every ``x`` is a solution, return an empty list.
    """
    if a == 0:
        return [] if b == 0 else [float(-c / b)]

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return []
    if discriminant == 0:
        return [float(-b / (2 * a))]

    # q cannot vanish here, since it would require b = 0 and discriminant = 0
    q = -0.5 * (b + np.copysign(np.sqrt(discriminant), b))
    return sorted([float(q / a), float(c / q)])


def sinusoid(x, amplitude, offset, phase):
    """Sinusoidal fit function."""
    return np.cos(x + phase) * amplitude + offset


def fit_flux_amplitude(matrix, amps, times):
    """Estimate amplitude for CZ gate.


    Given the pattern of a chevron plot (see for example Fig. 2 here
    https://arxiv.org/pdf/1907.04818.pdf). This function estimates
    the CZ amplitude by finding the amplitude which gives the standard
    deviation, indicating that there are oscillation along the z axis.

    Args:
     matrix (np.ndarray): signal matrix
     amps (np.ndarray): amplitudes swept
     times (np.ndarray): duration swept

    Returns:
     amplitude (float): estimated amplitude
     index (int): amplitude index
     delta (float):  omega for estimated amplitude
    """

    size_amp = len(amps)
    fs = []
    std = []
    for i in range(size_amp):
        y = matrix[i, :]
        period = fallback_period(guess_period(times, y)).tolist()
        fs.append(1 / period)
        std.append(np.std(y))

    amplitude = amps[np.argmax(std)]
    delta = fs[np.argmax(std)]
    index = int(np.where(np.unique(amps) == amplitude)[0][0])
    return amplitude, index, delta


def phase_diff(phase_1, phase_2):
    """Return the phase difference of two sinusoids, normalized in the range [0, 2*pi]."""
    return np.mod(phase_2 - phase_1, 2 * np.pi)


def best_snz_point(angles, leakages, x, y):
    """Return ``(x, y, angle, leakage)`` of the lowest-leakage grid point with an
    angle within 10% of pi, or ``None`` if there is no such point."""
    mask = (angles > np.pi * 0.9) & (angles < np.pi * 1.1)
    if not mask.any():
        return None
    i = leakages[mask].argmin()
    return (
        x[mask][i].item(),
        y[mask][i].item(),
        angles[mask][i].item(),
        leakages[mask][i].item(),
    )


def fit_sinusoid(thetas, data):
    """Fit sinusoid to the given data."""
    pguess = [
        np.max(data) - np.min(data),
        np.mean(data),
        np.pi,
    ]

    popt, _ = curve_fit(
        sinusoid,
        thetas,
        data,
        p0=pguess,
        bounds=(
            (0, -np.max(data), 0),
            (np.max(data), np.max(data), 2 * np.pi),
        ),
    )
    return popt.tolist()


def fit_virtualz(
    data: dict,
    pair: list,
    thetas: list,
    key=None,
):
    fitted_parameters = {}
    angle = {}
    virtual_phase = {}
    leakage = {}
    fitted_param = {}
    if key is None:
        key = pair

    target, control = pair
    for setup in ["I", "X"]:
        target_data = data[target, control, setup][0]
        try:
            params = fit_sinusoid(np.array(thetas), target_data)
            fitted_param[target, control, setup] = params
        except Exception as e:
            log.warning(f"CZ fit failed for pair ({target, control}) due to {e}.")

    for setup in ["I", "X"]:
        # leakage estimate: L = m /2
        # See NZ paper from Di Carlo
        # approximation which does not need qutrits
        # https://arxiv.org/pdf/1903.02492.pdf
        leakage[key] = 0.5 * float(
            np.mean(data[target, control, "X"][1] - data[target, control, "I"][1])
        )

        angle[key] = phase_diff(
            fitted_param[target, control, "X"][2],
            fitted_param[target, control, "I"][2],
        )
        virtual_phase[key] = fitted_param[target, control, "I"][2]
        fitted_parameters[key, setup] = fitted_param[target, control, setup]

    return fitted_parameters, virtual_phase, angle, leakage
