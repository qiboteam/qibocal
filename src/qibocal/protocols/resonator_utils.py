import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import leastsq, minimize

PHASES_THRESHOLD_PERCENTAGE = 80
r"""Threshold percentage to ensure the phase data covers a significant portion of the full 2 :math:\pi circle."""
STD_DEV_GAUSSIAN_KERNEL = 30
"""Standard deviation for the Gaussian kernel."""
PHASE_ELEMENTS = 5
"""Number of values to better guess :math:`\theta` (in rad) in the phase fit function."""


def cable_delay(frequencies: NDArray, phases: NDArray, num_points: int) -> float:
    """Evaluates the cable delay :math:`\tau` (in s).

    The cable delay :math:`\tau` (in s) is caused by the length of the cable and the finite speed of light.
    This is estimated fitting a first-grade polynomial fit of the `phases` (in rad) as a function of the
    `frequencies` (in Hz), and extracting the angular coefficient, which is then expressed
    in seconds.

    The `num_points` is used to select how many points should be fitted, from both the
    start and the end of the frequency range.
    """
    frequencies_selected = np.concatenate(
        (frequencies[:num_points], frequencies[-num_points:])
    )
    phase_selected = np.concatenate((phases[:num_points], phases[-num_points:]))

    pvals = np.polyfit(frequencies_selected, phase_selected, 1)

    return pvals[0] / (-2 * np.pi)


def remove_cable_delay(frequencies: NDArray, z: NDArray, tau: float) -> NDArray:
    """Corrects the cable delay :math:`\tau` (in s).

    The cable delay :math:`\tau` (in s) is removed from the scattering matrix element array `z` by performing
    an exponential product which also depends from the `frequencies` (in Hz).
    """

    return z * np.exp(2j * np.pi * frequencies * tau)


def circle_fit(z: NDArray) -> tuple[complex, float]:
    """Fits the circle of a scattering matrix element array.

    The circle fit exploits the algebraic fit described in
    "Efficient and robust analysis of complex scattering data under noise in microwave resonators"
    (https://doi.org/10.1063/1.4907935) by S. Probst et al and
    "The physics of superconducting microwave resonators"
    (https://doi.org/10.7907/RAT0-VM75) by J. Gao.

    The function, from the scattering matrix element array, evaluates the center coordinates
    `x_c` and `y_c` and the radius of the circle `r_0`.
    """

    z = z.copy()
    x_norm = 0.5 * (np.max(z.real) + np.min(z.real))
    y_norm = 0.5 * (np.max(z.imag) + np.min(z.imag))
    z -= x_norm + 1j * y_norm

    amplitude_norm = np.max(np.abs(z))
    z /= amplitude_norm

    coords = np.stack(
        [np.abs(z) ** 2, z.real, z.imag, np.ones_like(z, dtype=np.float64)]
    )
    m_matrix = np.einsum("in,jn->ij", coords, coords)

    b_matrix = np.array([[0, 0, 0, -2], [0, 1, 0, 0], [0, 0, 1, 0], [-2, 0, 0, 0]])

    coefficients = np.linalg.eigvals(np.linalg.inv(b_matrix).dot(m_matrix))

    eta = np.min(
        np.real(
            [
                coefficient
                for coefficient in coefficients
                if np.isreal(coefficient) and coefficient > 0
            ]
        )
    )

    def f_matrix(a_vector, m_matrix, b_matrix, eta):
        return a_vector.T @ m_matrix @ a_vector - eta * (
            a_vector.T @ b_matrix @ a_vector - 1
        )

    def constraint(a_vector, b_matrix):
        return a_vector.T @ b_matrix @ a_vector - 1

    constraints = [{"type": "eq", "fun": constraint, "args": (b_matrix,)}]

    a_vector = np.ones(4)
    result = minimize(
        f_matrix, a_vector, args=(m_matrix, b_matrix, eta), constraints=constraints
    )
    a_vector = result.x

    x_c = -a_vector[1] / (2 * a_vector[0])
    y_c = -a_vector[2] / (2 * a_vector[0])
    r_0 = 1 / (
        2
        * np.abs(a_vector[0])
        * np.sqrt(
            a_vector[1] * a_vector[1]
            + a_vector[2] * a_vector[2]
            - 4 * a_vector[0] * a_vector[3]
        )
    )

    return (
        complex(x_c * amplitude_norm + x_norm, y_c * amplitude_norm + y_norm),
        r_0 * amplitude_norm,
    )


def phase_fit(frequencies: NDArray, phases: NDArray) -> NDArray:
    r"""Fits the phase response of a resonator.

    The phase fit firstly ensure the phase data (in rad) covers a significant portion of the full 2 :math:`\pi`
    circle evaluating a `roll_off`. If the data do not cover a full circle it is possible to increase
    the frequency span around the resonance. Data are smoothed using a Gaussian filter and the
    derivative is evaluated while initial guesses for the parameters (`resonance_guess` (in Hz)),
    `q_loaded_guess`, `tau_guess` (in s) and `theta_guess` (in rad) are computed with `frequencies` (in Hz).

    The parameter estimation is done through an iterative least squares process to optimize the model
    parameters. The defined functions: `residuals_q_loaded`, `residuals_resonance_theta`
    `residuals_resonance_theta`, `residuals_tau`, `residuals_resonance_q_loaded`, `residuals_full`
    take the parameters to be fitted and return the residuals calculated by subtracting the phase
    centered model from the phase data (in rad).
    """

    if np.max(phases) - np.min(phases) <= PHASES_THRESHOLD_PERCENTAGE / 100 * 2 * np.pi:
        roll_off = np.max(phases) - np.min(phases)
    else:
        roll_off = 2 * np.pi

    phases_smooth = gaussian_filter1d(phases, STD_DEV_GAUSSIAN_KERNEL)
    phases_derivative = np.gradient(phases_smooth)
    resonance_guess = frequencies[np.argmax(np.abs(phases_derivative))]
    q_loaded_guess = 2 * resonance_guess / (frequencies[-1] - frequencies[0])
    slope = phases[-1] - phases[0] + roll_off
    tau_guess = -slope / (2 * np.pi * (frequencies[-1] - frequencies[0]))
    theta_guess = 0.5 * (
        np.mean(phases[:PHASE_ELEMENTS]) + np.mean(phases[-PHASE_ELEMENTS:])
    )

    def residuals_q_loaded(params):
        (q_loaded,) = params
        return residuals_full((resonance_guess, q_loaded, theta_guess, tau_guess))

    def residuals_resonance_theta(params):
        resonance, theta = params
        return residuals_full((resonance, q_loaded_guess, theta, tau_guess))

    def residuals_tau(params):
        (tau,) = params
        return residuals_full((resonance_guess, q_loaded_guess, theta_guess, tau))

    def residuals_resonance_q_loaded(params):
        resonance, q_loaded = params
        return residuals_full((resonance, q_loaded, theta_guess, tau_guess))

    def residuals_full(params):
        return phase_dist(phases - phase_centered(frequencies, *params))

    p_final = leastsq(residuals_q_loaded, [q_loaded_guess])
    (q_loaded_guess,) = p_final[0]
    p_final = leastsq(residuals_resonance_theta, [resonance_guess, theta_guess])
    resonance_guess, theta_guess = p_final[0]
    p_final = leastsq(residuals_tau, [tau_guess])
    (tau_guess,) = p_final[0]
    p_final = leastsq(residuals_resonance_q_loaded, [resonance_guess, q_loaded_guess])
    resonance_guess, q_loaded_guess = p_final[0]
    p_final = leastsq(
        residuals_full, [resonance_guess, q_loaded_guess, theta_guess, tau_guess]
    )

    return p_final[0][:-1]


def phase_dist(phases: NDArray) -> NDArray:
    """Maps `phases` (in rad) [-2pi, 2pi] to phase distance on circle [0, pi]."""
    return np.pi - np.abs(np.pi - np.abs(phases))


def phase_centered(
    frequencies: NDArray,
    resonance: float,
    q_loaded: float,
    theta: float,
    tau: float = 0.0,
) -> NDArray:
    """Evaluates the phase (in rad) response of a resonator.

    The phase centered evaluates the phase angle (in rad) of a circle centered around the origin accounting
    for a phase offset :math:`\theta` (in rad), a linear background slope
    :math: 2\\pi `\tau` (in s) (`frequencies` (in Hz) - `resonance` (in Hz)) (if needed) and a dependency on
    the `q_loaded`.
    """
    return (
        theta
        - 2 * np.pi * tau * (frequencies - resonance)
        + 2.0 * np.arctan(2.0 * q_loaded * (1.0 - frequencies / resonance))
    )


def periodic_boundary(angle: float) -> float:
    """Maps arbitrary `angle` (in rad) to interval [-np.pi, np.pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
