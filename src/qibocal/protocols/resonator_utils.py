import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import leastsq, newton


def get_cable_delay(frequencies: NDArray, phases: NDArray, num_points: int) -> float:
    """Evaluates the cable delay :math:`\tau`.

     This delay is caused by the length of the cable and the finite speed of light.
    Performs a first-grade polynomial fit of the phase and extracts the angular coefficient.

        Args:
            frequencies (NDArray[float]): frequencies (Hz) at which the measurement was taken.
            phases (NDArray[complex]): Phase of the S21 scattering matrix element.
            num_points (int): number of points selected from both the start and the end of the
                              frequencies array to perform the linear fit.

        Returns:
            The value (float) of the cable delay :math:`\tau` in seconds.
    """
    phases = np.unwrap(phases)
    frequencies_selected = np.concatenate(
        (frequencies[:num_points], frequencies[-num_points:])
    )
    phase_selected = np.concatenate((phases[:num_points], phases[-num_points:]))

    pvals = np.polyfit(frequencies_selected, phase_selected, 1)

    return pvals[0] / (-2 * np.pi)


def remove_cable_delay(frequencies: NDArray, z: NDArray, tau: float) -> NDArray:
    """
    Corrects the S21 scattering matrix element array from the cable delay.

        Args:
            frequencies (NDArray[float]): frequencies (Hz) at which the measurement was taken.
            z (NDArray[complex]): S21 scattering matrix element.
            tau (float): the cable delay τ in seconds.
        Returns:
            The corrected S21 scattering matrix element (NDArray[complex]).
    """
    return z * np.exp(2j * np.pi * frequencies * tau)


def circle_fit(z: NDArray) -> tuple[float, float, float]:
    """
    Fits the circle of an S21 scattering matrix element array using the algebraic fit described in
    "Efficient and robust analysis of complex scattering data under noise in microwave resonators"
    (https://doi.org/10.1063/1.4907935) by S. Probst et al.

        Args:
            z (NDArray[complex]): S21 scattering matrix element.
            tau (float): the cable delay τ in seconds
        Returns:
            (tuple[float, float, float]): the (x,y) coordinates of the circle's center and
            the radius of the circle.
    """
    z = z.copy()
    x_norm = 0.5 * (np.max(z.real) + np.min(z.real))
    y_norm = 0.5 * (np.max(z.imag) + np.min(z.imag))
    z -= x_norm + 1j * y_norm

    amplitude_norm = np.max(np.abs(z))
    z /= amplitude_norm

    x_i = z.real
    x_i2 = x_i**2

    y_i = z.imag
    y_i2 = y_i**2

    z_i = x_i2 + y_i2
    z_i2 = z_i**2

    n = len(x_i)

    x_i_sum = np.sum(x_i)
    y_i_sum = np.sum(y_i)
    z_i_sum = np.sum(z_i)
    x_i_y_i_sum = np.sum(x_i * y_i)
    x_i_z_i_sum = np.sum(x_i * z_i)
    y_i_z_i_sum = np.sum(y_i * z_i)
    z_i2_sum = np.sum(z_i2)
    x_i2_sum = np.sum(x_i2)
    y_i2_sum = np.sum(y_i2)

    m_matrix = np.array(
        [
            [z_i2_sum, x_i_z_i_sum, y_i_z_i_sum, z_i_sum],
            [x_i_z_i_sum, x_i2_sum, x_i_y_i_sum, x_i_sum],
            [y_i_z_i_sum, x_i_y_i_sum, y_i2_sum, y_i_sum],
            [z_i_sum, x_i_sum, y_i_sum, n],
        ]
    )

    a_0 = (
        (
            (m_matrix[2][0] * m_matrix[3][2] - m_matrix[2][2] * m_matrix[3][0])
            * m_matrix[1][1]
            - m_matrix[1][2] * m_matrix[2][0] * m_matrix[3][1]
            - m_matrix[1][0] * m_matrix[2][1] * m_matrix[3][2]
            + m_matrix[1][0] * m_matrix[2][2] * m_matrix[3][1]
            + m_matrix[1][2] * m_matrix[2][1] * m_matrix[3][0]
        )
        * m_matrix[0][3]
        + (
            m_matrix[0][2] * m_matrix[2][3] * m_matrix[3][0]
            - m_matrix[0][2] * m_matrix[2][0] * m_matrix[3][3]
            + m_matrix[0][0] * m_matrix[2][2] * m_matrix[3][3]
            - m_matrix[0][0] * m_matrix[2][3] * m_matrix[3][2]
        )
        * m_matrix[1][1]
        + (
            m_matrix[0][1] * m_matrix[1][3] * m_matrix[3][0]
            - m_matrix[0][1] * m_matrix[1][0] * m_matrix[3][3]
            - m_matrix[0][0] * m_matrix[1][3] * m_matrix[3][1]
        )
        * m_matrix[2][2]
        + (
            -m_matrix[0][1] * m_matrix[1][2] * m_matrix[2][3]
            - m_matrix[0][2] * m_matrix[1][3] * m_matrix[2][1]
        )
        * m_matrix[3][0]
        + (
            (m_matrix[2][3] * m_matrix[3][1] - m_matrix[2][1] * m_matrix[3][3])
            * m_matrix[1][2]
            + m_matrix[2][1] * m_matrix[3][2] * m_matrix[1][3]
        )
        * m_matrix[0][0]
        + (
            m_matrix[1][0] * m_matrix[2][3] * m_matrix[3][2]
            + m_matrix[2][0]
            * (m_matrix[1][2] * m_matrix[3][3] - m_matrix[1][3] * m_matrix[3][2])
        )
        * m_matrix[0][1]
        + (
            (m_matrix[2][1] * m_matrix[3][3] - m_matrix[2][3] * m_matrix[3][1])
            * m_matrix[1][0]
            + m_matrix[1][3] * m_matrix[2][0] * m_matrix[3][1]
        )
        * m_matrix[0][2]
    )
    a_1 = (
        (
            (m_matrix[3][0] - 2.0 * m_matrix[2][2]) * m_matrix[1][1]
            - m_matrix[1][0] * m_matrix[3][1]
            + m_matrix[2][2] * m_matrix[3][0]
            + 2.0 * m_matrix[1][2] * m_matrix[2][1]
            - m_matrix[2][0] * m_matrix[3][2]
        )
        * m_matrix[0][3]
        + (
            2.0 * m_matrix[2][0] * m_matrix[3][2]
            - m_matrix[0][0] * m_matrix[3][3]
            - 2.0 * m_matrix[2][2] * m_matrix[3][0]
            + 2.0 * m_matrix[0][2] * m_matrix[2][3]
        )
        * m_matrix[1][1]
        + (
            -m_matrix[0][0] * m_matrix[3][3]
            + 2.0 * m_matrix[0][1] * m_matrix[1][3]
            + 2.0 * m_matrix[1][0] * m_matrix[3][1]
        )
        * m_matrix[2][2]
        + (
            -m_matrix[0][1] * m_matrix[1][3]
            + 2.0 * m_matrix[1][2] * m_matrix[2][1]
            - m_matrix[0][2] * m_matrix[2][3]
        )
        * m_matrix[3][0]
        + (m_matrix[1][3] * m_matrix[3][1] + m_matrix[2][3] * m_matrix[3][2])
        * m_matrix[0][0]
        + (m_matrix[1][0] * m_matrix[3][3] - 2.0 * m_matrix[1][2] * m_matrix[2][3])
        * m_matrix[0][1]
        + (m_matrix[2][0] * m_matrix[3][3] - 2.0 * m_matrix[1][3] * m_matrix[2][1])
        * m_matrix[0][2]
        - 2.0 * m_matrix[1][2] * m_matrix[2][0] * m_matrix[3][1]
        - 2.0 * m_matrix[1][0] * m_matrix[2][1] * m_matrix[3][2]
    )
    a_2 = (
        (2.0 * m_matrix[1][1] - m_matrix[3][0] + 2.0 * m_matrix[2][2]) * m_matrix[0][3]
        + (2.0 * m_matrix[3][0] - 4.0 * m_matrix[2][2]) * m_matrix[1][1]
        - 2.0 * m_matrix[2][0] * m_matrix[3][2]
        + 2.0 * m_matrix[2][2] * m_matrix[3][0]
        + m_matrix[0][0] * m_matrix[3][3]
        + 4.0 * m_matrix[1][2] * m_matrix[2][1]
        - 2.0 * m_matrix[0][1] * m_matrix[1][3]
        - 2.0 * m_matrix[1][0] * m_matrix[3][1]
        - 2.0 * m_matrix[0][2] * m_matrix[2][3]
    )
    a_3 = (
        -2.0 * m_matrix[3][0]
        + 4.0 * m_matrix[1][1]
        + 4.0 * m_matrix[2][2]
        - 2.0 * m_matrix[0][3]
    )
    a_4 = -4

    def pol_func(x: float) -> float:
        """
        Polynomio for find a root of a real or complex function using the Newton-Raphson method.

            Args:
                x (float): independent variable.
            Returns:
                Evaluated polynomial (float).
        """
        return a_0 + a_1 * x + a_2 * x**2 + a_3 * x**3 + a_4 * x**4

    def der_pol_func(x: float) -> float:
        """
        Derivative of the polynomio for find a root of a real or complex function using the
        Newton-Raphson method.

            Args:
                x (float): independent variable.
            Returns:
                Evaluated derivative of the polynomial (float).
        """
        return a_1 + 2 * a_2 * x + 3 * a_3 * x**2 + 4 * a_4 * x**3

    eta = newton(pol_func, 0.0, fprime=der_pol_func)

    m_matrix[3][0] = m_matrix[3][0] + 2 * eta
    m_matrix[0][3] = m_matrix[0][3] + 2 * eta
    m_matrix[1][1] = m_matrix[1][1] - eta
    m_matrix[2][2] = m_matrix[2][2] - eta

    _, s, vt = np.linalg.svd(m_matrix)
    a_vec = vt[np.argmin(s), :]

    x_c = -a_vec[1] / (2.0 * a_vec[0])
    y_c = -a_vec[2] / (2.0 * a_vec[0])

    r_0 = (
        1.0
        / (2.0 * np.absolute(a_vec[0]))
        * np.sqrt(a_vec[1] * a_vec[1] + a_vec[2] * a_vec[2] - 4.0 * a_vec[0] * a_vec[3])
    )

    return (
        x_c * amplitude_norm + x_norm,
        y_c * amplitude_norm + y_norm,
        r_0 * amplitude_norm,
    )


def phase_fit(frequencies: NDArray, z: NDArray) -> NDArray:
    """
    Fits the phase response of a resonator.

        Args:
            frequencies (NDArray[float]): frequencies (Hz) at which the measurement was taken.
            z (NDArray[complex]): S21 scattering matrix element.

        Returns:
            Resonance frequency, loaded quality factor, offset phase and time delay between output
            and input signal leading to linearly frequency dependent phase shift (NDArray[float]).
    """
    phase = np.unwrap(np.angle(z))

    if np.max(phase) - np.min(phase) <= 0.8 * 2 * np.pi:
        roll_off = np.max(phase) - np.min(phase)
    else:
        roll_off = 2 * np.pi

    phase_smooth = gaussian_filter1d(phase, 30)
    phase_derivative = np.gradient(phase_smooth)
    resonance_guess = frequencies[np.argmax(np.abs(phase_derivative))]
    q_loaded_guess = 2 * resonance_guess / (frequencies[-1] - frequencies[0])
    slope = phase[-1] - phase[0] + roll_off
    tau_guess = -slope / (2 * np.pi * (frequencies[-1] - frequencies[0]))
    theta_guess = 0.5 * (np.mean(phase[:5]) + np.mean(phase[-5:]))

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
        return phase_dist(phase - phase_centered(frequencies, *params))

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

    return p_final[0]


def phase_dist(angle: float) -> float:
    """
    Maps angle [-2pi, 2pi] to phase distance on circle [0, pi].

        Args:
            angle (float): angle to be mapped.
        Returns:
            Mapped angle (float).
    """
    return np.pi - np.abs(np.pi - np.abs(angle))


def phase_centered(
    frequencies: NDArray,
    resonance: float,
    q_loaded: float,
    theta: float,
    tau: float = 0.0,
) -> NDArray:
    """
    Evaluates the phase response of a resonator which corresponds to a circle centered around
    the origin. Additionally, a linear background slope is accounted for if needed.

        Args:
            frequencies (NDArray[float]): frequencies (Hz) at which the measurement was taken.
            resonance (float): resonance frequency.
            q_loaded (float): loaded quality factor.
            theta (float): offset phase.
            tau (float): time delay between output and input signal leading to linearly frequency
                         dependent phase shift.
        Returns:
            Phase centered array (NDArray[float]).
    """
    return (
        theta
        - 2 * np.pi * tau * (frequencies - resonance)
        + 2.0 * np.arctan(2.0 * q_loaded * (1.0 - frequencies / resonance))
    )


def periodic_boundary(angle: float) -> float:
    """
    Maps arbitrary angle to interval [-np.pi, np.pi).

    Args:
        angle (float): angle to be mapped.
    Returns:
        Mapped angle (float).
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
