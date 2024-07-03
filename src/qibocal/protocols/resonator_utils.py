import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import leastsq, newton


def resonator_fit(
    frequencies: np.array,
    resonance: float,
    q_loaded: float,
    q_coupling: float,
    phi: float = 0.0,
    amplitude: float = 1.0,
    alpha: float = 0.0,
    tau: float = 0.0,
):
    return (
        amplitude
        * np.exp(1j * alpha)
        * np.exp(-2 * np.pi * 1j * frequencies * tau)
        * (
            1
            - ((q_loaded / (np.abs(q_coupling))) * np.exp(1j * phi))
            / (1 + 2j * q_loaded * (frequencies / resonance - 1))
        )
    )


def get_cable_delay(frequencies: np.array, z: np.array, num_points: int = 10):
    phases = np.unwrap(np.angle(z))

    frequencies_selected = np.concatenate(
        (frequencies[:num_points], frequencies[-num_points:])
    )
    phase_selected = np.concatenate((phases[:num_points], phases[-num_points:]))

    pvals = np.polyfit(frequencies_selected, phase_selected, 1)

    return pvals[0] / (-2 * np.pi)


def remove_cable_delay(frequencies: np.array, z: np.array, tau: float):
    return z * np.exp(2j * np.pi * frequencies * tau)


def circle_fit(z: np.array):
    z = z.deepcopy()
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

    def char_pol(x: np.array):
        return a_0 + a_1 * x + a_2 * x**2 + a_3 * x**3 + a_4 * x**4

    def d_char_pol(x: np.array):
        return a_1 + 2 * a_2 * x + 3 * a_3 * x**2 + 4 * a_4 * x**3

    eta = newton(char_pol, 0.0, fprime=d_char_pol)

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


def phase_fit(frequencies: np.array, z: np.array):
    phase = np.unwrap(np.angle(z))

    # For centered circle roll-off should be close to 2pi. If not warn user.
    if np.max(phase) - np.min(phase) <= 0.8 * 2 * np.pi:
        roll_off = np.max(phase) - np.min(phase)
    else:
        roll_off = 2 * np.pi
    # Use maximum of derivative of phase as guess for fr
    phase_smooth = gaussian_filter1d(phase, 30)
    phase_derivative = np.gradient(phase_smooth)
    resonance_guess = frequencies[np.argmax(np.abs(phase_derivative))]
    q_loaded_guess = 2 * resonance_guess / (frequencies[-1] - frequencies[0])
    slope = phase[-1] - phase[0] + roll_off
    tau_guess = -slope / (2 * np.pi * (frequencies[-1] - frequencies[0]))
    theta_guess = 0.5 * (np.mean(phase[:5]) + np.mean(phase[-5:]))

    # Fit model with less parameters first to improve stability of fit
    def residuals_q_loaded(params):
        q_loaded = params
        return residuals_full((resonance_guess, q_loaded, theta_guess, tau_guess))

    def residuals_resonance_theta(params):
        resonance, theta = params
        return residuals_full((resonance, q_loaded_guess, theta, tau_guess))

    def residuals_tau(params):
        tau = params
        return residuals_full((resonance_guess, q_loaded_guess, theta_guess, tau))

    def residuals_resonance_q_loaded(params):
        resonance, q_loaded = params
        return residuals_full((resonance, q_loaded, theta_guess, tau_guess))

    def residuals_full(params):
        return phase_dist(phase - phase_centered(frequencies, *params))

    p_final = leastsq(residuals_q_loaded, [q_loaded_guess])
    q_loaded_guess = p_final[0]
    p_final = leastsq(residuals_resonance_theta, [resonance_guess, theta_guess])
    resonance_guess, theta_guess = p_final[0]
    p_final = leastsq(residuals_tau, [tau_guess])
    tau_guess = p_final[0]
    p_final = leastsq(residuals_resonance_q_loaded, [resonance_guess, q_loaded_guess])
    resonance_guess, q_loaded_guess = p_final[0]
    p_final = leastsq(
        residuals_full, [resonance_guess, q_loaded_guess, theta_guess, tau_guess]
    )

    return p_final[0]


def phase_dist(phase: np.array):
    return np.pi - np.abs(np.pi - np.abs(phase))


def phase_centered(
    frequencies: np.array,
    resonance: float,
    q_loaded: float,
    theta: float,
    tau: float = 0.0,
):
    return (
        theta
        - 2 * np.pi * tau * (frequencies - resonance)
        + 2.0 * np.arctan(2.0 * q_loaded * (1.0 - frequencies / resonance))
    )


def periodic_boundary(angle: np.array):
    return (angle + np.pi) % (2 * np.pi) - np.pi
