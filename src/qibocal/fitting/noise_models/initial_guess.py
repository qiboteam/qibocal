import numpy as np
from scipy import signal
from scipy.signal import hilbert

"""
Advantages of Direct Interpolation method:
It is a simple and fast way to estimate the full-width half-maximum (FWHM) of a signal.
It does not require any additional transforms of the data beyond the Fourier transform.

Disadvantages of Direct Interpolation method:
It assumes that the signal is symmetric around its maximum value, which may not be the case for all cases.
The accuracy of the FWHM estimate can be limited by the resolution of the data.


Advantages of Gaussian envelope FFT method:
It is a more robust method for finding the FWHM of a signal because it does not require the signal to be symmetric.
It can provide additional information about the shape of the signal beyond just the FWHM.

Disadvantages of Gaussian envelope FFT method:
It requires additional transforms of the data beyond the Fourier transform, which can be computationally expensive.
The accuracy of the FWHM estimate can be limited by the resolution of the data.
"""


def ramsey_initial_guess_direct_interpolation(
    x_data: np.ndarray, y_data: np.ndarray
) -> List[float]:
    """
    Estimate initial parameters for Ramsey fit using direct interpolation method for FWHM.

    Args:
    - x_data: 1D numpy array representing the x-axis data points
    - y_data: 1D numpy array representing the y-axis data points

    Returns:
    - A list of initial guess parameters for the Ramsey fit: [offset, amplitude, frequency, phase, t2s]
    """
    # Estimate initial amplitude and offset
    amplitude = (np.max(y_data) - np.min(y_data)) / 2
    offset = np.interp(
        0, y_data[::-1], x_data[::-1]
    )  # Use interpolation to estimate offset

    # Perform FFT on the data
    fft = np.fft.fft(y_data - offset)
    fft_freq = np.fft.fftfreq(len(x_data), x_data[1] - x_data[0])

    # Find the index of the peak frequency in the FFT
    peak_idx = np.argmax(np.abs(fft[1 : len(fft) // 2])) + 1
    frequency = np.abs(fft_freq[peak_idx])
    phase = np.angle(fft[peak_idx])

    # Direct interpolation method to estimate FWHM
    half_max = (np.max(y_data) - np.min(y_data)) / 2 + np.min(y_data)
    fwhm_indices = np.where(np.diff(np.sign(y_data - half_max)))[0]
    fwhm_idx = fwhm_indices[0] if len(fwhm_indices) > 0 else np.argmax(y_data)
    t2s = 2 * np.sqrt(2 * np.log(2)) * np.abs(x_data[fwhm_idx])

    # Assemble the initial parameter guess
    initial_guess = [offset, amplitude, frequency, phase, t2s]

    return initial_guess


def ramsey_initial_guess_gaussian_envelope(
    x_data: np.ndarray, y_data: np.ndarray
) -> List[float]:
    """
    Estimate the initial parameters for a Ramsey experiment with a Gaussian envelope.

    Args:
        x_data (array-like): The time values of the signal.
        y_data (array-like): The amplitude values of the signal.

    Returns:
        list: A list containing the initial guesses for the offset, amplitude, frequency, phase,
        and T2* coherence time of the signal.

    """
    # Estimate the amplitude and offset of the signal
    amplitude = (np.max(y_data) - np.min(y_data)) / 2
    offset = np.mean(y_data)

    # Compute the Fast Fourier Transform (FFT) of the signal
    fft = np.fft.fft(y_data - offset)
    fft_freq = np.fft.fftfreq(len(x_data), x_data[1] - x_data[0])

    # Find the index of the maximum FFT peak, which corresponds to the frequency of the signal
    peak_idx = np.argmax(np.abs(fft[1 : len(fft) // 2])) + 1
    frequency = np.abs(fft_freq[peak_idx])
    phase = np.angle(fft[peak_idx])

    # Compute the Gaussian envelope of the FFT signal
    gaussian_envelope = np.abs(np.fft.ifft(fft * np.exp(-1j * phase)))

    # Estimate the full-width half-maximum (FWHM) of the Gaussian envelope using a mask
    half_max = (np.max(gaussian_envelope) - np.min(gaussian_envelope)) / 2 + np.min(
        gaussian_envelope
    )
    mask = gaussian_envelope >= half_max
    fwhm_idx = np.argmin(mask)
    t2s = 2 * np.sqrt(2 * np.log(2)) * np.abs(x_data[fwhm_idx])

    # Construct the initial guess for the Ramsey experiment parameters
    initial_guess = [offset, amplitude, frequency, phase, t2s]

    return initial_guess


"""
Advantages of Hilbert transform method:
The Hilbert transform method provides a way to compute the analytic signal of a real signal, which can be used to extract useful information
such as the amplitude envelope and instantaneous phase of the signal.
The Hilbert transform method does not assume periodicity of the signal, making it suitable for analyzing non-periodic signals.
The Hilbert transform method is less sensitive to noise and other artifacts in the data than the FFT method.


Disadvantages of Hilbert transform method:
The Hilbert transform is more computationally intensive than the FFT, which can be a disadvantage.
The Hilbert transform method does not provide an estimate of the T2* coherence time.
"""


def ramsey_initial_guess_gaussian_envelope_hilbert(
    x_data: np.ndarray, y_data: np.ndarray
) -> List[float]:
    """
    Calculates the initial guess for the Ramsey experiment parameters using the Gaussian envelope method with Hilbert
    transform.

    Args:
        x_data (array): 1-D array containing the time points for the signal.
        y_data: A 1D numpy array representing the y-axis data of the signal.

    Returns:
        A list of initial guess values for the Ramsey experiment parameters: [offset, amplitude, frequency, phase, t2s].
    """
    # Calculate the amplitude and offset of the signal
    amplitude = (np.max(y_data) - np.min(y_data)) / 2
    offset = np.mean(y_data)

    # Compute the analytic signal using the Hilbert transform
    analytic_signal = hilbert(y_data - offset)
    amplitude_envelope = np.abs(analytic_signal)

    # Find the index of the maximum amplitude envelope, which corresponds to the frequency of the signal
    peak_idx = np.argmax(amplitude_envelope)
    frequency = np.abs(
        np.mean(np.diff(np.unwrap(np.angle(analytic_signal))))
        / (2 * np.pi * np.mean(np.diff(x_data)))
    )  # Frequency of the signal
    phase = np.angle(analytic_signal[0])  # Phase of the signal

    # Compute the Gaussian envelope of the analytic signal
    gaussian_envelope = np.abs(np.fft.ifft(amplitude_envelope))
    half_max = (np.max(gaussian_envelope) - np.min(gaussian_envelope)) / 2 + np.min(
        gaussian_envelope
    )
    mask = gaussian_envelope >= half_max
    fwhm_idx = np.argmin(mask)  # Index of the full-width half-maximum (FWHM)
    t2s = (
        2 * np.sqrt(2 * np.log(2)) * np.abs(x_data[fwhm_idx])
    )  # Calculate the T2* coherence time

    # Construct the initial guess for the Ramsey experiment parameters
    initial_guess = [offset, amplitude, frequency, phase, t2s]
    return initial_guess


def ramsey_initial_guess_hilbert(x_data: np.ndarray, y_data: np.ndarray) -> List[float]:
    """
    Estimate initial parameters for a Ramsey experiment using the Hilbert transform.

    Args:
    x_data (array): 1-D array containing the time points for the signal.
    y_data (array): 1-D array containing the signal values.

    Returns:
    list: Initial parameter guess for the Ramsey experiment, consisting of [offset, amplitude, frequency, phase, t2s].
    """
    # Calculate initial guess for Ramsey experiment parameters
    amplitude = (
        np.max(y_data) - np.min(y_data)
    ) / 2  # Calculate the amplitude of the signal
    offset = np.mean(y_data)  # Calculate the offset of the signal

    # Compute the analytic signal using the Hilbert transform
    analytic_signal = hilbert(y_data - offset)

    # Find the index of the maximum amplitude envelope, which corresponds to the frequency of the signal
    peak_idx = np.argmax(np.abs(analytic_signal))
    frequency = np.abs(
        np.mean(np.diff(np.unwrap(np.angle(analytic_signal))))
        / (2 * np.pi * np.mean(np.diff(x_data)))
    )  # Frequency of the signal
    phase = np.angle(analytic_signal[0])  # Phase of the signal

    # Calculate the T2* coherence time
    t2s = np.abs(np.pi / (2 * np.log(np.abs(analytic_signal[peak_idx]))))

    # Construct the initial guess for the Ramsey experiment parameters
    initial_guess = [offset, amplitude, frequency, phase, t2s]
    return initial_guess
