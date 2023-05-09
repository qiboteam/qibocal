import numpy as np


def single_exp_decay(t, offset, amplitude, frequency, phase, t2s):
    """
    Calculates the predicted values of a single exponential decay model.
    This is a simple damped sinusoidal function, which is a common fit function in Ramsey experiments.

    Args:
    - t (numpy.ndarray): Time vector.
    - offset (float): Offset parameter of the model.
    - amplitude (float): Amplitude parameter of the cosine term in the model.
    - frequency (float): Frequency parameter of the cosine term in the model.
    - phase (float): Phase parameter of the cosine term in the model.
    - t2s (float): T2* parameter of the exponential decay term in the model.

    Returns:
    - y_pred (numpy.ndarray): Predicted values of the model.
    """

    # Calculate predicted values
    y_pred = offset + amplitude * np.cos(2 * np.pi * frequency * t + phase) * np.exp(-t / t2s)

    # Return predicted values
    return y_pred


def gaussian_decay(t, offset, amplitude, frequency, phase, t2s):
    """
    Calculates the predicted values of a Gaussian decay model.
    This is a variation of the previous formula, where the exponential decay is replaced with a Gaussian decay.
    This can be a more accurate fit in situations where the noise sources are not purely exponential, and the decoherence
    is affected by other factors such as noise in the magnetic field or spin-spin interactions.

    Args:
    - t (numpy.ndarray): Time vector.
    - offset (float): Offset parameter of the model.
    - amplitude (float): Amplitude parameter of the cosine term in the model.
    - frequency (float): Frequency parameter of the cosine term in the model.
    - phase (float): Phase parameter of the cosine term in the model.
    - t2s (float): T2* parameter of the Gaussian decay term in the model.

    Returns:
    - y_pred (numpy.ndarray): Predicted values of the model.
    """
    # Calculate predicted values
    y_pred = offset + amplitude * np.cos(2 * np.pi * frequency * t + phase) * np.exp(-(t / t2s) ** 2)

    # Return predicted values
    return y_pred


def stretched_exp_decay(t, offset, amplitude, frequency, phase, t2s, beta):
    """
    Calculates the predicted values of a stretched exponential decay model.
    This function models a stretched exponential decay with a cosine oscillation.
    The function takes in the following parameters:

    Args:
    - t (numpy.ndarray): Time vector.
    - offset (float): Offset parameter of the model.
    - amplitude (float): Amplitude parameter of the cosine term in the model.
    - frequency (float): Frequency parameter of the cosine term in the model.
    - phase (float): Phase parameter of the cosine term in the model.
    - t2s (float): T2* parameter of the stretched exponential decay term in the model.
    - beta (float): Stretching parameter of the stretched exponential decay term in the model.

    Returns:
    - y_pred (numpy.ndarray): Predicted values of the model.
    """

    # Calculate predicted values
    y_pred = offset + amplitude * np.cos(2 * np.pi * frequency * t + phase) * np.exp(-(t / t2s) ** beta)

    # Return predicted values
    return y_pred


def lorentzian_decay(t, offset, amplitude, frequency, phase, t2s):
    """
    Calculates the predicted values of a stretched exponential decay model.
    This model assumes that the coherence decays as a Lorentzian function, and is commonly used when the decoherence
    is due to a single, dominant source.

    Args:
    - t (numpy.ndarray): Time vector.
    - offset (float): Offset parameter of the model.
    - amplitude (float): Amplitude parameter of the cosine term in the model.
    - frequency (float): Frequency parameter of the cosine term in the model.
    - phase (float): Phase parameter of the cosine term in the model.
    - t2s (float): T2* parameter of the Gaussian decay term in the model.

    Returns:
    - y_pred (numpy.ndarray): Predicted values of the model.
    """

    # Calculate predicted values
    y_pred = offset + amplitude * np.cos(2 * np.pi * frequency * t + phase) / (1 + (t / t2s) ** 2)

    # Return predicted values
    return y_pred


def inverse_gaussian_decay(t, offset, amplitude, frequency, phase, t2s, beta):
    """
    Calculates the predicted values of a stretched exponential decay model.
    This model assumes that the coherence decays as an inverse Gaussian function, and is commonly used when the
    decoherence is due to a combination of Gaussian and Lorentzian sources.

    Args:
    - t (numpy.ndarray): Time vector.
    - offset (float): Offset parameter of the model.
    - amplitude (float): Amplitude parameter of the cosine term in the model.
    - frequency (float): Frequency parameter of the cosine term in the model.
    - phase (float): Phase parameter of the cosine term in the model.
    - t2s (float): T2* parameter of the inverse Gaussian decay term in the model.
    - beta (float): Parameter of the inverse Gaussian decay term in the model.

    Returns:
    - y_pred (numpy.ndarray): Predicted values of the model.
    """

    # Calculate predicted values
    y_pred = offset + amplitude * np.cos(2 * np.pi * frequency * t + phase) / np.sqrt(1 + (beta * t / t2s) ** 2)

    # Return predicted values
    return y_pred
