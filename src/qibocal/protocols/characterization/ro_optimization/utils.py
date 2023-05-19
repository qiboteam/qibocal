import numpy as np
from scipy.optimize import curve_fit

from qibocal.data import Data


def ro_optimization_fit(data, *labels, debug=False):
    """
    Fit the fidelities from parameters swept as labels, and extract rotation angle and threshold

    Args:
        data (Data): data to fit
        labels (str): variable used in the routine with format "variable_name"

    Returns:
        Data: data with the fit results
    """
    quantities = [
        *labels,
        "rotation_angle",
        "threshold",
        "fidelity",
        "assignment_fidelity",
        "average_state0",
        "average_state1",
    ]
    data_fit = Data(
        name="fit",
        quantities=quantities,
    )

    # Create a ndarray for i and q shots for all labels
    # shape=(i + j*q, qubit, state, label1, label2, ...)

    shape = (*[len(data.df[label].unique()) for label in labels],)
    nb_shots = len(data.df["iteration"].unique())

    iq_complex = data.df["i"].pint.magnitude.to_numpy().reshape(shape) + 1j * data.df[
        "q"
    ].pint.magnitude.to_numpy().reshape(shape)

    # Move state to 0, and iteration to -1
    labels = list(labels)
    iq_complex = np.moveaxis(
        iq_complex, [labels.index("state"), labels.index("iteration")], [0, -1]
    )
    labels.remove("state")
    labels.remove("iteration")
    labels = ["state"] + labels + ["iteration"]

    # Take the mean ground state
    mean_gnd_state = np.mean(iq_complex[0, ...], axis=-1, keepdims=True)
    mean_exc_state = np.mean(iq_complex[1, ...], axis=-1, keepdims=True)
    angle = np.angle(mean_exc_state - mean_gnd_state)

    # Rotate the data
    iq_complex = iq_complex * np.exp(-1j * angle)

    # Take the cumulative distribution of the real part of the data
    iq_complex_sorted = np.sort(iq_complex.real, axis=-1)

    def cum_dist(complex_row):
        state0 = complex_row.real
        state1 = complex_row.imag
        combined = np.sort(np.concatenate((state0, state1)))

        # Compute the indices where elements in state0 and state1 would be inserted in combined
        idx_state0 = np.searchsorted(combined, state0, side="left")
        idx_state1 = np.searchsorted(combined, state1, side="left")

        # Create a combined histogram for state0 and state1
        hist_combined = np.bincount(
            idx_state0, minlength=len(combined)
        ) + 1j * np.bincount(idx_state1, minlength=len(combined))

        return hist_combined.cumsum()

    cum_dist = (
        np.apply_along_axis(
            func1d=cum_dist,
            axis=-1,
            arr=iq_complex_sorted[0, ...] + 1j * iq_complex_sorted[1, ...],
        )
        / nb_shots
    )

    # Find the threshold for which the difference between the cumulative distribution of the two states is maximum
    argmax = np.argmax(np.abs(cum_dist.real - cum_dist.imag), axis=-1, keepdims=True)

    # Use np.take_along_axis to get the correct indices for the threshold calculation
    threshold = np.take_along_axis(
        np.concatenate((iq_complex_sorted[0, ...], iq_complex_sorted[1, ...]), axis=-1),
        argmax,
        axis=-1,
    )

    # Calculate the fidelity
    fidelity = np.take_along_axis(
        np.abs(cum_dist.real - cum_dist.imag), argmax, axis=-1
    )
    assignment_fidelity = (
        1
        - (
            1
            - np.take_along_axis(cum_dist.real, argmax, axis=-1)
            + np.take_along_axis(cum_dist.imag, argmax, axis=-1)
        )
        / 2
    )

    # Add all the results to the data with labels as subnet without "state", "iteration"
    data_fit.df = (
        data.df.drop_duplicates(
            subset=[i for i in labels if i not in ["state", "iteration"]]
        )
        .reset_index(drop=True)
        .apply(pint_to_float)
    )
    data_fit.df["rotation_angle"] = angle.flatten()
    data_fit.df["threshold"] = threshold.flatten()
    data_fit.df["fidelity"] = fidelity.flatten()
    data_fit.df["assignment_fidelity"] = assignment_fidelity.flatten()
    data_fit.df["average_state0"] = mean_gnd_state.flatten()
    data_fit.df["average_state1"] = mean_exc_state.flatten()

    if debug:
        return data_fit, cum_dist, iq_complex
    else:
        return data_fit


def ro_optimization_plot(data, *labels):
    pass
