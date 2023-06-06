import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

from qibocal.data import DataUnits


# Fitting
def fit_amplitude_balance_cz(data):
    """Fit phase of the target qubit detuning for both the ON and OFF sequence.

    After normalizing the data between -1 and 1, the phase, phi, of the target
    qubit is fitted using sin(2*pi*x/360 + phi) where x is the detuning of the
    target.

    Args:
        data (dict): The data to fit.

    Returns:
        dict: The fit results.
    """

    data_fit = DataUnits(
        name=f"fit",
        quantities={},
        options=[
            "flux_pulse_amplitude",
            "flux_pulse_ratio",
            "initial_phase_ON",
            "initial_phase_OFF",
            "phase_difference",
            "leakage",
            "controlqubit",
            "targetqubit",
        ],
    )

    combinations = unique_combination(data)

    amplitude_unique = np.unique(data.df["flux_pulse_amplitude"].to_numpy())
    ratio_unique = np.unique(data.df["flux_pulse_ratio"].to_numpy())
    detuning_unique = np.unique(data.df["detuning"].to_numpy())

    # Fitting function
    def f(x, phi):
        return np.sin(2 * np.pi * x / 360 + phi)

    for i in combinations:
        q_target = i[0]
        q_control = i[1]

        # Extracting Data
        iq_distance_dict = sort_data(data, q_control, q_target, "iq_distance")
        amplitude_dict = sort_data(data, q_control, q_target, "flux_pulse_amplitude")
        ratio_dict = sort_data(data, q_control, q_target, "flux_pulse_ratio")
        detuning_dict = sort_data(data, q_control, q_target, "detuning")

        for amp in amplitude_unique:
            for ratio in ratio_unique:
                try:
                    # Normalizing between -1 and 1
                    for on_off in ["ON", "OFF"]:
                        iq_distance_dict["target"][on_off][
                            (amplitude_dict["target"][on_off] == amp)
                            & (ratio_dict["target"][on_off] == ratio)
                        ] = iq_distance_dict["target"][on_off][
                            (amplitude_dict["target"][on_off] == amp)
                            & (ratio_dict["target"][on_off] == ratio)
                        ] - np.nanmean(
                            iq_distance_dict["target"][on_off][
                                (amplitude_dict["target"][on_off] == amp)
                                & (ratio_dict["target"][on_off] == ratio)
                            ]
                        )
                        iq_distance_dict["target"][on_off][
                            (amplitude_dict["target"][on_off] == amp)
                            & (ratio_dict["target"][on_off] == ratio)
                        ] = iq_distance_dict["target"][on_off][
                            (amplitude_dict["target"][on_off] == amp)
                            & (ratio_dict["target"][on_off] == ratio)
                        ] - np.max(
                            np.abs(
                                iq_distance_dict["target"][on_off][
                                    (amplitude_dict["target"][on_off] == amp)
                                    & (ratio_dict["target"][on_off] == ratio)
                                ]
                            )
                        )

                    # Fitting the data
                    popt_ON, pcov_ON = curve_fit(
                        f,
                        detuning_dict["target"]["ON"][
                            (amplitude_dict["target"]["ON"] == amp)
                            & (ratio_dict["target"]["ON"] == ratio)
                        ],
                        iq_distance_dict["target"]["ON"][
                            (amplitude_dict["target"]["ON"] == amp)
                            & (ratio_dict["target"]["ON"] == ratio)
                        ],
                        p0=[0],
                        maxfev=100000,
                    )
                    popt_OFF, pcov_OFF = curve_fit(
                        f,
                        detuning_dict["target"]["OFF"][
                            (amplitude_dict["target"]["OFF"] == amp)
                            & (ratio_dict["target"]["OFF"] == ratio)
                        ],
                        iq_distance_dict["target"]["OFF"][
                            (amplitude_dict["target"]["OFF"] == amp)
                            & (ratio_dict["target"]["OFF"] == ratio)
                        ],
                        p0=[0],
                        maxfev=100000,
                    )

                    results = {
                        "flux_pulse_amplitude": amp,
                        "flux_pulse_ratio": ratio,
                        "initial_phase_ON": np.rad2deg(popt_ON[0]) % 360,
                        "initial_phase_OFF": np.rad2deg(popt_OFF[0]) % 360,
                        "phase_difference": np.rad2deg(popt_ON[0] - popt_OFF[0]) % 360,
                        "leakage": np.mean(
                            iq_distance_dict["control"]["OFF"][
                                (amplitude_dict["target"]["OFF"] == amp)
                                & (ratio_dict["target"]["OFF"] == ratio)
                            ]
                        )
                        / 2,
                        "controlqubit": q_control,
                        "targetqubit": q_target,
                    }
                except:
                    results = {
                        "flux_pulse_amplitude": amp,
                        "flux_pulse_ratio": ratio,
                        "initial_phase_ON": np.nan,
                        "initial_phase_OFF": np.nan,
                        "phase_difference": np.nan,
                        "controlqubit": q_control,
                        "targetqubit": q_target,
                    }

                data_fit.add(results)

    return data_fit


# Function to index out data for given control and target qubits
def sort_data(data, q_control, q_target, options):
    data_dict = {"control": {}, "target": {}}
    for k in ["ON", "OFF"]:
        data_dict["control"][k] = data.df[options][
            (data.df["on_off"] == k)
            & (data.df["controlqubit"] == q_control)
            & (data.df["targetqubit"] == q_target)
            & (data.df["result_qubit"] == q_control)
        ].to_numpy()
        data_dict["target"][k] = data.df[options][
            (data.df["on_off"] == k)
            & (data.df["controlqubit"] == q_control)
            & (data.df["targetqubit"] == q_target)
            & (data.df["result_qubit"] == q_target)
        ].to_numpy()
    return data_dict


# Plotting
def amplitude_balance_cz_acquired_phase(data, data_fit, pair):
    r"""
    Plotting the acquired phase as a function of the flux pulse amplitude for a given pair of qubits.

    Args:
        data (DataUnits): DataUnits containing the data.
        data_fit (DataUnits): DataUnits containing the fitted data.
        pair (tuple): Tuple containing the control and target qubit.

    Returns:
        fig (plotly.graph_objects.Figure): Figure containing the plot.

    """

    fig = make_subplots(
        cols=2,
        rows=1,
    )
    q_target = pair[1]
    q_control = pair[0]
    fig.add_trace(
        go.Heatmap(
            z=data_fit.df["initial_phase_ON"][
                (data_fit.df["controlqubit"] == q_control)
                & (data_fit.df["targetqubit"] == q_target)
            ],
            x=data_fit.df["flux_pulse_amplitude"][
                (data_fit.df["controlqubit"] == q_control)
                & (data_fit.df["targetqubit"] == q_target)
            ],
            y=data_fit.df["flux_pulse_ratio"][
                (data_fit.df["controlqubit"] == q_control)
                & (data_fit.df["targetqubit"] == q_target)
            ],
            name=f"Q{q_control} Q{q_target}",
            colorscale="balance",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=data_fit.df["initial_phase_OFF"][
                (data_fit.df["controlqubit"] == q_control)
                & (data_fit.df["targetqubit"] == q_target)
            ],
            x=data_fit.df["flux_pulse_amplitude"][
                (data_fit.df["controlqubit"] == q_control)
                & (data_fit.df["targetqubit"] == q_target)
            ],
            y=data_fit.df["flux_pulse_ratio"][
                (data_fit.df["controlqubit"] == q_control)
                & (data_fit.df["targetqubit"] == q_target)
            ],
            name=f"Q{q_control} Q{q_target}",
            colorscale="balance",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (dimensionless)",
        yaxis_title="A/B ratio (dimensionless)",
    )
    return fig


def amplitude_balance_cz_phi2q(data, data_fit, pair):
    r"""
    Plotting the phase difference between the control and target qubit as a function of the flux pulse amplitude for a given pair of qubits.

    Args:
        data (DataUnits): DataUnits containing the data.
        data_fit (DataUnits): DataUnits containing the fitted data.
        pair (tuple): Tuple containing the control and target qubit.

    Returns:
        fig (plotly.graph_objects.Figure): Figure containing the plot.

    """

    fig = make_subplots(
        cols=1,
        rows=1,
    )

    q_target = pair[1]
    q_control = pair[0]
    fig.add_trace(
        go.Heatmap(
            z=data_fit.df["phase_difference"][
                (data_fit.df["controlqubit"] == q_control)
                & (data_fit.df["targetqubit"] == q_target)
            ],
            x=data_fit.df["flux_pulse_amplitude"][
                (data_fit.df["controlqubit"] == q_control)
                & (data_fit.df["targetqubit"] == q_target)
            ],
            y=data_fit.df["flux_pulse_ratio"][
                (data_fit.df["controlqubit"] == q_control)
                & (data_fit.df["targetqubit"] == q_target)
            ],
            name=f"Phi2Q - Q{q_control} Q{q_target}",
            colorscale="balance",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (dimensionless)",
        yaxis_title="A/B ratio (dimensionless)",
    )
    return fig


def amplitude_balance_cz_leakage(data, data_fit, pair):
    r"""
    Plotting the leakage as a function of the flux pulse amplitude for a given pair of qubits.

    Args:
        data (DataUnits): DataUnits containing the data.
        data_fit (DataUnits): DataUnits containing the fitted data.
        pair (tuple): Tuple containing the control and target qubit.

    Returns:
        fig (plotly.graph_objects.Figure): Figure containing the plot.
    """

    fig = go.Figure()

    q_target = pair[1]
    q_control = pair[0]

    fig.add_trace(
        go.Heatmap(
            z=data_fit.df["leakage"][
                (data_fit.df["controlqubit"] == q_control)
                & (data_fit.df["targetqubit"] == q_target)
            ],
            x=data_fit.df["flux_pulse_amplitude"][
                (data_fit.df["controlqubit"] == q_control)
                & (data_fit.df["targetqubit"] == q_target)
            ],
            y=data_fit.df["flux_pulse_ratio"][
                (data_fit.df["controlqubit"] == q_control)
                & (data_fit.df["targetqubit"] == q_target)
            ],
            name=f"Leakage Q{q_control} Q{q_target}",
            colorscale="balance",
        ),
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (dimensionless)",
        yaxis_title="A/B ratio (dimensionless)",
    )
    return fig


def amplitude_balance_cz_raw_data(data, data_fit, pair):
    r"""
    Plotting the raw data as a function of the flux pulse amplitude for a given pair of qubits.

    Args:
        data (DataUnits): DataUnits containing the data.
        data_fit (DataUnits): DataUnits containing the fitted data.
        pair (tuple): Tuple containing the control and target qubit.

    Returns:
        fig (plotly.graph_objects.Figure): Figure containing the plot.
    """

    fig = make_subplots(
        cols=2,
        rows=1,
    )

    q_target = pair[1]
    q_control = pair[0]

    # Point where Phi2Q is closest to 180 degrees
    idx = np.argmin(np.abs(data_fit.df["phase_difference"] - 180))
    fig.add_trace(
        go.Scatter(
            x=data.df["detuning"][
                (
                    data.df["flux_pulse_ratio"]
                    == data_fit.df["flux_pulse_ratio"].iloc[idx]
                )
                & (
                    data.df["flux_pulse_amplitude"]
                    == data_fit.df["flux_pulse_amplitude"].iloc[idx]
                )
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["on_off"] == "ON")
                & (data.df["result_qubit"] == q_target)
            ],
            y=data.df["iq_distance"][
                (
                    data.df["flux_pulse_ratio"]
                    == data_fit.df["flux_pulse_ratio"].iloc[idx]
                )
                & (
                    data.df["flux_pulse_amplitude"]
                    == data_fit.df["flux_pulse_amplitude"].iloc[idx]
                )
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["on_off"] == "ON")
                & (data.df["result_qubit"] == q_target)
            ],
            name=f"ON Q{q_target}",
        ),
        row=i + 1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=data.df["detuning"][
                (
                    data.df["flux_pulse_ratio"]
                    == data_fit.df["flux_pulse_ratio"].iloc[idx]
                )
                & (
                    data.df["flux_pulse_amplitude"]
                    == data_fit.df["flux_pulse_amplitude"].iloc[idx]
                )
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["on_off"] == "OFF")
                & (data.df["result_qubit"] == q_target)
            ],
            y=data.df["iq_distance"][
                (
                    data.df["flux_pulse_ratio"]
                    == data_fit.df["flux_pulse_ratio"].iloc[idx]
                )
                & (
                    data.df["flux_pulse_amplitude"]
                    == data_fit.df["flux_pulse_amplitude"].iloc[idx]
                )
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["on_off"] == "OFF")
                & (data.df["result_qubit"] == q_target)
            ],
            name=f"OFF Q{q_target}",
        ),
        row=i + 1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=data.df["detuning"][
                (
                    data.df["flux_pulse_ratio"]
                    == data_fit.df["flux_pulse_ratio"].iloc[idx]
                )
                & (
                    data.df["flux_pulse_amplitude"]
                    == data_fit.df["flux_pulse_amplitude"].iloc[idx]
                )
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["on_off"] == "ON")
                & (data.df["result_qubit"] == q_target)
            ],
            y=data.df["iq_distance"][
                (
                    data.df["flux_pulse_ratio"]
                    == data_fit.df["flux_pulse_ratio"].iloc[idx]
                )
                & (
                    data.df["flux_pulse_amplitude"]
                    == data_fit.df["flux_pulse_amplitude"].iloc[idx]
                )
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["on_off"] == "ON")
                & (data.df["result_qubit"] == q_control)
            ],
            name=f"ON Q{q_target}",
        ),
        row=i + 1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.df["detuning"][
                (
                    data.df["flux_pulse_ratio"]
                    == data_fit.df["flux_pulse_ratio"].iloc[idx]
                )
                & (
                    data.df["flux_pulse_amplitude"]
                    == data_fit.df["flux_pulse_amplitude"].iloc[idx]
                )
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["on_off"] == "OFF")
                & (data.df["result_qubit"] == q_target)
            ],
            y=data.df["iq_distance"][
                (
                    data.df["flux_pulse_ratio"]
                    == data_fit.df["flux_pulse_ratio"].iloc[idx]
                )
                & (
                    data.df["flux_pulse_amplitude"]
                    == data_fit.df["flux_pulse_amplitude"].iloc[idx]
                )
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["on_off"] == "OFF")
                & (data.df["result_qubit"] == q_control)
            ],
            name=f"OFF Q{q_control}",
        ),
        row=i + 1,
        col=1,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Detuning (degree)",
        yaxis_title="iq_distance",
        xaxis2_title="Detuning (degree)",
        yaxis2_title="iq_distance ",
    )
    return fig


def unique_combination(data):
    combinations = np.vstack(
        (data.df["targetqubit"].to_numpy(), data.df["controlqubit"].to_numpy())
    ).transpose()
    # Extracting unique values
    if isinstance(data.df["targetqubit"].to_numpy()[0], np.integer):
        combinations = np.unique(combinations, axis=0)
    else:
        flatdata_string = [str(p[0]) + "%" + str(p[1]) for p in combinations]
        flatdata_string = list(set(flatdata_string))
        combinations = [p.split("%") for p in flatdata_string]
    return combinations
