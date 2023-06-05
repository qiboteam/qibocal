import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
import plotly.graph_objects as go
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
        quantities={
            "flux_pulse_amplitude": "dimensionless",
            "flux_pulse_ratio": "dimensionless",
            "initial_phase_ON": "degree",
            "initial_phase_OFF": "degree",
            "phase_difference": "degree",
            "leakage": "dimensionless",
        },
        options=["controlqubit", "targetqubit"],
    )

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
        iq_distance_dict = sort_data(data, q_control, q_target, ["iq_distance", "dimensionless"])
        amplitude_dict = sort_data(
            data, q_control, q_target, ["flux_pulse_amplitude", "dimensionless"]
        )
        ratio_dict = sort_data(
            data, q_control, q_target, ["flux_pulse_ratio", "dimensionless"]
        )
        detuning_dict = sort_data(data, q_control, q_target, ["detuning", "degree"])

        for amp in amplitude_unique:
            for ratio in ratio_unique:
                try:
                    # Normalizing between -1 and 1
                    for ON_OFF in ["ON", "OFF"]:
                        iq_distance_dict["target"][ON_OFF][
                            (amplitude_dict["target"][ON_OFF] == amp)
                            & (ratio_dict["target"][ON_OFF] == ratio)
                        ] = iq_distance_dict["target"][ON_OFF][
                            (amplitude_dict["target"][ON_OFF] == amp)
                            & (ratio_dict["target"][ON_OFF] == ratio)
                        ] - np.nanmean(
                            iq_distance_dict["target"][ON_OFF][
                                (amplitude_dict["target"][ON_OFF] == amp)
                                & (ratio_dict["target"][ON_OFF] == ratio)
                            ]
                        )
                        iq_distance_dict["target"][ON_OFF][
                            (amplitude_dict["target"][ON_OFF] == amp)
                            & (ratio_dict["target"][ON_OFF] == ratio)
                        ] = iq_distance_dict["target"][ON_OFF][
                            (amplitude_dict["target"][ON_OFF] == amp)
                            & (ratio_dict["target"][ON_OFF] == ratio)
                        ] - np.max(
                            np.abs(
                                iq_distance_dict["target"][ON_OFF][
                                    (amplitude_dict["target"][ON_OFF] == amp)
                                    & (ratio_dict["target"][ON_OFF] == ratio)
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
                        "flux_pulse_amplitude[dimensionless]": amp,
                        "flux_pulse_ratio[dimensionless]": ratio,
                        "initial_phase_ON[degree]": np.rad2deg(popt_ON[0]) % 360,
                        "initial_phase_OFF[degree]": np.rad2deg(popt_OFF[0]) % 360,
                        "phase_difference[degree]": np.rad2deg(popt_ON[0] - popt_OFF[0])
                        % 360,
                        "leakage[dimensionless]": np.mean(
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
                        "flux_pulse_amplitude[dimensionless]": amp,
                        "flux_pulse_ratio[dimensionless]": ratio,
                        "initial_phase_ON[degree]": np.nan,
                        "initial_phase_OFF[degree]": np.nan,
                        "phase_difference[degree]": np.nan,
                        "controlqubit": q_control,
                        "targetqubit": q_target,
                    }

                data_fit.add(results)

    return data_fit


# Function to index out data for given control and target qubits
def sort_data(data, q_control, q_target, options):
    data_dict = {"control": {}, "target": {}}
    for k in ["ON", "OFF"]:
        data_dict["control"][k] = data.get_values(options[0], options[1])[
            (data.df["ON_OFF"] == k)
            & (data.df["controlqubit"] == q_control)
            & (data.df["targetqubit"] == q_target)
            & (data.df["result_qubit"] == q_control)
        ].to_numpy()
        data_dict["target"][k] = data.get_values(options[0], options[1])[
            (data.df["ON_OFF"] == k)
            & (data.df["controlqubit"] == q_control)
            & (data.df["targetqubit"] == q_target)
            & (data.df["result_qubit"] == q_target)
        ].to_numpy()
    return data_dict

# Plotting
def amplitude_balance_cz_acquired_phase(data, data_fit, qubit):
    r"""
    Plotting function for the amplitude balance of the CZ gate.

    Args:
        folder (str): The folder where the data is stored.
        routine (str): The routine used to generate the data.
        qubit (int): The qubit to plot.
        format (str): The format of the data.

    Returns:
        fig (plotly.graph_objects.Figure): The figure.
    """

    combinations = np.vstack(
        (data_fit.df["targetqubit"].to_numpy(), data_fit.df["controlqubit"].to_numpy())
    ).transpose()
    # Extracting unique values
    if isinstance(data_fit.df["targetqubit"].to_numpy()[0], np.integer):
        combinations = np.unique(combinations, axis=0)
    else:
        flatdata_string = [str(p[0]) + "%" + str(p[1]) for p in combinations]
        flatdata_string = list(set(flatdata_string))
        combinations = [p.split("%") for p in flatdata_string]

    fig = make_subplots(
        cols=2,
        rows=len(combinations),
    )

    for i, comb in enumerate(combinations):
        q_target = comb[0]
        q_control = comb[1]
        fig.add_trace(
            go.Heatmap(
                z=data_fit.get_values("initial_phase_ON", "degree")[
                    (data_fit.df["controlqubit"] == q_control)
                    & (data_fit.df["targetqubit"] == q_target)
                ],
                x=data_fit.get_values("flux_pulse_amplitude", "dimensionless")[
                    (data_fit.df["controlqubit"] == q_control)
                    & (data_fit.df["targetqubit"] == q_target)
                ],
                y=data_fit.get_values("flux_pulse_ratio", "dimensionless")[
                    (data_fit.df["controlqubit"] == q_control)
                    & (data_fit.df["targetqubit"] == q_target)
                ],
                name=f"Q{q_control} Q{q_target}",
                colorscale="balance",
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=data_fit.get_values("initial_phase_OFF", "degree")[
                    (data_fit.df["controlqubit"] == q_control)
                    & (data_fit.df["targetqubit"] == q_target)
                ],
                x=data_fit.get_values("flux_pulse_amplitude", "dimensionless")[
                    (data_fit.df["controlqubit"] == q_control)
                    & (data_fit.df["targetqubit"] == q_target)
                ],
                y=data_fit.get_values("flux_pulse_ratio", "dimensionless")[
                    (data_fit.df["controlqubit"] == q_control)
                    & (data_fit.df["targetqubit"] == q_target)
                ],
                name=f"Q{q_control} Q{q_target}",
                colorscale="balance",
            ),
            row=i + 1,
            col=2,
        )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (dimensionless)",
        yaxis_title="A/B ratio (dimensionless)",
    )
    return fig


def amplitude_balance_cz_phi2q(data, data_fit, qubit):
    r"""
    Plotting function for the amplitude balance of the CZ gate.

    Args:
        folder (str): The folder where the data is stored.
        routine (str): The routine used to generate the data.
        qubit (int): The qubit to plot.
        format (str): The format of the data.

    Returns:
        fig (plotly.graph_objects.Figure): The figure.
    """

    combinations = np.vstack(
        (data_fit.df["targetqubit"].to_numpy(), data_fit.df["controlqubit"].to_numpy())
    ).transpose()
    # Extracting unique values
    if isinstance(data_fit.df["targetqubit"].to_numpy()[0], np.integer):
        combinations = np.unique(combinations, axis=0)
    else:
        flatdata_string = [str(p[0]) + "%" + str(p[1]) for p in combinations]
        flatdata_string = list(set(flatdata_string))
        combinations = [p.split("%") for p in flatdata_string]

    fig = make_subplots(
        cols=1,
        rows=len(combinations),
    )

    for i, comb in enumerate(combinations):
        q_target = comb[0]
        q_control = comb[1]
        fig.add_trace(
            go.Heatmap(
                z=data_fit.get_values("phase_difference", "degree")[
                    (data_fit.df["controlqubit"] == q_control)
                    & (data_fit.df["targetqubit"] == q_target)
                ],
                x=data_fit.get_values("flux_pulse_amplitude", "dimensionless")[
                    (data_fit.df["controlqubit"] == q_control)
                    & (data_fit.df["targetqubit"] == q_target)
                ],
                y=data_fit.get_values("flux_pulse_ratio", "dimensionless")[
                    (data_fit.df["controlqubit"] == q_control)
                    & (data_fit.df["targetqubit"] == q_target)
                ],
                name=f"Phi2Q - Q{q_control} Q{q_target}",
                colorscale="balance",
            ),
            row=i + 1,
            col=1,
        )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (dimensionless)",
        yaxis_title="A/B ratio (dimensionless)",
    )
    return fig


def amplitude_balance_cz_leakage(data, data_fit, qubit):
    r"""
    Plotting function for the amplitude balance of the CZ gate.

    Args:
        folder (str): The folder where the data is stored.
        routine (str): The routine used to generate the data.
        qubit (int): The qubit to plot.
        format (str): The format of the data.

    Returns:
        fig (plotly.graph_objects.Figure): The figure.
    """

    combinations = np.vstack(
        (data_fit.df["targetqubit"].to_numpy(), data_fit.df["controlqubit"].to_numpy())
    ).transpose()
    # Extracting unique values
    if isinstance(data_fit.df["targetqubit"].to_numpy()[0], np.integer):
        combinations = np.unique(combinations, axis=0)
    else:
        flatdata_string = [str(p[0]) + "%" + str(p[1]) for p in combinations]
        flatdata_string = list(set(flatdata_string))
        combinations = [p.split("%") for p in flatdata_string]

    fig = make_subplots(
        cols=1,
        rows=len(combinations),
    )

    for i, comb in enumerate(combinations):
        q_target = comb[0]
        q_control = comb[1]
        fig.add_trace(
            go.Heatmap(
                z=data_fit.get_values("leakage", "dimensionless")[
                    (data_fit.df["controlqubit"] == q_control)
                    & (data_fit.df["targetqubit"] == q_target)
                ],
                x=data_fit.get_values("flux_pulse_amplitude", "dimensionless")[
                    (data_fit.df["controlqubit"] == q_control)
                    & (data_fit.df["targetqubit"] == q_target)
                ],
                y=data_fit.get_values("flux_pulse_ratio", "dimensionless")[
                    (data_fit.df["controlqubit"] == q_control)
                    & (data_fit.df["targetqubit"] == q_target)
                ],
                name=f"Leakage Q{q_control} Q{q_target}",
                colorscale="balance",
            ),
            row=i + 1,
            col=1,
        )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (dimensionless)",
        yaxis_title="A/B ratio (dimensionless)",
    )
    return fig


def amplitude_balance_cz_raw_data(data, data_fit, qubit):
    r"""
    Plotting function for the amplitude balance of the CZ gate.

    Args:
        folder (str): The folder where the data is stored.
        routine (str): The routine used to generate the data.
        qubit (int): The qubit to plot.
        format (str): The format of the data.

    Returns:
        fig (plotly.graph_objects.Figure): The figure.
    """
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

    fig = make_subplots(
        cols=2,
        rows=len(combinations),
    )

    for i, comb in enumerate(combinations):
        q_target = comb[0]
        q_control = comb[1]

        # Point where Phi2Q is closest to 180 degrees
        idx = np.argmin(np.abs(data_fit.get_values("phase_difference", "degree") - 180))

        fig.add_trace(
            go.Scatter(
                x=data.get_values("detuning", "degree")[
                    (
                        data.df["flux_pulse_ratio"]
                        == data_fit.df["flux_pulse_ratio"][idx]
                    )
                    & (
                        data.df["flux_pulse_amplitude"]
                        == data_fit.df["flux_pulse_amplitude"][idx]
                    )
                    & (data.df["controlqubit"] == q_control)
                    & (data.df["targetqubit"] == q_target)
                    & (data.df["on_off"] == "ON")
                    & (data.df["result_qubit"] == q_target)
                ],
                y=data.get_values("prob", "dimensionless")[
                    (
                        data.df["flux_pulse_ratio"]
                        == data_fit.df["flux_pulse_ratio"][idx]
                    )
                    & (
                        data.df["flux_pulse_amplitude"]
                        == data_fit.df["flux_pulse_amplitude"][idx]
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
                x=data.get_values("detuning", "degree")[
                    (
                        data.df["flux_pulse_ratio"]
                        == data_fit.df["flux_pulse_ratio"][idx]
                    )
                    & (
                        data.df["flux_pulse_amplitude"]
                        == data_fit.df["flux_pulse_amplitude"][idx]
                    )
                    & (data.df["controlqubit"] == q_control)
                    & (data.df["targetqubit"] == q_target)
                    & (data.df["on_off"] == "OFF")
                    & (data.df["result_qubit"] == q_target)
                ],
                y=data.get_values("prob", "dimensionless")[
                    (
                        data.df["flux_pulse_ratio"]
                        == data_fit.df["flux_pulse_ratio"][idx]
                    )
                    & (
                        data.df["flux_pulse_amplitude"]
                        == data_fit.df["flux_pulse_amplitude"][idx]
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
                x=data.get_values("detuning", "degree")[
                    (
                        data.df["flux_pulse_ratio"]
                        == data_fit.df["flux_pulse_ratio"][idx]
                    )
                    & (
                        data.df["flux_pulse_amplitude"]
                        == data_fit.df["flux_pulse_amplitude"][idx]
                    )
                    & (data.df["controlqubit"] == q_control)
                    & (data.df["targetqubit"] == q_target)
                    & (data.df["on_off"] == "ON")
                    & (data.df["result_qubit"] == q_target)
                ],
                y=data.get_values("prob", "dimensionless")[
                    (
                        data.df["flux_pulse_ratio"]
                        == data_fit.df["flux_pulse_ratio"][idx]
                    )
                    & (
                        data.df["flux_pulse_amplitude"]
                        == data_fit.df["flux_pulse_amplitude"][idx]
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
                x=data.get_values("detuning", "degree")[
                    (
                        data.df["flux_pulse_ratio"]
                        == data_fit.df["flux_pulse_ratio"][idx]
                    )
                    & (
                        data.df["flux_pulse_amplitude"]
                        == data_fit.df["flux_pulse_amplitude"][idx]
                    )
                    & (data.df["controlqubit"] == q_control)
                    & (data.df["targetqubit"] == q_target)
                    & (data.df["on_off"] == "OFF")
                    & (data.df["result_qubit"] == q_target)
                ],
                y=data.get_values("prob", "dimensionless")[
                    (
                        data.df["flux_pulse_ratio"]
                        == data_fit.df["flux_pulse_ratio"][idx]
                    )
                    & (
                        data.df["flux_pulse_amplitude"]
                        == data_fit.df["flux_pulse_amplitude"][idx]
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
        yaxis_title="iq_distance ",
        xaxis2_title="Detuning (degree)",
        yaxis2_title="iq_distance ",
    )
    return fig

