import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import scipy.signal as ss
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.signal import lfilter

from qibocal.data import Data, DataUnits
from qibocal.fitting import sort_data


def amplitude_balance_cz_acquired_phase(folder, routine, qubit, format):
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
    try:
        data_fit = DataUnits.load_data(folder, routine, format, f"fit")
    except:
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


def amplitude_balance_cz_phi2q(folder, routine, qubit, format):
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
    try:
        data_fit = DataUnits.load_data(folder, routine, format, f"fit")
    except:
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


def amplitude_balance_cz_leakage(folder, routine, qubit, format):
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
    try:
        data_fit = DataUnits.load_data(folder, routine, format, f"fit")
    except:
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


def amplitude_balance_cz_raw_data(folder, routine, qubit, format):
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
    try:
        data = DataUnits.load_data(folder, routine, format, f"data")
        data_fit = DataUnits.load_data(folder, routine, format, f"fit")
    except:
        data = DataUnits(
            name=f"data",
            quantities={
                "detuning": "degree",
                "flux_pulse_amplitude": "dimensionless",
                "flux_pulse_ratio": "dimensionless",
                "prob": "dimensionless",
            },
            options=["controlqubit", "targetqubit", "ON_OFF", "result_qubit"],
        )
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
                    & (data.df["ON_OFF"] == "ON")
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
                    & (data.df["ON_OFF"] == "ON")
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
                    & (data.df["ON_OFF"] == "OFF")
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
                    & (data.df["ON_OFF"] == "OFF")
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
                    & (data.df["ON_OFF"] == "ON")
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
                    & (data.df["ON_OFF"] == "ON")
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
                    & (data.df["ON_OFF"] == "OFF")
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
                    & (data.df["ON_OFF"] == "OFF")
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
        yaxis_title="Probability ",
        xaxis2_title="Detuning (degree)",
        yaxis2_title="Probability ",
    )
    return fig


# For cz_ramsey
def snz(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")

    ON = "ON"
    OFF = "OFF"

    # Making figure
    figs = {}
    combinations = np.unique(
        np.vstack(
            (data.df["targetqubit"].to_numpy(), data.df["controlqubit"].to_numpy())
        ).transpose(),
        axis=0,
    )
    m_types = pd.unique(data.df["Mtype"])
    for i in combinations:
        q_target = i[0]
        q_control = i[1]
        for m_type in m_types:
            # Extracting Data
            QubitControl_ON = data.get_values("prob", "dimensionless")[
                (data.df["ON_OFF"] == ON)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_control)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()
            QubitControl_OFF = data.get_values("prob", "dimensionless")[
                (data.df["ON_OFF"] == OFF)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_control)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()
            QubitTarget_ON = data.get_values("prob", "dimensionless")[
                (data.df["ON_OFF"] == ON)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_target)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()
            QubitTarget_OFF = data.get_values("prob", "dimensionless")[
                (data.df["ON_OFF"] == OFF)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_target)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()

            amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
            duration = data.get_values("flux_pulse_duration", "ns")
            flux_pulse_duration = duration[
                (data.df["ON_OFF"] == OFF)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_target)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()
            flux_pulse_amplitude = amplitude[
                (data.df["ON_OFF"] == OFF)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_target)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()

            amplitude_unique = flux_pulse_amplitude[
                flux_pulse_duration == flux_pulse_duration[0]
            ]
            duration_unique = flux_pulse_duration[
                flux_pulse_amplitude == flux_pulse_amplitude[0]
            ]

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"] = make_subplots(
                rows=2,
                cols=2,
                horizontal_spacing=0.1,
                vertical_spacing=0.2,
                subplot_titles=(
                    f"Qubit Target {q_target}",
                    f"Qubit Control {q_control}",
                ),
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_duration,
                    y=flux_pulse_amplitude,
                    z=QubitTarget_ON,
                    name="QubitTarget ON",
                    colorbar=dict(len=0.46, y=0.75),
                ),
                row=1,
                col=1,
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_duration,
                    y=flux_pulse_amplitude,
                    z=QubitTarget_OFF,
                    colorbar=dict(len=0.46, y=0.25),
                    name="QubitTarget OFF",
                ),
                row=2,
                col=1,
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_duration,
                    y=flux_pulse_amplitude,
                    z=QubitControl_ON,
                    colorbar=dict(len=0.46, y=0.75),
                    name="QubitControl ON",
                ),
                row=1,
                col=2,
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_duration,
                    y=flux_pulse_amplitude,
                    z=QubitControl_OFF,
                    colorbar=dict(len=0.46, y=0.25),
                    name="QubitControl OFF",
                ),
                row=2,
                col=2,
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].update_layout(
                xaxis_title="Pulse detuning (deg)",
                yaxis_title="Amplitude (a.u.)",
                xaxis2_title="Pulse detuning (deg)",
                yaxis2_title="Amplitude (a.u.)",
                title=f"Raw data",
            )

            # Normalizing between -1 and 1, and getting simple phase
            QubitTarget_ON_matrix = (
                np.ones((len(amplitude_unique), len(duration_unique))) * np.nan
            )
            QubitTarget_OFF_matrix = (
                np.ones((len(amplitude_unique), len(duration_unique))) * np.nan
            )

            for i, t in enumerate(duration_unique):
                n = int(np.where(flux_pulse_duration == duration_unique[-1])[0][-1]) + 1
                idx = np.where(flux_pulse_duration[:n] == t)
                QubitTarget_OFF_matrix[
                    : n // len(duration_unique), i
                ] = QubitTarget_OFF[idx]
                QubitTarget_ON_matrix[: n // len(duration_unique), i] = QubitTarget_ON[
                    idx
                ]

            figs[f"c{q_control}_t{q_target}_norm_{m_type}"] = make_subplots(
                rows=1,
                cols=2,
                horizontal_spacing=0.1,
                vertical_spacing=0.2,
                subplot_titles=(
                    f"Qubit Target {q_target} ON",
                    f"Qubit Target {q_target} OFF",
                ),
            )

            QubitTarget_ON_norm = QubitTarget_ON_matrix - 0.5
            QubitTarget_ON_norm = QubitTarget_ON_norm / np.max(
                np.abs(QubitTarget_ON_norm)
            )
            QubitTarget_OFF_norm = QubitTarget_OFF_matrix - 0.5
            QubitTarget_OFF_norm = QubitTarget_OFF_norm / np.max(
                np.abs(QubitTarget_OFF_norm)
            )

            figs[f"c{q_control}_t{q_target}_norm_{m_type}"].add_trace(
                go.Heatmap(
                    x=duration_unique,
                    y=amplitude_unique,
                    z=QubitTarget_OFF_norm,
                    colorbar=dict(len=0.46, y=0.75),
                    name="QubitControl OFF",
                ),
                row=1,
                col=1,
            )

            figs[f"c{q_control}_t{q_target}_norm_{m_type}"].add_trace(
                go.Heatmap(
                    x=duration_unique,
                    y=amplitude_unique,
                    z=QubitTarget_ON_norm,
                    colorbar=dict(len=0.46, y=0.25),
                    name="QubitControl ON",
                ),
                row=1,
                col=2,
            )

            figs[f"c{q_control}_t{q_target}_norm_{m_type}"].update_layout(
                uirevision="0",
                yaxis_title="Amplitude (a.u.)",
                xaxis_title="t_p",
                title=f"Control {q_control}, Target {q_target}",
            )

            # Calculating the phase
            figs[f"c{q_control}_t{q_target}_phase_{m_type}"] = go.Figure()
            figs[f"c{q_control}_t{q_target}_phase_{m_type}"].add_trace(
                go.Heatmap(
                    x=duration_unique,
                    y=amplitude_unique,
                    z=np.rad2deg(
                        np.arccos(QubitTarget_ON_norm) - np.arccos(QubitTarget_OFF_norm)
                    ),
                    name="Phase arcos(ON) - arcos(OFF)",
                ),
            )
            figs[f"c{q_control}_t{q_target}_phase_{m_type}"].update_layout(
                uirevision="0",
                yaxis_title="Amplitude (a.u.)",
                xaxis_title="t_p",
                title=f"Control {q_control}, Target {q_target}",
            )

            # Calculating the phase from complex number
            figs[f"c{q_control}_t{q_target}_phase_complex_{m_type}"] = go.Figure()
            figs[f"c{q_control}_t{q_target}_phase_complex_{m_type}"].add_trace(
                go.Heatmap(
                    x=duration_unique,
                    y=amplitude_unique,
                    z=np.rad2deg(
                        np.angle(QubitTarget_ON_matrix + 1j * QubitTarget_OFF_matrix)
                    ),
                    name="Phase Complex ON + 1j * OFF",
                ),
            )
            figs[f"c{q_control}_t{q_target}_phase_complex_{m_type}"].update_layout(
                uirevision="0",
                yaxis_title="Amplitude (a.u.)",
                xaxis_title="t_p",
                title=f"Control {q_control}, Target {q_target}",
            )

            # Calculating the phase from complex number
            figs[f"c{q_control}_t{q_target}_phase_complex_rad_{m_type}"] = go.Figure()
            figs[f"c{q_control}_t{q_target}_phase_complex_rad_{m_type}"].add_trace(
                go.Heatmap(
                    x=duration_unique,
                    y=amplitude_unique,
                    z=np.angle(QubitTarget_ON_matrix + 1j * QubitTarget_OFF_matrix),
                    name="Phase Complex ON + 1j * OFF",
                ),
            )
            figs[f"c{q_control}_t{q_target}_phase_complex_rad_{m_type}"].update_layout(
                uirevision="0",
                yaxis_title="Amplitude (a.u.)",
                xaxis_title="t_p",
                title=f"Control {q_control}, Target {q_target}",
            )

            # Calculating the differance between ON and OFF
            figs[f"c{q_control}_t{q_target}_diff_{m_type}"] = go.Figure()
            figs[f"c{q_control}_t{q_target}_diff_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_duration,  # duration_unique,
                    y=flux_pulse_amplitude,  # amplitude_unique,
                    z=QubitTarget_OFF
                    - QubitTarget_ON,  # QubitTarget_ON_matrix - QubitTarget_OFF_matrix,
                    name="Diff",
                ),
            )
            figs[f"c{q_control}_t{q_target}_phase_complex_rad_{m_type}"].update_layout(
                uirevision="0",
                yaxis_title="Amplitude (a.u.)",
                xaxis_title="t_p",
                title=f"Control {q_control}, Target {q_target}",
            )

            # Unwrap phase
            figs[f"c{q_control}_t{q_target}_unwrap_rad_{m_type}"] = make_subplots(
                rows=1,
                cols=2,
                horizontal_spacing=0.1,
                vertical_spacing=0.2,
                subplot_titles=(
                    f"Qubit Target {q_target} ON",
                    f"Qubit Target {q_target} OFF",
                ),
            )
            figs[f"c{q_control}_t{q_target}_unwrap_rad_{m_type}"].add_trace(
                go.Heatmap(
                    x=duration_unique,
                    y=amplitude_unique,
                    z=np.unwrap(
                        np.clip(QubitTarget_ON_matrix, a_min=0, a_max=1),
                        axis=0,
                        period=1,
                    ),
                    name="Phase Complex ON + 1j * OFF",
                ),
                row=1,
                col=1,
            )
            figs[f"c{q_control}_t{q_target}_unwrap_rad_{m_type}"].add_trace(
                go.Heatmap(
                    x=duration_unique,
                    y=amplitude_unique,
                    z=np.unwrap(
                        np.clip(QubitTarget_OFF_matrix, a_min=0, a_max=-1),
                        axis=0,
                        period=1,
                    ),
                    name="Phase Complex ON + 1j * OFF",
                ),
                row=1,
                col=2,
            )
            figs[f"c{q_control}_t{q_target}_unwrap_rad_{m_type}"].update_layout(
                uirevision="0",
                yaxis_title="Amplitude (a.u.)",
                xaxis_title="t_p",
                title=f"Control {q_control}, Target {q_target}",
            )

    return figs


def snz_detuning(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")

    ON = "ON"
    OFF = "OFF"

    # Making figure
    figs = {}
    combinations = np.unique(
        np.vstack(
            (data.df["targetqubit"].to_numpy(), data.df["controlqubit"].to_numpy())
        ).transpose(),
        axis=0,
    )
    m_types = pd.unique(data.df["Mtype"])
    for i in combinations:
        q_target = i[0]
        q_control = i[1]
        for m_type in m_types:
            # Extracting Data
            QubitControl_ON = data.get_values("prob", "dimensionless")[
                (data.df["ON_OFF"] == ON)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_control)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()
            QubitControl_OFF = data.get_values("prob", "dimensionless")[
                (data.df["ON_OFF"] == OFF)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_control)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()
            QubitTarget_ON = data.get_values("prob", "dimensionless")[
                (data.df["ON_OFF"] == ON)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_target)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()
            QubitTarget_OFF = data.get_values("prob", "dimensionless")[
                (data.df["ON_OFF"] == OFF)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_target)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()

            amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
            detuning = data.get_values("flux_pulse_detuning", "degree")
            flux_pulse_detuning = detuning[
                (data.df["ON_OFF"] == OFF)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_target)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()
            flux_pulse_amplitude = amplitude[
                (data.df["ON_OFF"] == OFF)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_target)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()

            amplitude_unique = flux_pulse_amplitude[
                flux_pulse_detuning == flux_pulse_detuning[0]
            ]
            detuning_unique = flux_pulse_detuning[
                flux_pulse_amplitude == flux_pulse_amplitude[0]
            ]

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"] = make_subplots(
                rows=2,
                cols=2,
                horizontal_spacing=0.1,
                vertical_spacing=0.2,
                subplot_titles=(
                    f"Qubit Target {q_target}",
                    f"Qubit Control {q_control}",
                ),
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_detuning,
                    y=flux_pulse_amplitude,
                    z=QubitTarget_ON,
                    name="QubitTarget ON",
                    colorbar=dict(len=0.46, y=0.75),
                ),
                row=1,
                col=1,
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_detuning,
                    y=flux_pulse_amplitude,
                    z=QubitTarget_OFF,
                    colorbar=dict(len=0.46, y=0.25),
                    name="QubitTarget OFF",
                ),
                row=2,
                col=1,
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_detuning,
                    y=flux_pulse_amplitude,
                    z=QubitControl_ON,
                    colorbar=dict(len=0.46, y=0.75),
                    name="QubitControl ON",
                ),
                row=1,
                col=2,
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_detuning,
                    y=flux_pulse_amplitude,
                    z=QubitControl_OFF,
                    colorbar=dict(len=0.46, y=0.25),
                    name="QubitControl OFF",
                ),
                row=2,
                col=2,
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].update_layout(
                xaxis_title="Pulse detuning (deg)",
                yaxis_title="Amplitude (a.u.)",
                xaxis2_title="Pulse detuning (deg)",
                yaxis2_title="Amplitude (a.u.)",
                title=f"Raw data",
            )

            # Normalizing between -1 and 1, and getting simple phase
            QubitTarget_ON_matrix = (
                np.ones((len(amplitude_unique), len(detuning_unique))) * np.nan
            )
            QubitTarget_OFF_matrix = (
                np.ones((len(amplitude_unique), len(detuning_unique))) * np.nan
            )

            for i, t in enumerate(detuning_unique):
                n = int(np.where(flux_pulse_detuning == detuning_unique[-1])[0][-1]) + 1
                idx = np.where(flux_pulse_detuning[:n] == t)
                QubitTarget_OFF_matrix[
                    : n // len(detuning_unique), i
                ] = QubitTarget_OFF[idx]
                QubitTarget_ON_matrix[: n // len(detuning_unique), i] = QubitTarget_ON[
                    idx
                ]

            figs[f"c{q_control}_t{q_target}_norm_{m_type}"] = make_subplots(
                rows=1,
                cols=2,
                horizontal_spacing=0.1,
                vertical_spacing=0.2,
                subplot_titles=(
                    f"Qubit Target {q_target} ON",
                    f"Qubit Target {q_target} OFF",
                ),
            )

            QubitTarget_ON_norm = QubitTarget_ON_matrix - np.vstack(
                (
                    np.max(QubitTarget_ON_matrix, axis=1)
                    + np.min(QubitTarget_ON_matrix, axis=1)
                )
                / 2
            )
            QubitTarget_ON_norm = QubitTarget_ON_norm / np.vstack(
                np.max(QubitTarget_ON_norm, axis=1)
            )

            QubitTarget_OFF_norm = QubitTarget_OFF_matrix - np.vstack(
                (
                    np.max(QubitTarget_OFF_matrix, axis=1)
                    + np.min(QubitTarget_OFF_matrix, axis=1)
                )
                / 2
            )
            QubitTarget_OFF_norm = QubitTarget_OFF_norm / np.vstack(
                np.max(QubitTarget_OFF_norm, axis=1)
            )

            figs[f"c{q_control}_t{q_target}_norm_{m_type}"].add_trace(
                go.Heatmap(
                    x=detuning_unique,
                    y=amplitude_unique,
                    z=QubitTarget_OFF_norm,
                    colorbar=dict(len=0.46, y=0.75),
                    name="QubitControl OFF",
                ),
                row=1,
                col=1,
            )

            figs[f"c{q_control}_t{q_target}_norm_{m_type}"].add_trace(
                go.Heatmap(
                    x=detuning_unique,
                    y=amplitude_unique,
                    z=QubitTarget_ON_norm,
                    colorbar=dict(len=0.46, y=0.25),
                    name="QubitControl ON",
                ),
                row=1,
                col=2,
            )

            figs[f"c{q_control}_t{q_target}_norm_{m_type}"].update_layout(
                uirevision="0",
                yaxis_title="Amplitude (a.u.)",
                xaxis_title="t_p",
                title=f"Control {q_control}, Target {q_target}",
            )

            # Calculating the phase
            figs[f"c{q_control}_t{q_target}_phase_{m_type}"] = go.Figure()
            figs[f"c{q_control}_t{q_target}_phase_{m_type}"].add_trace(
                go.Heatmap(
                    x=detuning_unique,
                    y=amplitude_unique,
                    z=np.rad2deg(
                        np.arccos(QubitTarget_ON_norm) - np.arccos(QubitTarget_OFF_norm)
                    ),
                    name="Phase arcos(ON) - arcos(OFF)",
                ),
            )
            figs[f"c{q_control}_t{q_target}_phase_{m_type}"].update_layout(
                uirevision="0",
                yaxis_title="Amplitude (a.u.)",
                xaxis_title="t_p",
                title=f"Control {q_control}, Target {q_target}",
            )

            # Calculating the phase from complex number
            figs[f"c{q_control}_t{q_target}_phase_complex_{m_type}"] = go.Figure()
            figs[f"c{q_control}_t{q_target}_phase_complex_{m_type}"].add_trace(
                go.Heatmap(
                    x=detuning_unique,
                    y=amplitude_unique,
                    z=np.rad2deg(
                        np.angle(QubitTarget_ON_matrix + 1j * QubitTarget_OFF_matrix)
                    )
                    % 180,
                    name="Phase Complex ON + 1j * OFF",
                ),
            )
            figs[f"c{q_control}_t{q_target}_phase_complex_{m_type}"].update_layout(
                uirevision="0",
                yaxis_title="Amplitude (a.u.)",
                xaxis_title="t_p",
                title=f"Control {q_control}, Target {q_target}",
            )

            # Calculating the differance between ON and OFF
            figs[f"c{q_control}_t{q_target}_diff_{m_type}"] = go.Figure()
            figs[f"c{q_control}_t{q_target}_diff_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_detuning,  # duration_unique,
                    y=flux_pulse_amplitude,  # amplitude_unique,
                    z=QubitTarget_OFF
                    - QubitTarget_ON,  # QubitTarget_ON_matrix - QubitTarget_OFF_matrix,
                    name="Diff",
                ),
            )
            figs[f"c{q_control}_t{q_target}_diff_{m_type}"].update_layout(
                uirevision="0",
                yaxis_title="Amplitude (a.u.)",
                xaxis_title="t_p",
                title=f"Control {q_control}, Target {q_target}",
            )

            # Mean complex phase
            figs[f"c{q_control}_t{q_target}_phase_complex_mean_{m_type}"] = go.Figure()
            figs[f"c{q_control}_t{q_target}_phase_complex_mean_{m_type}"].add_trace(
                go.Scatter(
                    x=amplitude_unique,
                    y=np.nanmean(
                        np.rad2deg(
                            np.arccos(QubitTarget_ON_norm)
                            - np.arccos(QubitTarget_OFF_norm)
                        ),
                        axis=1,
                    ),
                    name="Phase Complex ON + 1j * OFF",
                ),
            )
            figs[f"c{q_control}_t{q_target}_phase_complex_mean_{m_type}"].update_layout(
                uirevision="0",
                yaxis_title="mean phi2Q (deg)",
                xaxis_title="Amplitude (a.u.)",
                title=f"Control {q_control}, Target {q_target}",
            )

            # Fit and plot phase

            def f(x, phi):
                return np.sin(2 * np.pi * x / 360 + phi)

            popt_ON = np.array([])
            popt_OFF = np.array([])
            for i, amp in enumerate(amplitude_unique):
                if not np.isnan(QubitTarget_OFF_norm[i, :]).any():
                    popt, _ = curve_fit(
                        f,
                        detuning_unique,
                        QubitTarget_ON_norm[i, :],
                        p0=[0],
                        maxfev=1000,
                    )
                    popt_ON = np.concatenate((popt_ON, popt))
                    popt, _ = curve_fit(
                        f,
                        detuning_unique,
                        QubitTarget_OFF_norm[i, :],
                        p0=[0],
                        maxfev=1000,
                    )
                    popt_OFF = np.concatenate((popt_OFF, popt))
                else:
                    popt_ON = np.concatenate((popt_ON, [np.nan]))
                    popt_OFF = np.concatenate((popt_OFF, [np.nan]))
            angle_ON = np.rad2deg(popt_ON) % 360
            angle_OFF = np.rad2deg(popt_OFF) % 360
            figs[f"c{q_control}_t{q_target}_phase_fft_{m_type}"] = go.Figure()
            figs[f"c{q_control}_t{q_target}_phase_fft_{m_type}"].add_trace(
                go.Scatter(
                    x=amplitude_unique,
                    y=(angle_OFF - angle_ON + 180) % 360 - 180,
                    name="Phase FFT",
                ),
            )
            figs[f"c{q_control}_t{q_target}_phase_fft_{m_type}"].update_layout(
                uirevision="0",
                yaxis_title="phi2Q (deg)",
                xaxis_title="Amplitude (a.u.)",
                title=f"Control {q_control}, Target {q_target}",
            )

    return figs  # [f"c{q_control}_t{q_target}_raw_{m_type}"]


def chevron_iswap(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")

    # Making figure
    figs = {}
    combinations = np.unique(
        np.vstack(
            (data.df["targetqubit"].to_numpy(), data.df["controlqubit"].to_numpy())
        ).transpose(),
        axis=0,
    )
    for i in combinations:
        q_target = i[0]
        q_control = i[1]

        # Extracting Data
        QubitTarget = data.get_values("prob", "dimensionless")[
            (data.df["controlqubit"] == q_control)
            & (data.df["targetqubit"] == q_target)
            & (data.df["result_qubit"] == q_target)
        ].to_numpy()

        amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
        duration = data.get_values("flux_pulse_duration", "ns")
        flux_pulse_duration = duration[
            (data.df["controlqubit"] == q_control)
            & (data.df["targetqubit"] == q_target)
            & (data.df["result_qubit"] == q_target)
        ].to_numpy()
        flux_pulse_amplitude = amplitude[
            (data.df["controlqubit"] == q_control)
            & (data.df["targetqubit"] == q_target)
            & (data.df["result_qubit"] == q_target)
        ].to_numpy()

        amplitude_unique = flux_pulse_amplitude[
            flux_pulse_duration == flux_pulse_duration[0]
        ]
        duration_unique = flux_pulse_duration[
            flux_pulse_amplitude == flux_pulse_amplitude[0]
        ]

        figs[f"c{q_control}_t{q_target}_raw"] = make_subplots(
            rows=1,
            cols=1,
            horizontal_spacing=0.1,
            vertical_spacing=0.2,
            subplot_titles=(
                f"Qubit Target {q_target}",
                # f"Qubit Control {q_control}",
            ),
        )

        figs[f"c{q_control}_t{q_target}_raw"].add_trace(
            go.Heatmap(
                x=flux_pulse_duration,
                y=flux_pulse_amplitude,
                z=QubitTarget,
                name="QubitTarget",
                colorbar=dict(len=0.46, y=0.75),
            ),
            row=1,
            col=1,
        )

        figs[f"c{q_control}_t{q_target}_raw"].update_layout(
            xaxis_title="Pulse duration (s)",
            yaxis_title="Amplitude (a.u.)",
            # xaxis2_title="Pulse detuning (deg)",
            # yaxis2_title="Amplitude (a.u.)",
            title=f"Raw data",
        )

        # # Normalizing between -1 and 1, and getting simple phase
        # QubitTarget_matrix = (
        #     np.ones((len(amplitude_unique), len(duration_unique))) * np.nan
        # )
        # QubitTarget_OFF_matrix = (
        #     np.ones((len(amplitude_unique), len(duration_unique))) * np.nan
        # )

        # for i, t in enumerate(duration_unique):
        #     n = int(np.where(flux_pulse_duration == duration_unique[-1])[0][-1]) + 1
        #     idx = np.where(flux_pulse_duration[:n] == t)
        #     QubitTarget_OFF_matrix[
        #         : n // len(duration_unique), i
        #     ] = QubitTarget_OFF[idx]
        #     QubitTarget_matrix[: n // len(duration_unique), i] = QubitTarget[
        #         idx
        #     ]

        # figs[f"c{q_control}_t{q_target}_norm"] = make_subplots(
        #     rows=1,
        #     cols=2,
        #     horizontal_spacing=0.1,
        #     vertical_spacing=0.2,
        #     subplot_titles=(
        #         f"Qubit Target {q_target} ON",
        #         f"Qubit Target {q_target} OFF",
        #     ),
        # )

        # QubitTarget_norm = QubitTarget_matrix - 0.5
        # QubitTarget_norm = QubitTarget_norm / np.vstack(
        #     np.max(np.abs(QubitTarget_norm), axis=1)
        # )
        # QubitTarget_OFF_norm = QubitTarget_OFF_matrix - 0.5
        # QubitTarget_OFF_norm = QubitTarget_OFF_norm / np.vstack(
        #     np.max(np.abs(QubitTarget_OFF_norm), axis=1)
        # )

        # figs[f"c{q_control}_t{q_target}_norm"].add_trace(
        #     go.Heatmap(
        #         x=duration_unique,
        #         y=amplitude_unique,
        #         z=QubitTarget_OFF_norm,
        #         colorbar=dict(len=0.46, y=0.75),
        #         name="QubitControl OFF",
        #     ),
        #     row=1,
        #     col=1,
        # )

        # figs[f"c{q_control}_t{q_target}_norm"].add_trace(
        #     go.Heatmap(
        #         x=duration_unique,
        #         y=amplitude_unique,
        #         z=QubitTarget_norm,
        #         colorbar=dict(len=0.46, y=0.25),
        #         name="QubitControl ON",
        #     ),
        #     row=1,
        #     col=2,
        # )

        # figs[f"c{q_control}_t{q_target}_norm"].update_layout(
        #     uirevision="0",
        #     yaxis_title="Amplitude (a.u.)",
        #     xaxis_title="t_p",
        #     title=f"Control {q_control}, Target {q_target}",
        # )

        # # Calculating the phase
        # figs[f"c{q_control}_t{q_target}_phase"] = go.Figure()
        # figs[f"c{q_control}_t{q_target}_phase"].add_trace(
        #     go.Heatmap(
        #         x=duration_unique,
        #         y=amplitude_unique,
        #         z=np.rad2deg(
        #             np.arccos(QubitTarget_norm) - np.arccos(QubitTarget_OFF_norm)
        #         ),
        #         name="Phase arcos(ON) - arcos(OFF)",
        #     ),
        # )
        # figs[f"c{q_control}_t{q_target}_phase"].update_layout(
        #     uirevision="0",
        #     yaxis_title="Amplitude (a.u.)",
        #     xaxis_title="t_p",
        #     title=f"Control {q_control}, Target {q_target}",
        # )

        # # Calculating the phase from complex number
        # figs[f"c{q_control}_t{q_target}_phase_complex"] = go.Figure()
        # figs[f"c{q_control}_t{q_target}_phase_complex"].add_trace(
        #     go.Heatmap(
        #         x=duration_unique,
        #         y=amplitude_unique,
        #         z=np.rad2deg(
        #             np.angle(QubitTarget_matrix + 1j * QubitTarget_OFF_matrix)
        #         )
        #         % 180,
        #         name="Phase Complex ON + 1j * OFF",
        #     ),
        # )
        # figs[f"c{q_control}_t{q_target}_phase_complex"].update_layout(
        #     uirevision="0",
        #     yaxis_title="Amplitude (a.u.)",
        #     xaxis_title="t_p",
        #     title=f"Control {q_control}, Target {q_target}",
        # )

        # # Calculating the differance between ON and OFF
        # figs[f"c{q_control}_t{q_target}_diff"] = go.Figure()
        # figs[f"c{q_control}_t{q_target}_diff"].add_trace(
        #     go.Heatmap(
        #         x=flux_pulse_duration,  # duration_unique,
        #         y=flux_pulse_amplitude,  # amplitude_unique,
        #         z=QubitTarget_OFF
        #         - QubitTarget,  # QubitTarget_matrix - QubitTarget_OFF_matrix,
        #         name="Diff",
        #     ),
        # )
        # figs[f"c{q_control}_t{q_target}_diff"].update_layout(
        #     uirevision="0",
        #     yaxis_title="Amplitude (a.u.)",
        #     xaxis_title="t_p",
        #     title=f"Control {q_control}, Target {q_target}",
        # )

        # # Mean complex phase
        # figs[f"c{q_control}_t{q_target}_phase_complex_mean"] = go.Figure()
        # figs[f"c{q_control}_t{q_target}_phase_complex_mean"].add_trace(
        #     go.Scatter(
        #         x=amplitude_unique,
        #         y=np.nanmean(
        #             np.rad2deg(
        #                 np.arccos(QubitTarget_norm)
        #                 - np.arccos(QubitTarget_OFF_norm)
        #             ),
        #             axis=1,
        #         ),
        #         name="Phase Complex ON + 1j * OFF",
        #     ),
        # )
        # figs[f"c{q_control}_t{q_target}_phase_complex_mean"].update_layout(
        #     uirevision="0",
        #     yaxis_title="mean phi2Q (deg)",
        #     xaxis_title="Amplitude (a.u.)",
        #     title=f"Control {q_control}, Target {q_target}",
        # )

    return figs  # [f"c{q_control}_t{q_target}_raw"]
