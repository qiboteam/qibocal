import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import DataUnits


# For calibrate qubit states
def exc_gnd(folder, routine, qubit, format):

    try:
        data_exc = DataUnits.load_data(folder, routine, format, f"data_exc_q{qubit}")
    except:
        data_exc = DataUnits(quantities={"iteration": "dimensionless"})

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("Calibrate qubit states",),
    )

    fig.add_trace(
        go.Scatter(
            x=data_exc.get_values("i", "V"),
            y=data_exc.get_values("q", "V"),
            name="exc_state",
            mode="markers",
            marker=dict(size=3, color="lightcoral"),
        ),
        row=1,
        col=1,
    )

    try:
        data_gnd = DataUnits.load_data(folder, routine, format, f"data_gnd_q{qubit}")
    except:
        data_gnd = DataUnits(quantities={"iteration": "dimensionless"})

    fig.add_trace(
        go.Scatter(
            x=data_gnd.get_values("i", "V"),
            y=data_gnd.get_values("q", "V"),
            name="gnd state",
            mode="markers",
            marker=dict(size=3, color="skyblue"),
        ),
        row=1,
        col=1,
    )

    i_exc = data_exc.get_values("i", "V")
    q_exc = data_exc.get_values("q", "V")

    i_mean_exc = i_exc.mean()
    q_mean_exc = q_exc.mean()
    iq_mean_exc = complex(i_mean_exc, q_mean_exc)
    mod_iq_exc = abs(iq_mean_exc) * 1e6

    fig.add_trace(
        go.Scatter(
            x=[i_mean_exc],
            y=[q_mean_exc],
            name=f" state1_voltage: {mod_iq_exc} <br> mean_state1: {iq_mean_exc}",
            mode="markers",
            marker=dict(size=10, color="red"),
        ),
        row=1,
        col=1,
    )

    i_gnd = data_gnd.get_values("i", "V")
    q_gnd = data_gnd.get_values("q", "V")

    i_mean_gnd = i_gnd.mean()
    q_mean_gnd = q_gnd.mean()
    iq_mean_gnd = complex(i_mean_gnd, q_mean_gnd)
    mod_iq_gnd = abs(iq_mean_gnd) * 1e6

    fig.add_trace(
        go.Scatter(
            x=[i_mean_gnd],
            y=[q_mean_gnd],
            name=f" state0_voltage: {mod_iq_gnd} <br> mean_state0: {iq_mean_gnd}",
            mode="markers",
            marker=dict(size=10, color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="i (V)",
        yaxis_title="q (V)",
        width=1000,
    )

    return fig
