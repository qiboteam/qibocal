# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_values(df, quantity, unit):
    return df[quantity].pint.to(unit).pint.magnitude


def resonator_spectroscopy_attenuation(data):
    trace3_att = 30

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)", "phase (deg)", f"Attenuation = {trace3_att}dB"),
        specs=[[{}, {}], [{"colspan": 2}, None]],
    )

    trace1 = go.Heatmap(
        x=get_values(data.df, "frequency", "GHz"),
        y=get_values(data.df, "attenuation", "dB"),
        z=get_values(data.df, "MSR", "V"),
        colorbar_x=0.45,
        colorbar_y=0.78,
        colorbar_len=0.45,
    )
    trace2 = go.Heatmap(
        x=get_values(data.df, "frequency", "GHz"),
        y=get_values(data.df, "attenuation", "dB"),
        z=get_values(data.df, "phase", "deg"),
        colorbar_x=1.0,
        colorbar_y=0.78,
        colorbar_len=0.45,
    )

    smalldf = data.df[get_values(data.df, "attenuation", "dB") == 30]
    trace3 = go.Scatter(
        x=get_values(smalldf, "frequency", "GHz"), y=get_values(smalldf, "MSR", "V")
    )

    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)
    fig.add_trace(trace3, row=2, col=1)
    fig.update_layout(
        height=1000,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="Attenuation (dB)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Attenuation (dB)",
        xaxis3_title="Frequency (GHz)",
        yaxis3_title="MSR (V)",
    )
    return fig
