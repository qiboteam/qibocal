# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_values(df, quantity, unit):
    return df[quantity].pint.to(unit).pint.magnitude


def resonator_spectroscopy_attenuation(data):
    plot1d_attenuation = 30  # attenuation value to use for 1D frequency vs MSR plot

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (deg)",
            f"Attenuation = {plot1d_attenuation}dB",
        ),
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
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)

    # index data on a specific attenuation value
    smalldf = data.df[
        get_values(data.df, "attenuation", "dB") == plot1d_attenuation
    ].copy()
    # split multiple software averages to different datasets
    datasets = []
    while len(smalldf):
        datasets.append(smalldf.drop_duplicates("frequency"))
        smalldf.drop(datasets[-1].index, inplace=True)
        fig.add_trace(
            go.Scatter(
                x=get_values(datasets[-1], "frequency", "GHz"),
                y=get_values(datasets[-1], "MSR", "V"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=1000,
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="Attenuation (dB)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Attenuation (dB)",
        xaxis3_title="Frequency (GHz)",
        yaxis3_title="MSR (V)",
    )
    return fig
