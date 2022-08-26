# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qcvv.fitting.utils import lorenzian


class resonator_spectroscopy:
    @staticmethod
    def frequency_vs_attenuation(data):
        fig = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles=(
                "MSR (V)",
                "phase (deg)",
            ),
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("attenuation", "dB"),
                z=data.get_values("MSR", "V"),
                colorbar_x=0.45,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("attenuation", "dB"),
                z=data.get_values("phase", "deg"),
                colorbar_x=1.0,
            ),
            row=1,
            col=2,
        )
        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title="Frequency (GHz)",
            yaxis_title="Attenuation (dB)",
            xaxis2_title="Frequency (GHz)",
            yaxis2_title="Attenuation (dB)",
        )
        return fig

    @staticmethod
    def msr_vs_frequency(data):
        # plot1d_attenuation = 30  # attenuation value to use for 1D frequency vs MSR plot

        fig = go.Figure()
        # index data on a specific attenuation value
        smalldf = data.df
        # [
        #     data.get_values("attenuation", "dB") == plot1d_attenuation
        # ].copy()
        # split multiple software averages to different datasets
        datasets = []
        while len(smalldf):
            datasets.append(smalldf.drop_duplicates("frequency"))
            smalldf.drop(datasets[-1].index, inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=datasets[-1]["frequency"].pint.to("GHz").pint.magnitude,
                    y=datasets[-1]["MSR"].pint.to("V").pint.magnitude,
                ),
            )

        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting,
            xaxis_title="Frequency (GHz)",
            yaxis_title="MSR (V)",
        )
        return fig

    @staticmethod
    def fit(data, params, fit):

        x = data.get_values("frequency", "Hz")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                name="Data",
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("MSR", "V"),
            ),
        )

        if fit is not None:
            fig.add_trace(
                go.Scatter(
                    name="Fit",
                    x=data.get_values("frequency", "GHz"),
                    y=lorenzian(x, **fit),
                    line=go.scatter.Line(dash="dot"),
                ),
            )

        fig.update_layout(
            showlegend=True,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title="Frequency (GHz)",
            yaxis_title="MSR (V)",
        )
        if fit is not None:
            fig.add_annotation(
                dict(
                    font=dict(color="black", size=16),
                    x=0,
                    y=-0.2,
                    showarrow=False,
                    text=f"The estimated resonator frequency is {int(params[0])} Hz.",
                    textangle=0,
                    xanchor="left",
                    xref="paper",
                    yref="paper",
                )
            )
            fig.add_annotation(
                dict(
                    font=dict(color="black", size=16),
                    x=0,
                    y=-0.30,
                    showarrow=False,
                    text=f"The estimated peak voltage is {int(params[1])} uV.",
                    textangle=0,
                    xanchor="left",
                    xref="paper",
                    yref="paper",
                )
            )
        return fig
