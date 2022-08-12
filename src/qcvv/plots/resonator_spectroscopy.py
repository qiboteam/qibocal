# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class resonator_spectroscopy_attenuation:
    @staticmethod
    def frequency_vs_attenuation(data):
        fig = go.Figure(
            data=go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("attenuation", "dB"),
                z=data.get_values("MSR", "V"),
            )
        )
        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title="Frequency (GHz)",
            yaxis_title="Attenuation (dB)",
        )
        return fig

    @staticmethod
    def msr_vs_frequency(data):
        plot1d_attenuation = 30  # attenuation value to use for 1D frequency vs MSR plot

        fig = go.Figure()
        # index data on a specific attenuation value
        smalldf = data.df[
            data.get_values("attenuation", "dB") == plot1d_attenuation
        ].copy()
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
