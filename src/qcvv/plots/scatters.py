# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qcvv.data import Dataset


class frequency_msr_phase__fast_precision:
    @staticmethod
    def frequency_vs_msr_phase(folder, routine, qubit, format):
        data_fast = Dataset.load_data(folder, routine, format, f"fast_sweep_q{qubit}")
        data_precision = Dataset.load_data(
            folder, routine, format, f"precision_sweep_q{qubit}"
        )
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
            go.Scatter(
                x=data_fast.get_values("frequency", "GHz"),
                y=data_fast.get_values("MSR", "uV"),
                name="Fast",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data_fast.get_values("frequency", "GHz"),
                y=data_fast.get_values("phase", "deg"),
                name="Fast",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=data_precision.get_values("frequency", "GHz"),
                y=data_precision.get_values("MSR", "uV"),
                name="Precision",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data_precision.get_values("frequency", "GHz"),
                y=data_precision.get_values("phase", "deg"),
                name="Precision",
            ),
            row=1,
            col=2,
        )
        fig.update_layout(
            showlegend=True,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title="Frequency (GHz)",
            yaxis_title="MSR (uV)",
            xaxis2_title="Frequency (GHz)",
            yaxis2_title="Phase (deg)",
        )
        return fig


class frequency_attenuation_msr_phase:
    @staticmethod
    def frequency_vs_attenuation(folder, routine, qubit, format):
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
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
    def msr_vs_frequency(folder, routine, qubit, format):
        data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
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
