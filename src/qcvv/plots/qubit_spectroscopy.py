# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class qubit_spectroscopy:
    @staticmethod
    def frequency_vs_msr_phase(data):
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
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("MSR", "uV"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("phase", "deg"),
            ),
            row=1,
            col=2,
        )
        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title="Frequency (GHz)",
            yaxis_title="MSR (uV)",
            xaxis2_title="Frequency (GHz)",
            yaxis2_title="Phase (deg)",
        )
        return fig

class qubit_spectroscopy_flux:
    @staticmethod
    def frequency_vs_current(data):
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
                y=data.get_values("current", "A"),
                z=data.get_values("MSR", "V"),
                colorbar_x=0.45,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("current", "A"),
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
            yaxis_title="Current (A)",
            xaxis2_title="Frequency (GHz)",
            yaxis2_title="Current (A)",
        )
        return fig
