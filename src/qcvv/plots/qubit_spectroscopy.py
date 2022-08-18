# -*- coding: utf-8 -*-
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qcvv.data import Dataset


class qubit_spectroscopy:
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
