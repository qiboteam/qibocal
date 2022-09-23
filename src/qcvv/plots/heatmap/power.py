# -*- coding: utf-8 -*-
import os.path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qcvv.data import Dataset


def amplitude_attenuation_msr_phase(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            y=data.get_values("attenuation", "dB"),
            x=data.get_values("amplitude", "dimensionless"),
            z=data.get_values("MSR", "V"),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            y=data.get_values("attenuation", "dB"),
            x=data.get_values("amplitude", "dimensionless"),
            z=data.get_values("phase", "rad"),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="amplitude (dimensionless)",
        yaxis_title="attenuation (dB)",
        xaxis2_title="amplitude (dimensionless)",
        yaxis2_title="attenuation (dB)",
    )
    return fig
