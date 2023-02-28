import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.plots.utils import get_color, get_data_subfolders


def offset_amplitude_msr_phase(folder, routine, qubit, format):
    figures = []

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

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    fig = make_subplots(
        rows=len(subfolders),
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )
    report_n = 0
    fitting_report = ""
    fit_done = False
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                quantities={"amplitude": "dimensionless", "offset": "Hz"},
                options=["qubit", "iteration"],
            )

        try:
            data_fit = DataUnits.load_data(folder, subfolder, routine, format, "fit")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
            fit_done = True
        except:
            data_fit = Data(
                quantities=["Ec", "Ej", "f01", "alpha", "qubit", "iteration"]
            )

        iterations = data.df["iteration"].unique()
        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1

        averaged_data = (
            data.df.drop(columns=["qubit", "iteration"])
            .groupby(["offset", "amplitude"], as_index=False)
            .mean()
        )
        params = data_fit.df.drop(columns=["qubit", "iteration"]).groupby(
            ["Ec", "Ej", "f01", "alpha"], as_index=False
        )

        fig.add_trace(
            go.Heatmap(
                x=averaged_data["offset"].pint.to("MHz").pint.magnitude,
                y=averaged_data["amplitude"].pint.to("dimensionless").pint.magnitude,
                z=averaged_data["MSR"].pint.to("V").pint.magnitude,
                colorbar_x=0.45,
            ),
            row=1 + report_n,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("offset", "MHz"),
                y=data.get_values("amplitude", "dimensionless"),
                z=data.get_values("phase", "rad"),
                colorbar_x=1.0,
            ),
            row=1 + report_n,
            col=2,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Offset (MHz)", row=1 + report_n, col=2
        )
        fig.update_yaxes(
            title_text="Amplitude (dimensionless)", row=1 + report_n, col=2
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Offset (MHz)", row=1 + report_n, col=1
        )
        fig.update_yaxes(
            title_text="Amplitude (dimensionless)", row=1 + report_n, col=1
        )
        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )

        if fit_done:
            fitting_report = (
                fitting_report
                + (f"q{qubit}/r{report_n} | charging energy: {params['Ec']:.2f} ns<br>")
                + (
                    f"q{qubit}/r{report_n} | inductive energy: {params['Ej']:.2f} ns<br>"
                )
                + (
                    f"q{qubit}/r{report_n} | qubit frequency: {params['f01']:.2f} MHz<br>"
                )
                + (f"q{qubit}/r{report_n} | alpha: {params['alpha']:.2f} MHz<br>")
            )

        report_n += 1
    if report_n > 1:
        fig.update_traces(showscale=False)

    figures.append(fig)

    return figures, fitting_report
