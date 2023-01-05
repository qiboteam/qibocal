import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import lorenzian
from qibocal.plots.utils import get_color, get_data_subfolders


# Resonator and qubit spectroscopies
def frequency_msr_phase(folder, routine, qubit, format):

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
    report_n = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data_fast = DataUnits.load_data(
                folder, subfolder, routine, format, "fast_sweep_data"
            )
            data_fast.df = data_fast.df[data_fast.df["qubit"] == qubit]
        except:
            data_fast = DataUnits(
                quantities={"frequency": "Hz"}, options=["qubit", "iteration"]
            )
        try:
            data_precision = DataUnits.load_data(
                folder, subfolder, routine, format, f"precision_sweep_data"
            )
            data_precision.df = data_precision.df[data_precision.df["qubit"] == qubit]
        except:
            data_precision = DataUnits(
                quantities={"frequency": "Hz"}, options=["qubit", "iteration"]
            )
        try:
            data_fit = Data.load_data(folder, subfolder, routine, format, f"fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
        except:
            data_fit = Data(
                quantities=[
                    "popt0",
                    "popt1",
                    "popt2",
                    "popt3",
                    "label1",
                    "label2",
                    "label3",
                    "qubit",
                ]
            )

        for i, label, data in list(
            zip((0, 1), ("Fast", "Precision"), (data_fast, data_precision))
        ):
            iterations = data.df["iteration"].unique()
            frequencies = data.df["frequency"].pint.to("Hz").pint.magnitude.unique()
            if len(iterations) > 1:
                opacity = 0.3
            else:
                opacity = 1
            for iteration in iterations:
                iteration_data = data.df[data.df["iteration"] == iteration]
                fig.add_trace(
                    go.Scatter(
                        x=iteration_data["frequency"].pint.to("Hz").pint.magnitude,
                        y=iteration_data["MSR"].pint.to("uV").pint.magnitude,
                        marker_color=get_color(2 * report_n + i),
                        opacity=opacity,
                        name=f"q{qubit}/r{report_n}: {label}",
                        showlegend=not bool(iteration),
                        legendgroup=f"q{qubit}/r{report_n}: {label}",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=iteration_data["frequency"].pint.to("Hz").pint.magnitude,
                        y=iteration_data["phase"].pint.to("rad").pint.magnitude,
                        marker_color=get_color(2 * report_n + i),
                        opacity=opacity,
                        showlegend=False,
                        legendgroup=f"q{qubit}/r{report_n}: {label}",
                    ),
                    row=1,
                    col=2,
                )
            if len(iterations) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=data.df.groupby("frequency")["MSR"]
                        .mean()
                        .pint.to("uV")
                        .pint.magnitude,
                        marker_color=get_color(2 * report_n + i),
                        name=f"q{qubit}/r{report_n}: {label} Average",
                        showlegend=True,
                        legendgroup=f"q{qubit}/r{report_n}: {label} Average",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=data.df.groupby("frequency")["phase"]
                        .mean()
                        .pint.to("rad")
                        .pint.magnitude,
                        marker_color=get_color(2 * report_n + i),
                        showlegend=False,
                        legendgroup=f"q{qubit}/r{report_n}: {label} Average",
                    ),
                    row=1,
                    col=2,
                )

        if len(data_fast) > 0 and (qubit in data_fit.df["qubit"].values):
            freqrange = np.linspace(
                min(data_fast.get_values("frequency", "Hz")),
                max(data_fast.get_values("frequency", "Hz")),
                2 * len(data_fast),
            )
            params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                orient="records"
            )[0]
            fig.add_trace(
                go.Scatter(
                    x=freqrange,
                    y=lorenzian(
                        freqrange,
                        data_fit.get_values("popt0"),
                        data_fit.get_values("popt1"),
                        data_fit.get_values("popt2"),
                        data_fit.get_values("popt3"),
                    ),
                    name=f"q{qubit}/r{report_n} Fit",
                    line=go.scatter.Line(dash="dot"),
                    marker_color=get_color(4 * report_n + 2),
                ),
                row=1,
                col=1,
            )

            for param, value in params.items():
                if "freq" in param:
                    fitting_report = fitting_report + (
                        f"q{qubit}/r{report_n} {param}: {value:,.0f} Hz.<br>"
                    )
                elif "voltage" in param:
                    fitting_report = fitting_report + (
                        f"q{qubit}/r{report_n} {param}: {value:,.0f} uV.<br>"
                    )
            fitting_report += "<br>"
        report_n += 1

    fig.add_annotation(
        dict(
            font=dict(color="black", size=12),
            x=0,
            y=1.2,
            showarrow=False,
            text="<b>FITTING DATA</b>",
            font_family="Arial",
            font_size=20,
            textangle=0,
            xanchor="left",
            xref="paper",
            yref="paper",
            font_color="#5e9af1",
            hovertext=fitting_report,
        )
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (Hz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="Phase (rad)",
    )
    return fig


# Punchout
def frequency_attenuation_msr_phase(folder, routine, qubit, format):

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)

    fig = make_subplots(
        rows=len(subfolders),
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "Normalised MSR",
            "phase (rad)",
        ),
    )

    report_n = 0
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"frequency": "Hz", "attenuation": "dB"},
                options=["qubit", "iteration"],
            )

        iterations = data.df["iteration"].unique()
        frequencies = data.df["frequency"].pint.to("Hz").pint.magnitude.unique()
        attenuations = data.df["attenuation"].pint.to("dB").pint.magnitude.unique()
        averaged_data = (
            data.df.drop(columns=["qubit", "iteration"])
            .groupby(["frequency", "attenuation"], as_index=False)
            .mean()
        )

        def norm(x):
            x_mags = x.pint.to("V").pint.magnitude
            return (x_mags - np.min(x_mags)) / (np.max(x_mags) - np.min(x_mags))

        normalised_data = averaged_data.groupby(["attenuation"], as_index=False)[
            ["MSR"]
        ].transform(norm)

        fig.add_trace(
            go.Heatmap(
                x=averaged_data["frequency"].pint.to("Hz").pint.magnitude,
                y=averaged_data["attenuation"].pint.to("dB").pint.magnitude,
                z=normalised_data["MSR"],
                # z=averaged_data["MSR"].pint.to("V").pint.magnitude,
                colorbar_x=0.46,
            ),
            row=1 + report_n,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Frequency (Hz)", row=1 + report_n, col=1
        )
        fig.update_yaxes(title_text="Attenuation (dB)", row=1 + report_n, col=1)
        fig.add_trace(
            go.Heatmap(
                x=averaged_data["frequency"].pint.to("Hz").pint.magnitude,
                y=averaged_data["attenuation"].pint.to("dB").pint.magnitude,
                z=averaged_data["phase"].pint.to("rad").pint.magnitude,
                colorbar_x=1.01,
            ),
            row=1 + report_n,
            col=2,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Frequency (Hz)", row=1 + report_n, col=2
        )
        fig.update_yaxes(title_text="Attenuation (dB)", row=1 + report_n, col=2)
        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )
        report_n += 1
    if report_n > 1:
        fig.update_traces(showscale=False)
    return fig


# Resonator and qubit spectroscopies
def frequency_attenuation_msr_phase_cut(folder, routine, qubit, format):

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

    subfolders = get_data_subfolders(folder)
    report_n = 0
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"frequency": "Hz", "attenuation": "dB"},
                options=["qubit", "iteration"],
            )
        attenuations = data.df["attenuation"].unique()
        middle_attenuation = attenuations[len(attenuations) // 2]
        data.df = data.df[data.df["qubit"] == qubit][
            data.df["attenuation"] == middle_attenuation
        ].drop(columns=["attenuation"])

        iterations = data.df["iteration"].unique()
        frequencies = data.df["frequency"].pint.to("Hz").pint.magnitude.unique()
        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["frequency"].pint.to("Hz").pint.magnitude,
                    y=iteration_data["MSR"].pint.to("uV").pint.magnitude,
                    marker_color=get_color(report_n),
                    opacity=opacity,
                    name=f"q{qubit}/r{report_n} Attenuation: {middle_attenuation.to('dB').magnitude} dB",
                    showlegend=not bool(iteration),
                    legendgroup=f"q{qubit}/r{report_n}",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["frequency"].pint.to("Hz").pint.magnitude,
                    y=iteration_data["phase"].pint.to("rad").pint.magnitude,
                    marker_color=get_color(report_n),
                    opacity=opacity,
                    showlegend=False,
                    legendgroup=f"q{qubit}/r{report_n}",
                ),
                row=1,
                col=2,
            )
        if len(iterations) > 1:
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=data.df.groupby("frequency")["MSR"]
                    .mean()
                    .pint.to("uV")
                    .pint.magnitude, # pylint: disable=E1101
                    marker_color=get_color(report_n),
                    name=f"q{qubit}/r{report_n}: Average",
                    showlegend=True,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=data.df.groupby("frequency")["phase"]
                    .mean()
                    .pint.to("rad")
                    .pint.magnitude, # pylint: disable=E1101
                    marker_color=get_color(report_n),
                    showlegend=False,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=2,
            )

        report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (Hz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="Phase (rad)",
    )
    return fig


# Resonator spectroscopy flux
def frequency_flux_msr_phase(folder, routine, qubit, format):

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)

    report_n = 0
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"frequency": "Hz", "current": "A"},
                options=["qubit", "fluxline", "iteration"],
            )

        iterations = data.df["iteration"].unique()
        fluxlines = data.df["fluxline"].unique()
        frequencies = data.df["frequency"].pint.to("Hz").pint.magnitude.unique()
        currents = data.df["current"].pint.to("A").pint.magnitude.unique()

        if len(fluxlines) > 1:
            fig = make_subplots(
                rows=len(subfolders),
                cols=len(fluxlines),
                horizontal_spacing=0.1,
                vertical_spacing=0.1,
                subplot_titles=tuple(
                    [f"MSR [V] - fluxline {fluxline}" for fluxline in fluxlines]
                ),
            )

            for fluxline_n, fluxline in enumerate(fluxlines):
                fluxline_df = data.df[data.df["fluxline"] == fluxline]
                fluxline_df = (
                    fluxline_df.drop(columns=["qubit", "fluxline", "iteration"])
                    .groupby(["frequency", "current"], as_index=False)
                    .mean()
                )

                fig.add_trace(
                    go.Heatmap(
                        x=fluxline_df["frequency"].pint.to("Hz").pint.magnitude,
                        y=fluxline_df["current"].pint.to("A").pint.magnitude,
                        z=fluxline_df["MSR"].pint.to("V").pint.magnitude,
                        showscale=False,
                    ),
                    row=1 + report_n,
                    col=1 + fluxline_n,
                )
                fig.update_xaxes(
                    title_text=f"q{qubit}/r{report_n}: Frequency (GHz)",
                    row=1 + report_n,
                    col=1 + fluxline_n,
                )

        elif len(fluxlines) == 1:
            fig = make_subplots(
                rows=len(subfolders),
                cols=2,
                horizontal_spacing=0.1,
                vertical_spacing=0.1,
                subplot_titles=(
                    f"MSR [V] - fluxline {fluxlines[0]}",
                    f"Phase [rad] - fluxline {fluxlines[0]}",
                ),
            )

            fluxline_df = data.df[data.df["fluxline"] == fluxlines[0]]
            fluxline_df = (
                fluxline_df.drop(columns=["qubit", "fluxline", "iteration"])
                .groupby(["frequency", "current"], as_index=False)
                .mean()
            )

            fig.add_trace(
                go.Heatmap(
                    x=fluxline_df["frequency"].pint.to("Hz").pint.magnitude,
                    y=fluxline_df["current"].pint.to("A").pint.magnitude,
                    z=fluxline_df["MSR"].pint.to("V").pint.magnitude,
                    colorbar_x=0.46,
                ),
                row=1 + report_n,
                col=1,
            )
            fig.update_xaxes(
                title_text=f"q{qubit}/r{report_n}: Frequency (Hz)",
                row=1 + report_n,
                col=1,
            )

            fig.add_trace(
                go.Heatmap(
                    x=fluxline_df["frequency"].pint.to("Hz").pint.magnitude,
                    y=fluxline_df["current"].pint.to("A").pint.magnitude,
                    z=fluxline_df["phase"].pint.to("rad").pint.magnitude,
                    colorbar_x=1.01,
                ),
                row=1 + report_n,
                col=2,
            )
            fig.update_xaxes(
                title_text=f"q{qubit}/r{report_n}: Frequency (Hz)",
                row=1 + report_n,
                col=2,
            )
            fig.update_yaxes(title_text="Current (A)", row=1 + report_n, col=2)

        fig.update_yaxes(title_text="Current (A)", row=1 + report_n, col=1)
        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )

        report_n += 1
    if report_n > 1:
        fig.update_traces(showscale=False)
    return fig


# Dispersive shift
def dispersive_frequency_msr_phase(folder, routine, qubit, format):

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
    report_n = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data_0 = DataUnits.load_data(folder, subfolder, routine, format, f"data_0")
            data_0.df = data_0.df[data_0.df["qubit"] == qubit]
        except:
            data_0 = DataUnits(
                name=f"data_0",
                quantities={"frequency": "Hz"},
                options=["qubit", "shifted", "iteration"],
            )
        try:
            data_1 = DataUnits.load_data(folder, subfolder, routine, format, f"data_1")
            data_1.df = data_1.df[data_1.df["qubit"] == qubit]
        except:
            data_1 = DataUnits(
                name=f"data_1",
                quantities={"frequency": "Hz"},
                options=["qubit", "shifted", "iteration"],
            )

        try:
            fit_data_0 = Data.load_data(
                folder, subfolder, routine, format, f"fit_data_0"
            )
        except:
            fit_data_0 = Data(
                quantities=[
                    "popt0",
                    "popt1",
                    "popt2",
                    "popt3",
                    "label1",
                    "label2",
                    "label3",
                ]
            )

        try:
            fit_data_1 = Data.load_data(
                folder, subfolder, routine, format, f"fit_data_1"
            )
        except:
            fit_data_1 = Data(
                quantities=[
                    "popt0",
                    "popt1",
                    "popt2",
                    "popt3",
                    "label1",
                    "label2",
                    "label3",
                ]
            )
        resonator_freqs = {}
        for i, label, data, data_fit in list(
            zip(
                (0, 1),
                ("Spectroscopy", "Shifted spectroscopy"),
                (data_0, data_1),
                (fit_data_0, fit_data_1),
            )
        ):
            iterations = data.df["iteration"].unique()
            frequencies = data.df["frequency"].pint.to("Hz").pint.magnitude.unique()
            if len(iterations) > 1:
                opacity = 0.3
            else:
                opacity = 1
            for iteration in iterations:
                iteration_data = data.df[data.df["iteration"] == iteration]
                fig.add_trace(
                    go.Scatter(
                        x=iteration_data["frequency"].pint.to("Hz").pint.magnitude,
                        y=iteration_data["MSR"].pint.to("uV").pint.magnitude,
                        marker_color=get_color(2 * report_n + i),
                        opacity=opacity,
                        name=f"q{qubit}/r{report_n}: {label}",
                        showlegend=not bool(iteration),
                        legendgroup=f"q{qubit}/r{report_n}: {label}",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=iteration_data["frequency"].pint.to("Hz").pint.magnitude,
                        y=iteration_data["phase"].pint.to("rad").pint.magnitude,
                        marker_color=get_color(2 * report_n + i),
                        opacity=opacity,
                        showlegend=False,
                        legendgroup=f"q{qubit}/r{report_n}: {label}",
                    ),
                    row=1,
                    col=2,
                )
            if len(iterations) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=data.df.groupby("frequency")["MSR"]
                        .mean()
                        .pint.to("uV")
                        .pint.magnitude,
                        marker_color=get_color(2 * report_n + i),
                        name=f"q{qubit}/r{report_n}: {label} Average",
                        showlegend=True,
                        legendgroup=f"q{qubit}/r{report_n}: {label} Average",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=data.df.groupby("frequency")["phase"]
                        .mean()
                        .pint.to("rad")
                        .pint.magnitude,
                        marker_color=get_color(2 * report_n + i),
                        showlegend=False,
                        legendgroup=f"q{qubit}/r{report_n}: {label} Average",
                    ),
                    row=1,
                    col=2,
                )

            if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
                freqrange = np.linspace(
                    min(data.get_values("frequency", "Hz")),
                    max(data.get_values("frequency", "Hz")),
                    2 * len(data),
                )
                params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                    orient="records"
                )[0]
                fig.add_trace(
                    go.Scatter(
                        x=freqrange,
                        y=lorenzian(
                            freqrange,
                            data_fit.get_values("popt0"),
                            data_fit.get_values("popt1"),
                            data_fit.get_values("popt2"),
                            data_fit.get_values("popt3"),
                        ),
                        name=f"q{qubit}/r{report_n}: {label} Fit",
                        line=go.scatter.Line(dash="dot"),
                        marker_color="rgb(255, 130, 67)",
                    ),
                    row=1,
                    col=1,
                )

                if "resonator_freq" in params:
                    resonator_freqs[label] = params["resonator_freq"]

                for param, value in params.items():
                    if "freq" in param:
                        fitting_report = fitting_report + (
                            f"q{qubit}/r{report_n} {label} {param}: {value:,.0f} Hz.<br>"
                        )
                    elif "voltage" in param:
                        fitting_report = fitting_report + (
                            f"q{qubit}/r{report_n} {label} {param}: {value:,.0f} uV.<br>"
                        )
        if ("Spectroscopy" in resonator_freqs) and (
            "Shifted spectroscopy" in resonator_freqs
        ):
            frequency_shift = (
                resonator_freqs["Shifted spectroscopy"]
                - resonator_freqs["Spectroscopy"]
            )
            fitting_report = fitting_report + (
                f"q{qubit}/r{report_n} Frequency shift: {frequency_shift:,.0f} Hz.<br>"
            )
        fitting_report += "<br>"
        report_n += 1

    fig.add_annotation(
        dict(
            font=dict(color="black", size=12),
            x=0,
            y=1.2,
            showarrow=False,
            text="<b>FITTING DATA</b>",
            font_family="Arial",
            font_size=20,
            textangle=0,
            xanchor="left",
            xref="paper",
            yref="paper",
            font_color="#5e9af1",
            hovertext=fitting_report,
        )
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (Hz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="Phase (rad)",
    )
    return fig
