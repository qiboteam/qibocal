import os
from functools import partial

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import (
    freq_q_mathieu,
    freq_q_transmon,
    freq_r_mathieu,
    freq_r_transmon,
    image_to_curve,
    line,
    lorenzian,
)
from qibocal.plots.utils import get_color, get_data_subfolders, load_data


# Resonator and qubit spectroscopies
def frequency_msr_phase(folder, routine, qubit, format):
    figures = []

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (uV)",
            "phase (rad)",
        ),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                quantities={"frequency": "Hz"}, options=["qubit", "iteration"]
            )
        try:
            data_fit = load_data(folder, subfolder, routine, format, "fits")
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

        data.df = data.df.drop(columns=["i", "q", "qubit"])
        iterations = data.df["iteration"].unique()
        frequencies = data.df["frequency"].unique() * 1e-9

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["frequency"] * 1e-9,
                    y=iteration_data["MSR"] * 1e6,
                    marker_color=get_color(2 * report_n),
                    opacity=opacity,
                    name=f"q{qubit}/r{report_n}: Data",
                    showlegend=not bool(iteration),
                    legendgroup=f"q{qubit}/r{report_n}: Data",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["frequency"] * 1e-9,
                    y=iteration_data["phase"],
                    marker_color=get_color(2 * report_n),
                    opacity=opacity,
                    showlegend=False,
                    legendgroup=f"q{qubit}/r{report_n}: Data",
                ),
                row=1,
                col=2,
            )
        if len(iterations) > 1:
            data.df = data.df.drop(columns=["iteration"])  # pylint: disable=E1101
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=data.df.groupby("frequency")[  # pylint: disable=E1101
                        "MSR"
                    ].mean()
                    * 1e6,
                    marker_color=get_color(2 * report_n),
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
                    y=data.df.groupby("frequency")[  # pylint: disable=E1101
                        "phase"
                    ].mean(),
                    marker_color=get_color(2 * report_n),
                    showlegend=False,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=2,
            )

        if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
            freqrange = np.linspace(
                min(frequencies),
                max(frequencies),
                2 * len(frequencies),
            )
            params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                orient="records"
            )[0]

            fig.add_trace(
                go.Scatter(
                    x=freqrange,
                    y=lorenzian(
                        freqrange,
                        float(data_fit.df["popt0"]),
                        float(data_fit.df["popt1"]),
                        float(data_fit.df["popt2"]),
                        float(data_fit.df["popt3"]),
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
                        f"q{qubit}/r{report_n} | {param}: {value:,.0f} Hz.<br>"
                    )
                elif "voltage" in param:
                    fitting_report = fitting_report + (
                        f"q{qubit}/r{report_n} | {param}: {value:,.0f} uV.<br>"
                    )
            fitting_report += "<br>"
        report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (Hz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="Phase (rad)",
    )

    figures.append(fig)

    return figures, fitting_report


# Punchout
def frequency_attenuation_msr_phase(folder, routine, qubit, format):
    figures = []
    fitting_report = "No fitting data"

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
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"frequency": "Hz", "attenuation": "dB"},
                options=["qubit", "iteration"],
            )

        iterations = data.df["iteration"].unique()
        frequencies = data.df["frequency"].unique()
        attenuations = data.df["attenuation"].unique()
        averaged_data = (
            data.df.drop(columns=["i", "q", "qubit", "iteration"])
            .groupby(["frequency", "attenuation"], as_index=False)
            .mean()
        )

        def norm(x_mags):
            return (x_mags - np.min(x_mags)) / (np.max(x_mags) - np.min(x_mags))

        normalised_data = averaged_data.groupby(["attenuation"], as_index=False)[
            ["MSR"]
        ].transform(norm)

        fig.add_trace(
            go.Heatmap(
                x=averaged_data["frequency"],
                y=averaged_data["attenuation"],
                z=normalised_data["MSR"],
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
                x=averaged_data["frequency"],
                y=averaged_data["attenuation"],
                z=averaged_data["phase"],
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

    figures.append(fig)

    return figures, fitting_report


# Punchout
def frequency_amplitude_msr_phase(folder, routine, qubit, format):
    figures = []
    fitting_report = "No fitting data"
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
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"frequency": "Hz", "amplitude": "dimensionless"},
                options=["qubit", "iteration"],
            )

        iterations = data.df["iteration"].unique()
        frequencies = data.df["frequency"].pint.to("Hz").pint.magnitude.unique()
        amplitudes = (
            data.df["amplitude"].pint.to("dimensionless").pint.magnitude.unique()
        )
        averaged_data = (
            data.df.drop(columns=["qubit", "iteration"])
            .groupby(["frequency", "amplitude"], as_index=False)
            .mean()
        )

        def norm(x):
            x_mags = x.pint.to("V").pint.magnitude
            return (x_mags - np.min(x_mags)) / (np.max(x_mags) - np.min(x_mags))

        normalised_data = averaged_data.groupby(["amplitude"], as_index=False)[
            ["MSR"]
        ].transform(norm)

        fig.add_trace(
            go.Heatmap(
                x=averaged_data["frequency"].pint.to("Hz").pint.magnitude,
                y=averaged_data["amplitude"].pint.to("dimensionless").pint.magnitude,
                z=normalised_data["MSR"],
                colorbar_x=0.46,
            ),
            row=1 + report_n,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Frequency (Hz)", row=1 + report_n, col=1
        )
        fig.update_yaxes(title_text="Amplitude", row=1 + report_n, col=1)
        fig.add_trace(
            go.Heatmap(
                x=averaged_data["frequency"].pint.to("Hz").pint.magnitude,
                y=averaged_data["amplitude"].pint.to("dimensionless").pint.magnitude,
                z=averaged_data["phase"].pint.to("rad").pint.magnitude,
                colorbar_x=1.01,
            ),
            row=1 + report_n,
            col=2,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Frequency (Hz)", row=1 + report_n, col=2
        )
        fig.update_yaxes(title_text="Amplitude", row=1 + report_n, col=2)
        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )
        report_n += 1
    if report_n > 1:
        fig.update_traces(showscale=False)

    figures.append(fig)

    return figures, fitting_report


# Resonator and qubit spectroscopies
def frequency_attenuation_msr_phase_cut(folder, routine, qubit, format):
    figures = []
    fitting_report = "No fitting data"

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
            data = load_data(folder, subfolder, routine, format, "data")
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
        frequencies = data.df["frequency"].unique()
        data.df = data.df.drop(columns=["i", "q", "qubit"])

        if len(iterations) > 1:
            opacity = 0.3
        else:
            opacity = 1
        for iteration in iterations:
            iteration_data = data.df[data.df["iteration"] == iteration]
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["frequency"],
                    y=iteration_data["MSR"] * 1e6,
                    marker_color=get_color(report_n),
                    opacity=opacity,
                    name=f"q{qubit}/r{report_n} Attenuation: {middle_attenuation} dB",
                    showlegend=not bool(iteration),
                    legendgroup=f"q{qubit}/r{report_n}",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=iteration_data["frequency"],
                    y=iteration_data["phase"],
                    marker_color=get_color(report_n),
                    opacity=opacity,
                    showlegend=False,
                    legendgroup=f"q{qubit}/r{report_n}",
                ),
                row=1,
                col=2,
            )
        if len(iterations) > 1:
            data.df = data.df.drop(columns=["iteration"])  # pylint: disable=E1101
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=data.df.groupby("frequency")[  # pylint: disable=E1101
                        "MSR"
                    ].mean()
                    * 1e6,
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
                    y=data.df.groupby("frequency")[  # pylint: disable=E1101
                        "phase"
                    ].mean(),  # pylint: disable=E1101
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

    figures.append(fig)

    return figures, fitting_report


# Resonator and qubit spectroscopies
def frequency_amplitude_msr_phase_cut(folder, routine, qubit, format):
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
    fitting_report = "No fitting data"

    subfolders = get_data_subfolders(folder)
    report_n = 0
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"frequency": "Hz", "amplitude": "dimensionless"},
                options=["qubit", "iteration"],
            )
        amplitudes = data.df["amplitude"].unique()
        middle_amplitude = amplitudes[len(amplitudes) // 2]
        data.df = data.df[data.df["qubit"] == qubit][
            data.df["amplitude"] == middle_amplitude
        ].drop(columns=["amplitude"])

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
                    name=f"q{qubit}/r{report_n} Amplitude: {middle_amplitude.to('amplitude').magnitude}",
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
                    y=data.df.groupby("frequency")["MSR"]  # pylint: disable=E1101
                    .mean()
                    .pint.to("uV")
                    .pint.magnitude,  # pylint: disable=E1101
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
                    y=data.df.groupby("frequency")["phase"]  # pylint: disable=E1101
                    .mean()
                    .pint.to("rad")
                    .pint.magnitude,  # pylint: disable=E1101
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
    return fig, fitting_report


# Resonator spectroscopy flux
def frequency_flux_msr_phase(folder, routine, qubit, format, method):
    figures = []
    fitting_report = ""

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)

    report_n = 0
    for subfolder in subfolders:
        try:
            data = load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"frequency": "Hz", "bias": "V"},
                options=["qubit", "fluxline", "iteration"],
            )

        try:
            data_fit = load_data(folder, subfolder, routine, format, "fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
        except:
            data_fit = Data(
                quantities=[
                    "curr_sp",
                    "xi",
                    "d",
                    "g",
                    "f_rh",
                    "f_qs",
                    "f_rs",
                    "f_offset",
                    "C_ii",
                    "qubit",
                    "fluxline",
                    "popt0",
                    "popt1",
                    "type",
                ]
            )
            fitting_report = "No fitting data"

        iterations = data.df["iteration"].unique()
        fluxlines = data.df["fluxline"].unique()
        frequencies = data.df["frequency"].unique()

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
        else:
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

        for fluxline_n, fluxline in enumerate(fluxlines):
            fluxline_df = data.df[data.df["fluxline"] == fluxline]
            fluxline_df = fluxline_df.drop(
                columns=["i", "q", "qubit", "fluxline", "iteration"]
            )

            fluxline_df = fluxline_df.groupby(
                ["frequency", "bias"], as_index=False
            ).mean()

            frequencies, biases = image_to_curve(
                fluxline_df["frequency"], fluxline_df["bias"], fluxline_df["MSR"] * 1e6
            )

            if len(fluxlines) > 1:
                fig.add_trace(
                    go.Heatmap(
                        x=fluxline_df["frequency"],
                        y=fluxline_df["bias"],
                        z=fluxline_df["MSR"] * 1e6,
                        showscale=False,
                    ),
                    row=1 + report_n,
                    col=1 + fluxline_n,
                )
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=biases,
                        mode="markers",
                        marker_color="green",
                    ),
                    row=1 + report_n,
                    col=1 + fluxline_n,
                )
                fig.update_xaxes(
                    title_text=f"q{qubit}/r{report_n}: Frequency (GHz)",
                    row=1 + report_n,
                    col=1 + fluxline_n,
                )
            else:
                fig.add_trace(
                    go.Heatmap(
                        x=fluxline_df["frequency"],
                        y=fluxline_df["bias"],
                        z=fluxline_df["MSR"] * 1e6,
                        colorbar_x=0.46,
                    ),
                    row=1 + report_n,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=biases,
                        mode="markers",
                        marker_color="green",
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
                        x=fluxline_df["frequency"],
                        y=fluxline_df["bias"],
                        z=fluxline_df["phase"],
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
                fig.update_yaxes(title_text="Bias (V)", row=1 + report_n, col=2)

            if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
                biases_fit = np.linspace(np.min(biases), np.max(biases), 100)
                params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
                    orient="records"
                )[0]
                if all(value != 0 for key, value in params.items() if key != "qubit"):
                    if params["type"] == 1:
                        popt = [params["curr_sp"], params["xi"], params["d"]]
                        if method == "resonator":
                            f = freq_r_transmon
                            popt.extend(
                                (params["f_q/f_rh"], params["g"], params["f_rh"])
                            )
                        elif method == "qubit":
                            f = freq_q_transmon
                            popt.append(params["f_qs"])
                    elif params["type"] == 2:
                        popt = [
                            params["curr_sp"],
                            params["xi"],
                            params["d"],
                            params["Ec"],
                            params["Ej"],
                        ]
                        if method == "resonator":
                            f = freq_r_mathieu
                            popt[:0] = [
                                params["f_rh"],
                                params["g"],
                            ]
                        elif method == "qubit":
                            f = freq_q_mathieu
                    else:
                        f = line
                        popt = [params["popt0"], params["popt1"]]
                    frequencies_fit = f(biases_fit, *popt)
                    fig.add_trace(
                        go.Scatter(
                            x=frequencies_fit,
                            y=biases_fit,
                            marker_color="red",
                        ),
                        row=1 + report_n,
                        col=1,
                    )
                    title_text = ""
                    if params["type"] == 1 or params["type"] == 2:
                        if method == "resonator":
                            title_text += f"q{qubit}/r{report_n} | resonator frequency at sweet spot: {params['f_rs']} Hz.<br>"
                            title_text += f"q{qubit}/r{report_n} | bias at sweet spot: {params['curr_sp']} V.<br>"
                            title_text += f"q{qubit}/r{report_n} | readout coupling: {params['g']} Hz.<br>"
                            title_text += f"q{qubit}/r{report_n} | resonator frequency at high power: {params['f_rh']} Hz.<br>"
                            title_text += f"q{qubit}/r{report_n} | qubit frequency at sweet spot: {params['f_qs']} Hz.<br>"
                        elif method == "qubit":
                            title_text += f"q{qubit}/r{report_n} | qubit frequency at sweet spot: {params['f_qs']} Hz.<br>"
                            title_text += f"q{qubit}/r{report_n} | bias at sweet spot: {params['curr_sp']} V.<br>"
                        title_text += (
                            f"q{qubit}/r{report_n} | asymmetry: {params['d']}.<br>"
                        )
                        title_text += f"q{qubit}/r{report_n} | frequency offset: {params['f_offset']} Hz.<br>"
                        title_text += (
                            f"q{qubit}/r{report_n} | C_ii: {params['C_ii']} Hz/V.<br>"
                        )
                        if params["type"] == 2:
                            title_text += (
                                f"q{qubit}/r{report_n} | Ec: {params['Ec']} Hz.<br>"
                            )
                            title_text += (
                                f"q{qubit}/r{report_n} | Ej: {params['Ej']} Hz.<br>"
                            )
                    else:
                        title_text += (
                            f"q{qubit}/r{report_n} | C_ij: {params['popt0']} Hz/V.<br>"
                        )
                    fitting_report = fitting_report + title_text
                else:
                    fitting_report = "No fitting data"
        fig.update_yaxes(title_text="Bias (V)", row=1 + report_n, col=1)
        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )

        report_n += 1
    if report_n > 1:
        fig.update_traces(showscale=False)

    figures.append(fig)

    return figures, fitting_report


frequency_flux_msr_phase_resonator = partial(
    frequency_flux_msr_phase, method="resonator"
)

frequency_flux_msr_phase_qubit = partial(frequency_flux_msr_phase, method="qubit")


# Dispersive shift
def dispersive_frequency_msr_phase(folder, routine, qubit, format):
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
    report_n = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data_0 = load_data(folder, subfolder, routine, format, "data_0")
            data_0.df = data_0.df[data_0.df["qubit"] == qubit]
        except:
            data_0 = DataUnits(
                name=f"data_0",
                quantities={"frequency": "Hz"},
                options=["qubit", "shifted", "iteration"],
            )
        try:
            data_1 = load_data(folder, subfolder, routine, format, "data_1")
            data_1.df = data_1.df[data_1.df["qubit"] == qubit]
        except:
            data_1 = DataUnits(
                name=f"data_1",
                quantities={"frequency": "Hz"},
                options=["qubit", "shifted", "iteration"],
            )

        try:
            fit_data_0 = load_data(folder, subfolder, routine, format, "fit_data_0")
            fit_data_0.df = fit_data_0.df[fit_data_0.df["qubit"] == qubit]
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
            fit_data_1 = load_data(folder, subfolder, routine, format, "fit_data_1")
            fit_data_1.df = fit_data_1.df[fit_data_1.df["qubit"] == qubit]
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
            data.df = data.df.drop(columns=["i", "q", "qubit"])
            iterations = data.df["iteration"].unique()
            frequencies = data.df["frequency"].unique()
            if len(iterations) > 1:
                opacity = 0.3
            else:
                opacity = 1
            for iteration in iterations:
                iteration_data = data.df[data.df["iteration"] == iteration]
                fig.add_trace(
                    go.Scatter(
                        x=iteration_data["frequency"],
                        y=iteration_data["MSR"] * 1e6,
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
                        x=iteration_data["frequency"],
                        y=iteration_data["phase"],
                        marker_color=get_color(2 * report_n + i),
                        opacity=opacity,
                        showlegend=False,
                        legendgroup=f"q{qubit}/r{report_n}: {label}",
                    ),
                    row=1,
                    col=2,
                )
            if len(iterations) > 1:
                data.df = data.df.drop(columns=["iteration"])
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=data.df.groupby("frequency")["MSR"].mean() * 1e6,
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
                        y=data.df.groupby("frequency")["phase"].mean(),
                        marker_color=get_color(2 * report_n + i),
                        showlegend=False,
                        legendgroup=f"q{qubit}/r{report_n}: {label} Average",
                    ),
                    row=1,
                    col=2,
                )

            if len(data) > 0 and (qubit in data_fit.df["qubit"].values):
                freqrange = np.linspace(
                    min(data.df["frequency"]),
                    max(data.df["frequency"]),
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
                            float(data_fit.df["popt0"]),
                            float(data_fit.df["popt1"]),
                            float(data_fit.df["popt2"]),
                            float(data_fit.df["popt3"]),
                        ),
                        name=f"q{qubit}/r{report_n}: {label} Fit",
                        line=go.scatter.Line(dash="dot"),
                        marker_color=get_color(3 * report_n + i),
                    ),
                    row=1,
                    col=1,
                )

                if "readout_frequency" in params:
                    resonator_freqs[label] = params["readout_frequency"]

                for param, value in params.items():
                    if "freq" in param:
                        fitting_report = fitting_report + (
                            f"q{qubit}/r{report_n} | {label} {param}: {value:,.0f} Hz.<br>"
                        )
                    elif "voltage" in param:
                        fitting_report = fitting_report + (
                            f"q{qubit}/r{report_n} | {label} {param}: {value:,.0f} uV.<br>"
                        )
        if ("Spectroscopy" in resonator_freqs) and (
            "Shifted spectroscopy" in resonator_freqs
        ):
            frequency_shift = (
                resonator_freqs["Shifted spectroscopy"]
                - resonator_freqs["Spectroscopy"]
            )
            fitting_report = fitting_report + (
                f"q{qubit}/r{report_n} | Frequency shift: {frequency_shift:,.0f} Hz.<br>"
            )
        fitting_report += "<br>"
        report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (Hz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="Phase (rad)",
    )

    figures.append(fig)

    return figures, fitting_report


def frequency_attenuation(folder, routine, qubit, format):
    """Plot of the experimental data for the flux resonator flux spectroscopy and its corresponding fit.
        Args:
        folder (str): Folder where the data files with the experimental and fit data are.
        routine (str): Routine name (resonator_flux_sample_matrix)
        qubit (int): qubit coupled to the resonator for which we want to plot the data.
        format (str): format of the data files.

    Returns:
        figures (Figure): Array of figures associated to data.

    """

    figures = []
    fitting_report = "No fitting data"

    try:
        data = DataUnits.load_data(folder, "data", routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"frequency": "Hz", "attenuation": "dB"})

    try:
        data1 = DataUnits.load_data(
            folder, "data", routine, format, f"results_q{qubit}"
        )
    except:
        data1 = DataUnits(
            quantities={"snr": "dimensionless", "frequency": "Hz", "attenuation": "dB"}
        )

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        x_title="Frequency (GHz)",
        y_title="Attenuation (db)",
    )
    fig.add_trace(
        go.Scatter(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("attenuation", "dB"),
            name="Punchout",
        ),
        row=1,
        col=1,
    )
    if len(data1) > 0:
        opt_f = float(data1.get_values("frequency", "GHz"))
        opt_att = float(data1.get_values("attenuation", "dB"))
        opt_snr = float(data1.get_values("snr", "dimensionless"))
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.30,
                showarrow=False,
                text=f"Best response found at frequency {round(opt_f,len(str(opt_f))-3)} GHz <br> for attenuation value of {opt_att} dB with snr {opt_snr :.3e}.\n",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
    fig.update_layout(
        margin=dict(l=60, r=20, t=20, b=130),
        autosize=False,
        width=500,
        height=500,
        uirevision="0",
    )

    figures.append(fig)

    return figures, fitting_report


def frequency_bias_flux(folder, routine, qubit, format):
    """Plot of the experimental data of the punchout.
        Args:
        folder (str): Folder where the data files with the experimental and fit data are.
        routine (str): Routine name (resonator_flux_sample_matrix)
        qubit (int): qubit coupled to the resonator for which we want to plot the data.
        format (str): format of the data files.

    Returns:
        figures [Figure]: Array of figures associated to data.

    """

    figures = []
    fitting_report = "No fitting data"

    fluxes = []
    fluxes_fit = []
    for i in range(5):  # FIXME: 5 is hardcoded
        file1 = f"{folder}/data/{routine}/data_q{qubit}_f{i}.csv"
        file2 = f"{folder}/data/{routine}/fit1_q{qubit}_f{i}.csv"
        if os.path.exists(file1):
            fluxes += [i]
        if os.path.exists(file2):
            fluxes_fit += [i]
    if len(fluxes) < 1:
        nb = 1
    else:
        nb = len(fluxes)
    fig = make_subplots(
        rows=1,
        cols=nb,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
        x_title="Bias (V)",
        y_title="Frequency (GHz)",
        shared_xaxes=False,
        shared_yaxes=True,
    )
    for k, j in enumerate(fluxes):
        data_spec = DataUnits.load_data(
            folder, "data", routine, format, f"data_q{qubit}_f{j}"
        )
        fig.add_trace(
            go.Scatter(
                x=data_spec.get_values("bias", "V"),
                y=data_spec.get_values("frequency", "GHz"),
                name=f"fluxline: {j}",
                mode="markers",
            ),
            row=1,
            col=k + 1,
        )

        if j in fluxes_fit:
            try:
                data_fit = Data.load_data(
                    folder, "data", routine, format, f"fit1_q{qubit}_f{j}"
                )
            except:
                data_fit = Data(quantities=[])
            if len(data_spec) > 0 and len(data_fit) > 0:
                curr_range = np.linspace(
                    min(data_spec.get_values("bias", "V")),
                    max(data_spec.get_values("bias", "V")),
                    100,
                )
                if int(j) == int(qubit):
                    f_qs = float(data_fit.get_values("f_qs"))
                    f_rs = float(data_fit.get_values("f_rs"))
                    curr_qs = float(data_fit.get_values("curr_sp"))
                    g = float(data_fit.get_values("g"))
                    d = float(data_fit.get_values("d"))
                    xi = float(data_fit.get_values("xi"))
                    C_ii = float(data_fit.get_values("C_ii"))
                    f_offset = float(data_fit.get_values("f_offset"))
                    text_data = f"Fluxline: {j} <br> freq_r{qubit}_sp = {f_rs :.4e} Hz <br> freq_q{qubit}_sp = {f_qs :.4e} Hz <br> curr_{qubit}_sp = {curr_qs :.2e} A <br> g = {g :.2e} Hz <br> d = {d :.2e} <br> xi = {xi :.2e} 1/A <br> C_{qubit}{j} = {C_ii :.4e} Hz/A <br> f_offset_q{qubit} = {f_offset :.4e} Hz"
                    if len(data_fit.df.keys()) != 10:
                        Ec = float(data_fit.get_values("Ec"))
                        Ej = float(data_fit.get_values("Ej"))
                        text_data += f" <br> Ec = {Ec :.3e} Hz <br> Ej = {Ej :.3e} Hz"
                        freq_r_fit = freq_r_mathieu
                        params = [
                            data_fit.get_values("f_rh"),
                            data_fit.get_values("g"),
                            data_fit.get_values("curr_sp"),
                            data_fit.get_values("xi"),
                            data_fit.get_values("d"),
                            data_fit.get_values("Ec"),
                            data_fit.get_values("Ej"),
                        ]
                    else:
                        freq_r_fit = freq_r_transmon
                        params = [
                            data_fit.get_values("curr_sp"),
                            data_fit.get_values("xi"),
                            data_fit.get_values("d"),
                            data_fit.get_values("f_q/f_rh"),
                            data_fit.get_values("g"),
                            data_fit.get_values("f_rh"),
                        ]
                    y = freq_r_fit(curr_range, *params) / 10**9
                    fig.add_trace(
                        go.Scatter(
                            x=curr_range,
                            y=y,
                            name=f"Fit fluxline {j}",
                            line=go.scatter.Line(dash="dot"),
                        ),
                        row=1,
                        col=k + 1,
                    )
                    fig.add_annotation(
                        dict(
                            font=dict(color="black", size=12),
                            x=np.mean(curr_range),
                            y=-0.9,
                            showarrow=False,
                            text=text_data,
                            textangle=0,
                            xanchor="auto",
                            align="center",
                            xref=f"x{k+1}",
                            yref="paper",  # "y1",
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=curr_range,
                            y=line(
                                curr_range,
                                data_fit.get_values("popt0"),
                                data_fit.get_values("popt1"),
                            )
                            / 10**9,
                            name=f"Fit fluxline {j}",
                            line=go.scatter.Line(dash="dot"),
                        ),
                        row=1,
                        col=k + 1,
                    )
                    C_ij = float(data_fit.get_values("popt0"))
                    fig.add_annotation(
                        dict(
                            font=dict(color="black", size=12),
                            x=np.mean(curr_range),
                            y=-0.9,
                            showarrow=False,
                            text=f"Fluxline: {j} <br> C_{qubit}{j} = {C_ij :.4e} Hz/A.",
                            textangle=0,
                            xanchor="center",
                            align="center",
                            xref=f"x{k+1}",
                            yref="paper",
                        )
                    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=230),
        showlegend=False,
        autosize=False,
        width=500 * max(1, len(fluxes)),
        height=500,
        uirevision="0",
    )

    figures.append(fig)

    return figures, fitting_report
