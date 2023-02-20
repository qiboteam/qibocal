import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import freq_r_mathieu, freq_r_transmon, line, lorenzian
from qibocal.plots.utils import get_color, get_data_subfolders, grouped_by_mean


# Resonator and qubit spectroscopies
def frequency_msr_phase(folder, routine, qubit, format):
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
            data.df = data.df.drop(columns=["i", "q", "qubit"])
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
                data.df = data.df.drop(columns=["iteration"])
                unique_frequencies, mean_measurements = grouped_by_mean(data.df, 2, 0)
                fig.add_trace(
                    go.Scatter(
                        x=unique_frequencies,
                        y=mean_measurements * 1e6,
                        marker_color=get_color(2 * report_n + i),
                        name=f"q{qubit}/r{report_n}: {label} Average",
                        showlegend=True,
                        legendgroup=f"q{qubit}/r{report_n}: {label} Average",
                    ),
                    row=1,
                    col=1,
                )
                unique_frequencies, mean_phases = grouped_by_mean(data.df, 2, 1)
                fig.add_trace(
                    go.Scatter(
                        x=unique_frequencies,
                        y=mean_phases,
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
            data = DataUnits.load_data(folder, subfolder, routine, format, f"data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(
                name=f"data",
                quantities={"frequency": "Hz", "attenuation": "dB"},
                options=["qubit", "iteration"],
            )

        iterations = data.df["iteration"].unique()

        averaged_data = data.df.drop(columns=["i", "q", "qubit", "iteration"])

        if len(iterations) > 1:
            (
                unique_frequencies,
                unique_attenuations,
                mean_measurements,
                mean_phases,
            ) = grouped_by_mean(averaged_data, 2, 0, 3, 1)
        else:
            unique_frequencies = averaged_data["frequency"].pint.to("Hz").pint.magnitude
            unique_attenuations = (
                averaged_data["attenuation"].pint.to("dB").pint.magnitude
            )
            mean_measurements = averaged_data["MSR"].pint.to("V").pint.magnitude
            mean_phases = averaged_data["phase"].pint.to("rad").pint.magnitude

        def norm(x_mags):
            return (x_mags - np.min(x_mags)) / (np.max(x_mags) - np.min(x_mags))

        normalised_data = norm(mean_measurements)

        fig.add_trace(
            go.Heatmap(
                x=unique_frequencies,
                y=unique_attenuations,
                z=normalised_data,
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
                x=unique_frequencies,
                y=unique_attenuations,
                z=mean_phases,
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
        data.df = data.df.drop(columns=["i", "q", "qubit"])

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
            data.df = data.df.drop(columns=["iteration"])
            unique_frequencies, mean_measurements = grouped_by_mean(data.df, 2, 0)
            fig.add_trace(
                go.Scatter(
                    x=unique_frequencies,
                    y=mean_measurements * 1e6,  # pylint: disable=E1101
                    marker_color=get_color(report_n),
                    name=f"q{qubit}/r{report_n}: Average",
                    showlegend=True,
                    legendgroup=f"q{qubit}/r{report_n}: Average",
                ),
                row=1,
                col=1,
            )
            unique_frequencies, mean_phases = grouped_by_mean(data.df, 2, 1)
            fig.add_trace(
                go.Scatter(
                    x=unique_frequencies,
                    y=mean_phases,  # pylint: disable=E1101
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


# Resonator spectroscopy flux
def frequency_flux_msr_phase(folder, routine, qubit, format):
    figures = []
    fitting_report = "No fitting data"

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
                quantities={"frequency": "Hz", "bias": "V"},
                options=["qubit", "fluxline", "iteration"],
            )

        iterations = data.df["iteration"].unique()
        fluxlines = data.df["fluxline"].unique()
        frequencies = data.df["frequency"].pint.to("Hz").pint.magnitude.unique()
        # biass = data.df["bias"].pint.to("V").pint.magnitude.unique()

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
                fluxline_df = fluxline_df.drop(
                    columns=["i", "q", "qubit", "fluxline", "iteration"]
                )

                if len(iterations) > 1:
                    (
                        unique_frequencies,
                        unique_bias,
                        mean_measurements,
                        mean_phases,
                    ) = grouped_by_mean(fluxline_df, 2, 0, 3, 1)
                else:
                    unique_frequencies = (
                        fluxline_df["frequency"].pint.to("Hz").pint.magnitude
                    )
                    unique_bias = fluxline_df["bias"].pint.to("V").pint.magnitude
                    mean_measurements = fluxline_df["MSR"].pint.to("V").pint.magnitude
                    mean_phases = fluxline_df["phase"].pint.to("rad").pint.magnitude

                fig.add_trace(
                    go.Heatmap(
                        x=unique_frequencies,
                        y=unique_bias,
                        z=mean_measurements,
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
            fluxline_df = fluxline_df.drop(
                columns=["i", "q", "qubit", "fluxline", "iteration"]
            )

            if len(iterations) > 1:
                (
                    unique_frequencies,
                    unique_bias,
                    mean_measurements,
                    mean_phases,
                ) = grouped_by_mean(fluxline_df, 2, 0, 3, 1)
            else:
                unique_frequencies = (
                    fluxline_df["frequency"].pint.to("Hz").pint.magnitude
                )
                unique_bias = fluxline_df["bias"].pint.to("V").pint.magnitude
                mean_measurements = fluxline_df["MSR"].pint.to("V").pint.magnitude
                mean_phases = fluxline_df["phase"].pint.to("rad").pint.magnitude

            fig.add_trace(
                go.Heatmap(
                    x=unique_frequencies,
                    y=unique_bias,
                    z=mean_measurements,
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
                    x=unique_frequencies,
                    y=unique_bias,
                    z=mean_phases,
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
            data.df = data.df.drop(columns=["i", "q", "qubit"])
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
                data.df = data.df.drop(columns=["iteration"])
                unique_frequencies, mean_measurements = grouped_by_mean(data.df, 2, 0)
                fig.add_trace(
                    go.Scatter(
                        x=unique_frequencies,
                        y=mean_measurements * 1e6,
                        marker_color=get_color(2 * report_n + i),
                        name=f"q{qubit}/r{report_n}: {label} Average",
                        showlegend=True,
                        legendgroup=f"q{qubit}/r{report_n}: {label} Average",
                    ),
                    row=1,
                    col=1,
                )
                unique_frequencies, mean_phases = grouped_by_mean(data.df, 2, 1)
                fig.add_trace(
                    go.Scatter(
                        x=unique_frequencies,
                        y=mean_phases,
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
        opt_f = data1.get_values("frequency", "GHz")[0]
        opt_att = data1.get_values("attenuation", "dB")[0]
        opt_snr = data1.get_values("snr", "dimensionless")[0]
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


# Not modified or checked
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
                    f_qs = data_fit.get_values("f_qs")[0]
                    f_rs = data_fit.get_values("f_rs")[0]
                    curr_qs = data_fit.get_values("curr_sp")[0]
                    g = data_fit.get_values("g")[0]
                    d = data_fit.get_values("d")[0]
                    xi = data_fit.get_values("xi")[0]
                    C_ii = data_fit.get_values("C_ii")[0]
                    f_offset = data_fit.get_values("f_offset")[0]
                    text_data = f"Fluxline: {j} <br> freq_r{qubit}_sp = {f_rs :.4e} Hz <br> freq_q{qubit}_sp = {f_qs :.4e} Hz <br> curr_{qubit}_sp = {curr_qs :.2e} A <br> g = {g :.2e} Hz <br> d = {d :.2e} <br> xi = {xi :.2e} 1/A <br> C_{qubit}{j} = {C_ii :.4e} Hz/A <br> f_offset_q{qubit} = {f_offset :.4e} Hz"
                    if len(data_fit.df.keys()) != 10:
                        Ec = data_fit.get_values("Ec")[0]
                        Ej = data_fit.get_values("Ej")[0]
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
                    C_ij = data_fit.get_values("popt0")[0]
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
