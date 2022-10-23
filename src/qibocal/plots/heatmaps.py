# -*- coding: utf-8 -*-
import os.path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import DataUnits


def get_data_subfolders(folder):
    # iterate over multiple data folders
    subfolders = []
    for file in os.listdir(folder):
        d = os.path.join(folder, file)
        if os.path.isdir(d):
            subfolders.append(os.path.basename(d))

    return subfolders[::-1]


# Resonator spectroscopy flux
def frequency_flux_msr_phase(folder, routine, qubit, format):

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)

    fig = make_subplots(
        rows=len(subfolders),
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    i = 1
    for subfolder in subfolders:

        data = DataUnits.load_data(folder, subfolder, routine, format, f"data_q{qubit}")

        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("current", "A"),
                z=data.get_values("MSR", "V"),
                colorbar_x=0.45,
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("current", "A"),
                z=data.get_values("phase", "rad"),
                colorbar_x=1.0,
            ),
            row=i,
            col=2,
        )

        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )

        if i == 1:
            fig["layout"]["xaxis"]["title"] = f"q{qubit}/r{i-1}: Frequency (GHz)"
            fig["layout"]["yaxis"]["title"] = "Current (A)"
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "Current (A)"

        else:
            xaxis = f"xaxis{2*i-1}"
            yaxis = f"yaxis{2*i-1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "Current (A)"
            xaxis = f"xaxis{2*i}"
            yaxis = f"yaxis{2*i}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "Current (A)"

        i += 1

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
            "MSR (V)",
            "phase (rad)",
        ),
    )

    i = 1
    for subfolder in subfolders:
        data = DataUnits.load_data(folder, subfolder, routine, format, f"data_q{qubit}")

        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("attenuation", "dB"),
                z=data.get_values("MSR", "V"),
                colorbar_x=0.45,
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("attenuation", "dB"),
                z=data.get_values("phase", "rad"),
                colorbar_x=1.0,
            ),
            row=i,
            col=2,
        )

        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )

        if i == 1:
            fig["layout"]["xaxis"]["title"] = f"q{qubit}/r{i-1}: Frequency (GHz)"
            fig["layout"]["yaxis"]["title"] = "Attenuation (dB)"
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "Attenuation (dB)"

        else:
            xaxis = f"xaxis{2*i-1}"
            yaxis = f"yaxis{2*i-1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "Attenuation (dB)"
            xaxis = f"xaxis{2*i}"
            yaxis = f"yaxis{2*i}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "Attenuation (dB)"

        i += 1

    return fig


# Resonator spectroscopy flux matrix
def frequency_flux_msr_phase__matrix(folder, routine, qubit, format):

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)

    k = 1
    last_axis_index = 1
    for subfolder in subfolders:

        fluxes = []
        for i in range(25):  # FIXME: 25 is hardcoded
            file = f"{folder}/{subfolder}/{routine}/data_q{qubit}_f{i}.csv"
            if os.path.exists(file):
                fluxes += [i]

        if len(fluxes) < 1:
            nb = 1
        else:
            nb = len(fluxes)

        if k == 1:
            fig = make_subplots(
                rows=len(subfolders) * 2,
                cols=nb,
                horizontal_spacing=0.1,
                vertical_spacing=0.2,
                shared_xaxes=True,
                shared_yaxes=True,
            )

        for j in fluxes:

            if j == fluxes[-1]:
                showscale = True
            else:
                showscale = False

            data = DataUnits.load_data(
                folder, subfolder, routine, format, f"data_q{qubit}_f{j}"
            )

            fig.add_trace(
                go.Heatmap(
                    x=data.get_values("frequency", "GHz"),
                    y=data.get_values("current", "A"),
                    z=data.get_values("MSR", "V"),
                    showscale=showscale,
                ),
                row=k,
                col=j,
            )
            fig.add_trace(
                go.Heatmap(
                    x=data.get_values("frequency", "GHz"),
                    y=data.get_values("current", "A"),
                    z=data.get_values("phase", "rad"),
                    showscale=showscale,
                ),
                row=k + 1,
                col=j,
            )

        if k == 1:
            fig["layout"]["xaxis"]["title"] = f"q{qubit}/r{k-1}: Frequency (GHz)"
            fig["layout"]["yaxis"]["title"] = "current (A)"
            xaxis = f"xaxis{k+1}"
            yaxis = f"yaxis{k+1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{k-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "current (A)"
            xaxis = f"xaxis{k+2}"
            yaxis = f"yaxis{k+2}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{k-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "current (A)"
            xaxis = f"xaxis{k+3}"
            yaxis = f"yaxis{k+3}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{k-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "current (A)"

        else:
            xaxis = f"xaxis{2*k-1}"
            yaxis = f"yaxis{2*k-1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{k-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "current (A)"
            xaxis = f"xaxis{2*k}"
            yaxis = f"yaxis{2*k}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{k-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "current (A)"
            xaxis = f"xaxis{2*k+1}"
            yaxis = f"yaxis{2*k+1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{k-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "current (A)"
            xaxis = f"xaxis{2*k+2}"
            yaxis = f"yaxis{2*k+2}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{k-1}: Frequency (GHz)"
            fig["layout"][yaxis]["title"] = "current (A)"

        k += 2

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )
    return fig


# Rabi pulse length and gain
def duration_gain_msr_phase(folder, routine, qubit, format):

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)

    fig = make_subplots(
        rows=len(subfolders),
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )
    i = 1
    for subfolder in subfolders:
        data = DataUnits.load_data(folder, subfolder, routine, format, f"data_q{qubit}")

        fig.add_trace(
            go.Heatmap(
                x=data.get_values("duration", "ns"),
                y=data.get_values("gain", "dimensionless"),
                z=data.get_values("MSR", "V"),
                colorbar_x=0.45,
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("duration", "ns"),
                y=data.get_values("gain", "dimensionless"),
                z=data.get_values("phase", "rad"),
                colorbar_x=1.0,
            ),
            row=i,
            col=2,
        )

        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )

        if i == 1:
            fig["layout"]["xaxis"]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"]["yaxis"]["title"] = "gain (dimensionless)"
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"][yaxis]["title"] = "gain (dimensionless)"

        else:
            xaxis = f"xaxis{2*i-1}"
            yaxis = f"yaxis{2*i-1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"][yaxis]["title"] = "gain (dimensionless)"
            xaxis = f"xaxis{2*i}"
            yaxis = f"yaxis{2*i}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"][yaxis]["title"] = "gain (dimensionless)"

        i += 1

    return fig


# Rabi pulse length and amplitude
def duration_amplitude_msr_phase(folder, routine, qubit, format):

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)

    fig = make_subplots(
        rows=len(subfolders),
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    i = 1
    for subfolder in subfolders:
        data = DataUnits.load_data(folder, subfolder, routine, format, f"data_q{qubit}")

        fig.add_trace(
            go.Heatmap(
                x=data.get_values("duration", "ns"),
                y=data.get_values("amplitude", "dimensionless"),
                z=data.get_values("MSR", "V"),
                colorbar_x=0.45,
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("duration", "ns"),
                y=data.get_values("amplitude", "dimensionless"),
                z=data.get_values("phase", "rad"),
                colorbar_x=1.0,
            ),
            row=i,
            col=2,
        )

        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )

        if i == 1:
            fig["layout"]["xaxis"]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"]["yaxis"]["title"] = "A (dimensionless)"
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"][yaxis]["title"] = "A (dimensionless)"

        else:
            xaxis = f"xaxis{2*i-1}"
            yaxis = f"yaxis{2*i-1}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"][yaxis]["title"] = "A (dimensionless)"
            xaxis = f"xaxis{2*i}"
            yaxis = f"yaxis{2*i}"
            fig["layout"][xaxis]["title"] = f"q{qubit}/r{i-1}: duration (ns)"
            fig["layout"][yaxis]["title"] = "A (dimensionless)"

        i += 1

    return fig
