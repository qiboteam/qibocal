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
    return subfolders

# Resonator spectroscopy flux- tested
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

        if(i == 1):
            fig['layout']['xaxis']['title']=f'q{qubit}/r{i-1}: Frequency (GHz)'
            fig['layout']['yaxis']['title']='Current (A)'
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig['layout'][xaxis]['title']=f'q{qubit}/r{i-1}: Frequency (GHz)'
            fig['layout'][yaxis]['title']='Current (A)'

        else:
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig['layout'][xaxis]['title']=f'q{qubit}/r{i-1}: Frequency (GHz)'
            fig['layout'][yaxis]['title']='Current (A)'
            xaxis = f"xaxis{i+2}"
            yaxis = f"yaxis{i+2}"
            fig['layout'][xaxis]['title']=f'q{qubit}/r{i-1}: Frequency (GHz)'
            fig['layout'][yaxis]['title']='Current (A)'

        i += 1

    return fig

# Punchout - tested
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

        if(i == 1):
            fig['layout']['xaxis']['title']=f'q{qubit}/r{i-1}: Frequency (GHz)'
            fig['layout']['yaxis']['title']='Attenuation (dB)'
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig['layout'][xaxis]['title']=f'q{qubit}/r{i-1}: Frequency (GHz)'
            fig['layout'][yaxis]['title']='Attenuation (dB)'

        else:
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig['layout'][xaxis]['title']=f'q{qubit}/r{i-1}: Frequency (GHz)'
            fig['layout'][yaxis]['title']='Attenuation (dB)'
            xaxis = f"xaxis{i+2}"
            yaxis = f"yaxis{i+2}"
            fig['layout'][xaxis]['title']=f'q{qubit}/r{i-1}: Frequency (GHz)'
            fig['layout'][yaxis]['title']='Attenuation (dB)'

        i += 1

    return fig

# Resonator spectroscopy flux matrix - to be adapted
def frequency_flux_msr_phase__matrix(folder, routine, qubit, format):
    fluxes = []
    for i in range(25):  # FIXME: 25 is hardcoded
        file = f"{folder}/data/{routine}/data_q{qubit}_f{i}.csv"
        if os.path.exists(file):
            fluxes += [i]

    if len(fluxes) < 1:
        nb = 1
    else:
        nb = len(fluxes)
    fig = make_subplots(
        rows=2,
        cols=nb,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        x_title="Frequency (Hz)",
        y_title="Current (A)",
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for j in fluxes:
        if j == fluxes[-1]:
            showscale = True
        else:
            showscale = False
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}_f{j}")
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("current", "A"),
                z=data.get_values("MSR", "V"),
                showscale=showscale,
            ),
            row=1,
            col=j,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("current", "A"),
                z=data.get_values("phase", "rad"),
                showscale=showscale,
            ),
            row=2,
            col=j,
        )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )
    return fig

# Rabi pulse length and gain - to be tested
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
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("duration", "ns"),
                y=data.get_values("gain", "dimensionless"),
                z=data.get_values("phase", "rad"),
                colorbar_x=1.0,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
        )

        if(i == 1):
            fig['layout']['xaxis']['title']=f'q{qubit}/r{i-1}: duration (ns)'
            fig['layout']['yaxis']['title']='gain (dimensionless)'
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig['layout'][xaxis]['title']=f'q{qubit}/r{i-1}: duration (ns)'
            fig['layout'][yaxis]['title']='gain (dimensionless)'

        else:
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig['layout'][xaxis]['title']=f'q{qubit}/r{i-1}: duration (ns)'
            fig['layout'][yaxis]['title']='gain (dimensionless)'
            xaxis = f"xaxis{i+2}"
            yaxis = f"yaxis{i+2}"
            fig['layout'][xaxis]['title']=f'q{qubit}/r{i-1}: duration (ns)'
            fig['layout'][yaxis]['title']='gain (dimensionless)'

        i += 1

    return fig

# Rabi pulse length and amplitude - to be tested
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

        if(i == 1):
            fig['layout']['xaxis']['title']=f'q{qubit}/r{i-1}: duration (ns)'
            fig['layout']['yaxis']['title']='amplitude (dimensionless)'
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig['layout'][xaxis]['title']=f'q{qubit}/r{i-1}: duration (ns)'
            fig['layout'][yaxis]['title']='amplitude (dimensionless)'

        else:
            xaxis = f"xaxis{i+1}"
            yaxis = f"yaxis{i+1}"
            fig['layout'][xaxis]['title']=f'q{qubit}/r{i-1}: duration (ns)'
            fig['layout'][yaxis]['title']='amplitude (dimensionless)'
            xaxis = f"xaxis{i+2}"
            yaxis = f"yaxis{i+2}"
            fig['layout'][xaxis]['title']=f'q{qubit}/r{i-1}: duration (ns)'
            fig['layout'][yaxis]['title']='amplitude (dimensionless)'

    return fig
