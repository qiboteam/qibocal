import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import freq_r_mathieu, freq_r_transmon, line, lorenzian
from qibocal.plots.utils import get_data_subfolders


# Resonator and qubit spectroscopies
def frequency_msr_phase__fast_precision(folder, routine, qubit, format):

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
    i = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data_fast = DataUnits.load_data(
                folder, subfolder, routine, format, f"fast_sweep_q{qubit}"
            )
        except:
            data_fast = DataUnits(quantities={"frequency": "Hz"})
        try:
            data_precision = DataUnits.load_data(
                folder, subfolder, routine, format, f"precision_sweep_q{qubit}"
            )
        except:
            data_precision = DataUnits(quantities={"frequency": "Hz"})
        try:
            data_fit = Data.load_data(
                folder, subfolder, routine, format, f"fit_q{qubit}"
            )
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
                ]
            )

        fig.add_trace(
            go.Scatter(
                x=data_fast.get_values("frequency", "GHz"),
                y=data_fast.get_values("MSR", "uV"),
                name=f"q{qubit}/r{i} Fast",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data_fast.get_values("frequency", "GHz"),
                y=data_fast.get_values("phase", "rad"),
                name=f"q{qubit}/r{i} Fast",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=data_precision.get_values("frequency", "GHz"),
                y=data_precision.get_values("MSR", "uV"),
                name=f"q{qubit}/r{i}Precision",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data_precision.get_values("frequency", "GHz"),
                y=data_precision.get_values("phase", "rad"),
                name=f"q{qubit}/r{i} Precision",
            ),
            row=1,
            col=2,
        )
        if len(data_fast) > 0 and len(data_fit) > 0:
            freqrange = np.linspace(
                min(data_fast.get_values("frequency", "GHz")),
                max(data_fast.get_values("frequency", "GHz")),
                2 * len(data_fast),
            )
            params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
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
                    name=f"q{qubit}/r{i} Fit",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )
            if data_fit.df[params[2]][0] == 0:
                param2 = ""
            else:
                param2 = f"q{qubit}/r{i} {params[2]}: {data_fit.df[params[2]][0]:.1f} Hz.<br>"

            fitting_report = fitting_report + (
                f"{param2}q{qubit}/r{i} {params[1]}: {data_fit.df[params[1]][0]:.3f} uV.<br>q{qubit}/r{i} {params[0]}: {data_fit.df[params[0]][0]:.0f} Hz.<br><br>"
            )

        i += 1

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
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Phase (rad)",
    )
    return fig


# Resonator and qubit spectroscopies
def frequency_attenuation_msr_phase__cut(folder, routine, qubit, format):

    plot1d_attenuation = 56  # attenuation value to use for 1D frequency vs MSR plot

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    i = 0
    for subfolder in subfolders:
        # print(subfolder)
        data = DataUnits.load_data(folder, subfolder, routine, format, f"data_q{qubit}")
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
                    name=f"q{qubit}/r{i}",
                ),
            )
        i += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting,
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (V)",
    )
    return fig


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


# Dispersive shift
def dispersive_frequency_msr_phase(folder, routine, qubit, formato):

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
    i = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data_spec = DataUnits.load_data(
                folder, subfolder, routine, formato, f"data_q{qubit}"
            )
        except:
            data_spec = DataUnits(name=f"data_q{qubit}", quantities={"frequency": "Hz"})

        try:
            data_shifted = DataUnits.load_data(
                folder, subfolder, routine, formato, f"data_shifted_q{qubit}"
            )
        except:
            data_shifted = DataUnits(
                name=f"data_shifted_q{qubit}", quantities={"frequency": "Hz"}
            )

        try:
            data_fit = Data.load_data(
                folder, subfolder, routine, formato, f"fit_q{qubit}"
            )
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
                ]
            )

        try:
            data_fit_shifted = Data.load_data(
                folder, subfolder, routine, formato, f"fit_shifted_q{qubit}"
            )
        except:
            data_fit_shifted = Data(
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

        fig.add_trace(
            go.Scatter(
                x=data_spec.get_values("frequency", "GHz"),
                y=data_spec.get_values("MSR", "uV"),
                name=f"q{qubit}/r{i} Spectroscopy",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data_spec.get_values("frequency", "GHz"),
                y=data_spec.get_values("phase", "rad"),
                name=f"q{qubit}/r{i} Spectroscopy",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=data_shifted.get_values("frequency", "GHz"),
                y=data_shifted.get_values("MSR", "uV"),
                name=f"q{qubit}/r{i} Shifted Spectroscopy",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data_shifted.get_values("frequency", "GHz"),
                y=data_shifted.get_values("phase", "rad"),
                name=f"q{qubit}/r{i} Shifted Spectroscopy",
            ),
            row=1,
            col=2,
        )

        # fitting traces
        if len(data_spec) > 0 and len(data_fit) > 0:
            freqrange = np.linspace(
                min(data_spec.get_values("frequency", "GHz")),
                max(data_spec.get_values("frequency", "GHz")),
                2 * len(data_spec),
            )
            params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
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
                    name=f"q{qubit}/r{i} Fit spectroscopy",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

            fitting_report = fitting_report + (
                f"q{qubit}/r{i} {params[2]}: {data_fit.df[params[2]][0]:.1f} Hz.<br>"
            )

        # fitting shifted  traces
        if len(data_shifted) > 0 and len(data_fit_shifted) > 0:
            freqrange = np.linspace(
                min(data_shifted.get_values("frequency", "GHz")),
                max(data_shifted.get_values("frequency", "GHz")),
                2 * len(data_shifted),
            )
            params = [i for i in list(data_fit_shifted.df.keys()) if "popt" not in i]
            fig.add_trace(
                go.Scatter(
                    x=freqrange,
                    y=lorenzian(
                        freqrange,
                        data_fit_shifted.get_values("popt0"),
                        data_fit_shifted.get_values("popt1"),
                        data_fit_shifted.get_values("popt2"),
                        data_fit_shifted.get_values("popt3"),
                    ),
                    name=f"q{qubit}/r{i} Fit shifted spectroscopy",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

            fitting_report = fitting_report + (
                f"q{qubit}/r{i} shifted {params[2]}: {data_fit_shifted.df[params[2]][0]:.1f} Hz.<br><br>"
            )

        i += 1

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
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Phase (rad)",
    )
    return fig


def frequency_attenuation(folder, routine, qubit, format):
    """Plot of the experimental data for the flux resonator flux spectroscopy and its corresponding fit.
        Args:
        folder (str): Folder where the data files with the experimental and fit data are.
        routine (str): Routine name (resonator_flux_sample_matrix)
        qubit (int): qubit coupled to the resonator for which we want to plot the data.
        format (str): format of the data files.

    Returns:
        fig (Figure): Figure associated to data.

    """
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
    return fig


def frequency_current_flux(folder, routine, qubit, format):
    """Plot of the experimental data of the punchout.
        Args:
        folder (str): Folder where the data files with the experimental and fit data are.
        routine (str): Routine name (resonator_flux_sample_matrix)
        qubit (int): qubit coupled to the resonator for which we want to plot the data.
        format (str): format of the data files.

    Returns:
        fig (Figure): Figure associated to data.

    """
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
        x_title="Current (A)",
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
                x=data_spec.get_values("current", "A"),
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
                    min(data_spec.get_values("current", "A")),
                    max(data_spec.get_values("current", "A")),
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
    return fig
