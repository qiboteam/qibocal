import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data
from qibocal.fitting.rb_methods import exp1_func, fit_exp1B_func


def standardrb(folder, routine, qubit, format):
    try:
        data = Data.load_data(folder, routine, format, "data")
    except:
        pass
    try:
        data_fit = Data.load_data(folder, routine, format, "fit")
    except:
        pass

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(routine,),
    )

    fig.add_trace(
        go.Scatter(
            x=data.df["depth"].to_numpy(),
            y=data.df["groundstate_probability"].to_numpy(),
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="runs",
        )
    )

    depthrange = np.linspace(
        min(data.get_values("depth")),
        max(data.get_values("depth")),
        2 * len(data),
    )

    params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
    fig.add_trace(
        go.Scatter(
            x=depthrange,
            y=exp1_func(
                depthrange,
                A=data_fit.get_values("A"),
                f=data_fit.get_values("p"),
                B=data_fit.get_values("B"),
            ),
            # name="A: {:.3f}, p: {:.3f}, B: {:.3f}".format(popt[0], popt[1], popt[2]),
            line=go.scatter.Line(dash="dot"),
        )
    )

    return fig


def standardrb_plot(folder, routine, qubit, format):
    from qibocal.calibrations.protocols.standardrb import StandardRBExperiment, analyze

    experimentpath = f"{folder}/data/{routine}/"
    experiment = Experiment.load(experimentpath)
    fig = analyze(experiment)
    try:
        data = Data.load_data(folder, routine, "pickle", "effectivedepol")
        depol = data.df.to_numpy()[0, 0]
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text="Effective depol param: {:.3f}".format(depol),
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
    except FileNotFoundError:
        pass
    return fig


def crosstalkrb_plot(folder, routine, qubit, format):
    from qibocal.calibrations.protocols.crosstalkrb import (
        CrosstalkRBExperiment,
        analyze,
    )

    experimentpath = f"{folder}/data/{routine}/"
    experiment = CrosstalkRBExperiment.load(experimentpath)
    fig = analyze(experiment)
    try:
        data = Data.load_data(folder, routine, "pickle", "effectivedepol")
        depol = data.df.to_numpy()[0, 0]
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text="Effective depol param: {:.3f}".format(depol),
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
    except FileNotFoundError:
        pass
    return fig


def XIdrb_plot(folder, routine, qubit, format):
    from qibocal.calibrations.protocols.XIdrb import XIdExperiment, analyze

    experimentpath = f"{folder}/data/{routine}/"
    experiment = XIdExperiment.load(experimentpath)
    fig = analyze(experiment)
    try:
        data = Data.load_data(folder, routine, "pickle", "effectivedepol")
        depol = data.df.to_numpy()[0, 0]
        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text="Effective depol param: {:.3f}".format(depol),
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )
    except FileNotFoundError:
        pass
    return fig
