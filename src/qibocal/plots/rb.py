import plotly.graph_objects as go

from qibocal.data import Data


def standardrb_plot(folder, routine, qubit, format):
    from qibocal.calibrations.protocols.standardrb import StandardRBExperiment, analyze

    experimentpath = f"{folder}/data/{routine}/"
    experiment = StandardRBExperiment.load(experimentpath)
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
