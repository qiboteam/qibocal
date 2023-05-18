import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import qibocal.calibrations.niGSC.basics.fitting as fitting_methods
from qibocal.calibrations.niGSC.basics.experiment import Experiment
from qibocal.calibrations.niGSC.basics.utils import number_to_str


def plot_qq(folder: str, routine: str, qubit, format):
    """Load the module for which the plot has to be done.

    Args:
        folder (str): The folder path where the data was stored.
        routine (str): The routine name, here the module name.
        qubit (Any): Is not used here
        format (Any): Is not used here.

    Returns:
        plotly figure: plotly graphical object figure.
    """

    import importlib

    # Load the module, something like 'standardrb'.
    module = importlib.import_module(f"qibocal.calibrations.niGSC.{routine}")
    # Load the experiment with the class method ``load``.
    experiment = module.ModuleExperiment.load(f"{folder}/data/{routine}/")
    # In this data frame the precomputed fitting parameters and other
    # parameters for fitting and plotting are stored.
    aggr_df = pd.read_pickle(f"{folder}/data/{routine}/fit_plot.pkl")
    # Build the figure/report using the responsible module.
    plotly_figure, fitting_report = module.build_report(experiment, aggr_df)

    return [plotly_figure], fitting_report


class Report:
    """Once initialized with the correct parameters an Report object can build
    reports to display results of a gate set characterization experiment.
    """

    def __init__(self) -> None:
        self.all_figures = []
        self.title = "Report"
        self.info_dict = {}

    def build(self):
        l = len(self.all_figures)
        if l < 3:
            divide_by = 1
        else:
            divide_by = 2
        subplot_titles = [figdict.get("subplot_title") for figdict in self.all_figures]
        fig = make_subplots(
            rows=int(l / divide_by) + l % divide_by,
            cols=1 if l == 1 else divide_by,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
        )
        for count, fig_dict in enumerate(self.all_figures):
            plot_list = fig_dict["figs"]
            for plot in plot_list:
                fig.add_trace(
                    plot, row=count // divide_by + 1, col=count % divide_by + 1
                )

        fig.update_xaxes(title_font_size=18, tickfont_size=16)
        fig.update_yaxes(title_font_size=18, tickfont_size=16)
        fig.update_layout(
            font_family="Averta",
            hoverlabel_font_family="Averta",
            title_text=self.title,
            title_font_size=24,
            legend_font_size=16,
            hoverlabel_font_size=16,
            showlegend=True,
            height=300 * (int(l / divide_by) + l % divide_by) if l > divide_by else 500,
            width=1000,
        )

        return fig, self.info_table()

    def info_table(self):
        return "".join(
            [f"q/r | {key}: {value}<br>" for key, value in self.info_dict.items()]
        )


def scatter_fit_fig(
    experiment: Experiment,
    df_aggr: pd.DataFrame,
    xlabel: str,
    index: str,
    fittingparam_label="popt",
):
    """
    Generate a figure dictionary for plotly.

    Args:
        experiment (:class:`qibocal.calibrations.niGSC.basics.experiment.Experiment`):
            an RB experiment that contains a dataframe with the information for the figure.
        df_aggr (pd.DataFrame): DataFrame containing aggregational data about the RB experiment.
        xlabel (str): key for x values in experiment.dataframe[xlabel] and df_aggr[xlabel].
        index (str): key for y values in experiment.dataframe[index] and df_aggr[index].
        fittingparam_label (str): key in df_aggr with a dictionary containing the parameters
        for the dfrow["fit_func"]. Default is "popt".

    Returns:
        dict: dictionary for the plotly figure.
    """
    fig_traces = []
    dfrow = df_aggr.loc[index]
    fig_traces.append(
        go.Scatter(
            x=experiment.dataframe[xlabel],
            y=np.real(experiment.dataframe[index]),
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="runs",
        )
    )
    fig_traces.append(
        go.Scatter(
            x=dfrow[xlabel],
            y=np.real(dfrow["data"]),
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )
    x_fit = np.linspace(min(dfrow[xlabel]), max(dfrow[xlabel]), len(dfrow[xlabel]) * 20)
    y_fit = np.real(
        getattr(fitting_methods, dfrow["fit_func"])(
            x_fit, *dfrow[fittingparam_label].values()
        )
    )
    fig_traces.append(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            name="".join(
                [
                    f"{key}={number_to_str(value)} "
                    for key, value in dfrow[fittingparam_label].items()
                ]
            ),
            line=go.scatter.Line(dash="dot"),
        )
    )
    return {"figs": fig_traces, "xlabel": xlabel, "ylabel": index}
