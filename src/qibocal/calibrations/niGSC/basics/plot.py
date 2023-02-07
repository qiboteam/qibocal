import plotly.graph_objects as go
from plotly.subplots import make_subplots
import qibocal.calibrations.niGSC.basics.fitting as fitting_methods
import numpy as np
from qibocal.calibrations.niGSC.basics.experiment import Experiment
import pandas as pd


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
    module = importlib.import_module(f"qibocal.calibrations.protocols.{routine}")
    # Load the experiment with the class method ``load``.
    experiment = module.moduleExperiment.load(f"{folder}/data/{routine}/")
    # In this data frame the precomputed fitting parameters and other 
    # parameters for fitting and plotting are stored.
    aggr_df = pd.read_pickle(f"{folder}/data/{routine}/fit_plot.pkl")
    # Build the figure/report using the responsible module.
    plotly_figure = module.build_report(experiment, aggr_df)
    return plotly_figure

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
        subplot_titles = [figdict.get("subplot_title") for figdict in self.all_figures]
        fig = make_subplots(
            rows=int(l / 2) + l % 2 + 1,
            cols=1 if l == 1 else 2,
            subplot_titles=subplot_titles,
        )
        for count, fig_dict in enumerate(self.all_figures):
            plot_list = fig_dict["figs"]
            for plot in plot_list:
                fig.add_trace(plot, row=count // 2 + 1, col=count % 2 + 1)

        fig.add_annotation(
            dict(
                bordercolor="black",
                font=dict(color="black", size=16),
                x=0.0,
                y=1.0 / (int(l / 2) + l % 2 + 1) - len(self.info_dict) * 0.005,
                showarrow=False,
                text="<br>".join(
                    [f"{key} : {value}\n" for key, value in self.info_dict.items()]
                ),
                align="left",
                textangle=0,
                yanchor="top",
                xref="paper",
                yref="paper",
            )
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
            height=500 * (int(l / 2) + l % 2) if l > 2 else 1000,
            width=1000,
        )

        return fig


def scatter_fit_fig(
    experiment: Experiment, df_aggr: pd.DataFrame, xlabel: str, index: str
):
    fig_traces = []
    dfrow = df_aggr.loc[index]
    fig_traces.append(
        go.Scatter(
            x=experiment.dataframe[xlabel],
            y=experiment.dataframe[index],
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="runs",
        )
    )
    fig_traces.append(
        go.Scatter(
            x=dfrow[xlabel],
            y=dfrow["data"],
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )
    x_fit = np.linspace(min(dfrow[xlabel]), max(dfrow[xlabel]), len(dfrow[xlabel]) * 20)
    y_fit = getattr(fitting_methods, dfrow["fit_func"])(x_fit, *dfrow["popt"].values())
    fig_traces.append(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            name="".join(
                ["{}:{:.3f} ".format(key, dfrow["popt"][key]) for key in dfrow["popt"]]
            ),
            line=go.scatter.Line(dash="dot"),
        )
    )
    return {"figs": fig_traces, "xlabel": xlabel, "ylabel": index}