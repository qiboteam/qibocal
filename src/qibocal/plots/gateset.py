import importlib

import pandas as pd


def plot(folder, routine, qubit, format):
    module = importlib.import_module(f"qibocal.calibrations.protocols.{routine}")
    experiment = module.moduleExperiment.load(f"{folder}/data/{routine}/")
    aggr_df = pd.read_pickle(f"{folder}/data/{routine}/fit_plot.pkl")
    plotly_figure = module.build_report(experiment, aggr_df)
    return plotly_figure
