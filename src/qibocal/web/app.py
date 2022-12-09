import os

import pandas as pd
import yaml
from dash import Dash, Input, Output, dcc, html

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.web.server import server

DataUnits()  # dummy dataset call to suppress ``pint[V]`` error

app = Dash(
    server=server,
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Graph(id="graph", figure={}),
        dcc.Interval(
            id="interval",
            # TODO: Perhaps the user should be allowed to change the refresh rate
            interval=5000,
            n_intervals=0,
            disabled=False,
        ),
    ]
)


@app.callback(
    Output("graph", "figure"),
    Input("interval", "n_intervals"),
    Input("graph", "figure"),
    Input("url", "pathname"),
)
def get_graph(n, current_figure, url):
    method, folder, routine, qubit, format = url.split(os.sep)[2:]
    try:
        # data = DataUnits.load_data(folder, routine, format, "precision_sweep")
        # with open(f"{folder}/platform.yml", "r") as f:
        #     nqubits = yaml.safe_load(f)["nqubits"]
        # if len(data) > 2:
        #     params, fit = resonator_spectroscopy_fit(folder, format, nqubits)
        # else:
        #     params, fit = None, None
        # return getattr(plots.resonator_spectroscopy, method)(data, params, fit)

        # # FIXME: Temporarily hardcode the plotting method to test
        # # multiple routines with different names in one folder
        # # should be changed to:
        # # return getattr(getattr(plots, routine), method)(data)

        return getattr(plots, method)(folder, routine, qubit, format)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return current_figure
