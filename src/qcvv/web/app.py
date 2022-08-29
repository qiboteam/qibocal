# -*- coding: utf-8 -*-
import os

import pandas as pd
from dash import Dash, Input, Output, dcc, html

from qcvv import plots
from qcvv.data import Dataset
from qcvv.web.server import server

Dataset()  # dummy dataset call to suppress ``pint[V]`` error

app = Dash(
    server=server,
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Graph(id="graph"),
        dcc.Interval(
            id="interval",
            # TODO: Perhaps the user should be allowed to change the refresh rate
            interval=1000,
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
    path = os.path.join(*url.split("/")[2:])
    folder, format = os.path.split(path)
    folder, method = os.path.split(folder)
    folder, routine = os.path.split(folder)
    folder, _ = os.path.split(folder)
    try:
        # FIXME: Temporarily hardcode the plotting method to test
        # multiple routines with different names in one folder
        return getattr(plots.resonator_spectroscopy_attenuation, method)(
            folder, routine, format
        )
        # should be changed to:
        # return getattr(plots, method)(folder, routine, format)

    except (FileNotFoundError, pd.errors.EmptyDataError):
        return current_figure
