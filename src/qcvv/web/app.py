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
        dcc.Graph(id="graph", figure={}),
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
    method, folder, routine, qubit, format = url.split(os.sep)[2:]
    try:
        return getattr(plots, method)(folder, routine, qubit, format)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return current_figure
