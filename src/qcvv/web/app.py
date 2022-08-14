# -*- coding: utf-8 -*-
import os

import pandas as pd
import yaml
from dash import MATCH, Dash, Input, Output, dcc, html

from qcvv import plots
from qcvv.data import Dataset
from qcvv.web.layouts import live

Dataset()  # dummy dataset call to suppress ``pint[V]`` error

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="QCVV",
    update_title=None,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css"
    ],
)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
        html.Div(id="blank-output"),
    ]
)

app.clientside_callback(
    """
    function(url) {
        if (url === '/') {
            document.title = 'QCVV Home'
        } else {
            document.title = url.split('/')[2]
        }
    }
    """,
    Output("blank-output", "children"),
    Input("url", "pathname"),
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(url):
    if url == "/":
        return live()
    elif url[:5] == "/live":
        path = url.split("/")[-1]
        return live(path)
    else:
        return html.H1("This page does not exist.")


@app.callback(
    Output({"type": "graph", "index": MATCH}, "figure"),
    Input({"type": "interval", "index": MATCH}, "n_intervals"),
    Input({"type": "graph", "index": MATCH}, "id"),
    Input({"type": "graph", "index": MATCH}, "figure"),
)
def get_graph(n, graph_id, current_figure):
    folder, format = os.path.split(graph_id.get("index"))
    folder, method = os.path.split(folder)
    folder, routine = os.path.split(folder)
    folder, _ = os.path.split(folder)
    try:
        data = Dataset.load_data(folder, routine, format)
        # FIXME: Temporarily hardcode the plotting method to test
        # multiple routines with different names in one folder
        return getattr(plots.resonator_spectroscopy_attenuation, method)(data)
        # return getattr(getattr(plots, routine), method)(data)

    except (FileNotFoundError, pd.errors.EmptyDataError):
        return current_figure
