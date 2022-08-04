# -*- coding: utf-8 -*-
import os

import yaml
from dash import MATCH, Dash, Input, Output, dcc, html

from qcvv import plots
from qcvv.data import Dataset
from qcvv.web.layouts import home, live

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css"
    ],
)

app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(url):
    if url == "/":
        return home()
    elif url[:5] == "/live":
        path = url.split("/")[-1]
        return live(path)
    else:
        return html.H1("This page does not exist.")


@app.callback(
    Output({"type": "graph", "index": MATCH}, "figure"),
    Input({"type": "interval", "index": MATCH}, "n_intervals"),
    Input({"type": "graph", "index": MATCH}, "id"),
    Input("path", "value"),
)
def get_graph(n, graph_id, folder):
    _, routine = os.path.split(graph_id.get("index"))
    # find data format
    with open(os.path.join(folder, "runcard.yml"), "r") as file:
        runcard = yaml.safe_load(file)
    format = runcard.get("format")

    try:
        data = Dataset.load_data(folder, routine, format)
        return getattr(plots, routine)(
            data.df, autosize=False, width=1200, height=800, uirevision="0"
        )
        # TODO: Find a better way to fix width and height
        # ``uirevision`` allows zooming while live plotting
    except FileNotFoundError:
        return dict()
