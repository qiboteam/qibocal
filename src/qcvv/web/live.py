# -*- coding: utf-8 -*-
import os

import plotly.graph_objects as go
import yaml
from dash import MATCH, Dash, Input, Output

from qcvv import plots
from qcvv.data import Dataset

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)


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
        return getattr(plots, routine)(data.df, autosize=False, width=1200, height=800)
    except FileNotFoundError:
        return go.Figure()


@app.callback(
    Output({"type": "last-modified", "index": MATCH}, "value"),
    Output({"type": "interval", "index": MATCH}, "disabled"),
    Input("stopper-interval", "n_intervals"),
    Input({"type": "last-modified", "index": MATCH}, "value"),
    Input({"type": "graph", "index": MATCH}, "id"),
    Input("path", "value"),
)
def toggle_interval(n, last_modified, graph_id, folder):
    """Disables live plotting if data file is not being modified."""
    path = graph_id.get("index")
    if not os.path.exists(path):
        return 0, True

    path = os.path.join(path, os.listdir(path)[0])
    new_modified = os.stat(path)[-1]
    return new_modified, new_modified == last_modified
