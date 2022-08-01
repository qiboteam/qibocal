# -*- coding: utf-8 -*-
import os

import pandas as pd
import plotly.graph_objects as go
import yaml
from dash import MATCH, Dash, Input, Output, dcc, html

from qcvv import plots
from qcvv.data import Dataset


def serve_layout(path):
    # show metadata in the layout
    with open(os.path.join(path, "meta.yml"), "r") as file:
        metadata = yaml.safe_load(file)

    layout = [
        dcc.Interval(
            id=f"stopper-interval", interval=1000, n_intervals=0, disabled=False
        ),
        html.P(f"Path name: {path}"),
        html.P(f"Run date: {metadata.get('date')}"),
        html.P(f"Versions: "),
        html.Table(
            [
                html.Tr([html.Th(library), html.Th(version)])
                for library, version in metadata.get("versions").items()
            ]
        ),
        html.Br(),
    ]

    data_path = os.path.join(path, "data")
    for routine in os.listdir(data_path):
        routine_path = os.path.join(data_path, routine)
        layout.append(
            html.Details(
                children=[
                    html.Summary(routine),
                    dcc.Graph(
                        id={"type": "graph", "index": routine_path},
                    ),
                    dcc.Interval(
                        id={"type": "interval", "index": routine_path},
                        interval=1000,
                        n_intervals=0,
                        disabled=False,
                    ),
                    dcc.Input(
                        id={
                            "type": "last-modified",
                            "index": routine_path,
                        },
                        value=0,
                        type="number",
                        style={"display": "none"},
                    ),
                ]
            )
        )
        layout.append(html.Br())

    return html.Div(children=layout)


app = Dash(__name__)


@app.callback(
    Output({"type": "graph", "index": MATCH}, "figure"),
    Input({"type": "interval", "index": MATCH}, "n_intervals"),
    Input({"type": "graph", "index": MATCH}, "id"),
)
def get_graph(n, graph_id):
    # path = os.path.join(graph_id.get("index"), "data.pkl")
    # if not os.path.exists(path):
    #    return go.Figure()

    data = Dataset()
    folder, routine = os.path.split(graph_id.get("index"))
    try:
        data.load_data(folder, routine, "pickle")
        return getattr(plots, routine)(data.df, autosize=False, width=1200, height=800)
    except FileNotFoundError:
        return go.Figure()


@app.callback(
    Output({"type": "last-modified", "index": MATCH}, "value"),
    Output({"type": "interval", "index": MATCH}, "disabled"),
    Input("stopper-interval", "n_intervals"),
    Input({"type": "last-modified", "index": MATCH}, "value"),
    Input({"type": "graph", "index": MATCH}, "id"),
)
def toggle_interval(n, last_modified, graph_id):
    """Disables live plotting if data file is not being modified."""
    path = graph_id.get("index")
    if not os.path.exists(path):
        return 0, True
    new_modified = os.stat(path)[-1]
    return new_modified, new_modified == last_modified
