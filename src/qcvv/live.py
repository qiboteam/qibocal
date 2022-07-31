# -*- coding: utf-8 -*-
import os

import pandas as pd
import plotly.graph_objects as go
import yaml
from dash import MATCH, Dash, Input, Output, dcc, html

from qcvv import plots


def serve_layout(path):
    layout = [
        dcc.Interval(
            id=f"stopper-interval", interval=1000, n_intervals=0, disabled=False
        )
    ]

    for action in os.listdir(path):
        action_path = os.path.join(path, action)
        if os.path.isdir(action_path):
            layout.append(html.H2(action))
            for run in sorted(os.listdir(action_path)):
                run_path = os.path.join(action_path, run)
                layout.append(
                    html.Details(
                        children=[
                            html.Summary(run),
                            dcc.Graph(
                                id={"type": "graph", "index": run_path},
                            ),
                            dcc.Interval(
                                id={"type": "interval", "index": run_path},
                                interval=1000,
                                n_intervals=0,
                                disabled=False,
                            ),
                            dcc.Input(
                                id={
                                    "type": "last-modified",
                                    "index": run_path,
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
    path = os.path.join(graph_id.get("index"), "data.yml")
    if not os.path.exists(path):
        return go.Figure()

    with open(path, "r") as file:
        data = yaml.safe_load(file)

    df = pd.DataFrame(
        {f"{v.get('name')} ({v.get('unit')})": v.get("data") for v in data.values()}
    )

    action = graph_id.get("index").split("/")[1]
    return getattr(plots, action)(df, autosize=False, width=1200, height=800)


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
