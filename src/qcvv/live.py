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
                layout.append(
                    html.Details(
                        children=[
                            html.Summary(run),
                            dcc.Graph(
                                id={"type": "graph", "index": f"{action}/{run}"},
                            ),
                            dcc.Interval(
                                id={"type": "interval", "index": f"{action}/{run}"},
                                interval=1000,
                                n_intervals=0,
                                disabled=False,
                            ),
                            dcc.Input(
                                id={"type": "path", "index": f"{action}/{run}"},
                                value=f"test/{action}/{run}/data.yml",
                                type="text",
                                style={"display": "none"},
                            ),
                            dcc.Input(
                                id={
                                    "type": "last-modified",
                                    "index": f"{action}/{run}",
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
    Input({"type": "path", "index": MATCH}, "value"),
    Input({"type": "path", "index": MATCH}, "id"),
)
def get_graph(n, path, action_id):
    if not os.path.exists(path):
        return go.Figure()

    with open(path, "r") as file:
        data = yaml.safe_load(file)

    df = pd.DataFrame(
        {f"{v.get('name')} ({v.get('unit')})": v.get("data") for v in data.values()}
    )

    action = action_id.get("index").split("/")[0]
    return getattr(plots, action)(df, autosize=False, width=1200, height=800)


@app.callback(
    Output({"type": "last-modified", "index": MATCH}, "value"),
    Output({"type": "interval", "index": MATCH}, "disabled"),
    Input("stopper-interval", "n_intervals"),
    Input({"type": "last-modified", "index": MATCH}, "value"),
    Input({"type": "path", "index": MATCH}, "value"),
)
def toggle_interval(n, last_modified, path):
    """Disables live plotting if data file is not being modified."""
    if not os.path.exists(path):
        return 0, True
    new_modified = os.stat(path)[-1]
    return new_modified, new_modified == last_modified
