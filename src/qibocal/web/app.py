import datetime
import os
import time

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
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Refresh rate:",
                            style={
                                "margin-right": "2em",
                                "margin-top": "5px",
                                "font-size": "1.1em",
                                "font-family": "verdana",
                            },
                        )
                    ],
                ),
                dcc.Dropdown(
                    id="interval-refresh",
                    placeholder="Select refresh rate",
                    options=[
                        {"label": "Auto", "value": 0},
                        {"label": "No refresh", "value": 3600},
                    ],
                    value=0,
                    style={
                        "width": "35%",
                        "margin-left": "-10px",
                        "font-family": "verdana",
                    },
                ),
                html.Div(
                    id="latest-timestamp",
                    style={"margin-left": "-150px", "margin-top": "10px"},
                ),
            ],
            style={"display": "flex", "font-family": "verdana"},
        ),
        html.Div(id="div-figures"),
        dcc.Location(id="url", refresh=False),
        dcc.Interval(
            id="interval",
            interval=160 * 1000,
            n_intervals=0,
            disabled=False,
        ),
    ]
)


@app.callback(
    Output(component_id="div-figures", component_property="children"),
    Output(component_id="latest-timestamp", component_property="children"),
    Output(component_id="interval", component_property="interval"),
    Input("interval", "n_intervals"),
    Input("url", "pathname"),
    Input("interval-refresh", "value"),
)
def get_graph(interval, url, value):
    st = time.time()

    figures = []

    if "data" not in url:
        url = f"/data{url}"

    method, folder, routine, qubit, format = url.split(os.sep)[2:]
    if qubit.isdigit():
        qubit = int(qubit)

    try:
        figs = getattr(plots, method)(folder, routine, qubit, format)
        et = time.time()

        print(f"elapsed time: {et-st}")

        if value == 0:
            refresh_rate = int(et - st) + 6
        else:
            refresh_rate = value

        for fig in figs:
            figures.append(dcc.Graph(figure=fig))

        timestamp = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        return (
            figures,
            [html.Span(f"Last update: {(timestamp)}")],
            refresh_rate * 1000,
        )
    except (FileNotFoundError, pd.errors.EmptyDataError):
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        return (
            figures,
            [html.Span(f"Last updated: {timestamp}")],
            refresh_rate * 1000,
        )
