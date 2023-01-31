import datetime
import os
import re
import time

import pandas as pd
import yaml
from dash import Dash, Input, Output, dcc, html

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.web.server import server

# import pandas as pd
# from collections import OrderedDict

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
                        {"label": "2 seconds", "value": 2},
                        {"label": "5 seconds", "value": 5},
                        {"label": "10 seconds", "value": 10},
                        {"label": "20 seconds", "value": 20},
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
                    style={"margin-left": "-120px", "margin-top": "10px"},
                ),
            ],
            style={"display": "flex", "font-family": "verdana"},
        ),
        html.Div(
            id="div-fitting",
            style={
                "margin-left": "40%",
                "margin-top": "40px",
                "font-family": "verdana",
            },
        ),
        html.Div(id="div-figures"),
        dcc.Location(id="url", refresh=False),
        dcc.Interval(
            id="interval",
            # TODO: Perhaps the user should be allowed to change the refresh rate
            interval=5 * 1000,
            n_intervals=0,
            disabled=False,
        ),
    ]
)


@app.callback(
    Output(component_id="div-figures", component_property="children"),
    Output(component_id="latest-timestamp", component_property="children"),
    Output(component_id="interval", component_property="interval"),
    Output(component_id="div-fitting", component_property="children"),
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
        figs, fitting_report = getattr(plots, method)(folder, routine, qubit, format)
        et = time.time()

        if value == 0:
            refresh_rate = (et - st) + 6
        else:
            refresh_rate = value

        for fig in figs:
            figures.append(dcc.Graph(figure=fig))

        fitting_params = re.split(r"<br>|:", fitting_report)
        table = (
            html.Div(
                [
                    html.Table(
                        style={"border": "none"},
                        className="fitting-table",
                        children=[
                            html.Tr(
                                [
                                    html.Th(
                                        "Fitting Parameter",
                                        style={
                                            "background-color": "gray",
                                            "border": "none",
                                            "padding": "10px",
                                        },
                                    ),
                                    html.Th(
                                        "Value",
                                        style={
                                            "background-color": "gray",
                                            "border": "none",
                                            "padding": "10px",
                                        },
                                    ),
                                ],
                            )
                        ]
                        + [
                            html.Tr(
                                [
                                    html.Td(
                                        fitting_params[0],
                                        style={"border": "none", "padding": "10px"},
                                    ),
                                    html.Td(
                                        fitting_params[1],
                                        style={"border": "none", "padding": "10px"},
                                    ),
                                ]
                            ),
                        ],
                    )
                ]
            ),
        )

        timestamp = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        return (
            figures,
            [html.Span(f"Last update: {(timestamp)}")],
            refresh_rate * 1000,
            table,
        )
    except (FileNotFoundError, pd.errors.EmptyDataError):
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        return (
            figures,
            [html.Span(f"Last updated: {timestamp}")],
            refresh_rate * 1000,
            table,
        )
