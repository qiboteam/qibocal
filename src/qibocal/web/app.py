import datetime
import os
import re
import time

import pandas as pd
from dash import Dash, Input, Output, dcc, html

from qibocal import plots
from qibocal.calibrations.niGSC.basics.plot import plot_qq
from qibocal.data import DataUnits
from qibocal.web.server import server

DataUnits()  # dummy dataset call to suppress ``pint[V]`` error

app = Dash(
    server=server,
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    [
        html.Link(href="/static/styles.css", rel="stylesheet"),
        html.Div(
            id="refresh-area",
            className="refresh-area",
            children=[
                html.Div(
                    id="refresh-text",
                    className="refresh-text",
                    children=[html.P("Refresh rate:")],
                ),
                dcc.Dropdown(
                    id="interval-refresh",
                    className="interval-refresh",
                    placeholder="Select refresh rate",
                    options=[
                        {"label": "Auto", "value": 0},
                        {"label": "No refresh", "value": 3600},
                    ],
                    value=0,
                ),
                html.Div(
                    id="latest-timestamp",
                    className="latest-timestamp",
                ),
            ],
        ),
        html.Div(id="div-fitting", className="div-fitting"),
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
        if hasattr(plots, method):
            figs, fitting_report = getattr(plots, method)(
                folder, routine, qubit, format
            )
        else:
            figs, fitting_report = plot_qq(folder, routine, qubit, format)
        et = time.time()

        if value == 0:
            refresh_rate = int(et - st) + 6
        else:
            refresh_rate = value

        for fig in figs:
            figures.append(dcc.Graph(figure=fig))

        table = ""
        if "No fitting data" not in fitting_report:
            fitting_params = re.split(r"<br>|:|\|", fitting_report)
            fitting_params = list(filter(lambda x: x.strip(), fitting_params))
            table_header = [
                html.Thead(
                    children=[
                        html.Tr(
                            children=[
                                html.Th(
                                    className="th_styles", children="qubit # / report #"
                                ),
                                html.Th(
                                    className="th_styles", children="Fitting Parameter"
                                ),
                                html.Th(className="th_styles", children="Value"),
                            ]
                        )
                    ]
                )
            ]
            table_rows = []

            for i in range(0, len(fitting_params), 3):
                table_rows.append(
                    html.Tr(
                        className="td_styles",
                        children=[
                            html.Td(fitting_params[i]),
                            html.Td(fitting_params[i + 1]),
                            html.Td(fitting_params[i + 2]),
                        ],
                    )
                )

            table = [
                html.Table(
                    className="fitting-table", children=table_header + table_rows
                )
            ]

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
