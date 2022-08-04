# -*- coding: utf-8 -*-
import os

import dash
import yaml
from dash import dcc, html


def home():
    folders = [
        folder
        for folder in os.listdir(os.getcwd())
        if os.path.isdir(folder) and "meta.yml" in os.listdir(folder)
    ]
    return html.Div(
        [
            html.Br(),
            html.H1("Available runs:", className="text-center"),
            html.Div(
                html.Ul(
                    [
                        html.A(
                            html.Div(f"{folder}", className="text-center"),
                            href=f"/live/{folder}",
                            target="_blank",  # to open in new tab
                            className="list-group-item list-group-item-action",
                        )
                        for folder in sorted(folders)
                    ],
                    className="list-group mx-auto justify-content-center",
                    style={"width": "50%"},
                ),
                className="container",
            ),
        ],
        className="container",
    )


def live(path=None):
    # show metadata in the layout
    try:
        with open(os.path.join(path, "meta.yml"), "r") as file:
            metadata = yaml.safe_load(file)
    except (FileNotFoundError, TypeError):
        return html.Div(children=[html.H2(f"Path {path} not available.")])

    children = [
        html.Title(path),
        dcc.Input(id="path", value=path, type="text", style={"display": "none"}),
        html.H1(path),
        html.H2(f"Date: {metadata.get('date')}"),
        html.Table(
            [
                html.Tbody(
                    [
                        html.Tr([html.Th(library), html.Th(version)])
                        for library, version in metadata.get("versions").items()
                    ]
                )
            ],
            className="table table-hover table-bordered",
            style={"width": "20%"},
        ),
        html.Br(),
    ]

    data_path = os.path.join(path, "data")
    for routine in os.listdir(data_path):
        routine_path = os.path.join(data_path, routine)
        children.append(
            html.Div(
                children=[
                    html.H3(routine),
                    dcc.Graph(
                        id={"type": "graph", "index": routine_path},
                    ),
                    dcc.Interval(
                        id={"type": "interval", "index": routine_path},
                        # TODO: Perhaps the user should be allowed to change the refresh rate
                        interval=1000,
                        n_intervals=0,
                        disabled=False,
                    ),
                ]
            )
        )
        children.append(html.Br())

    return html.Div(children=children, className="container")
