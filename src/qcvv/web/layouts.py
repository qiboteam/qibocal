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
            html.H1("Available runs:"),
            html.Div(
                [
                    html.H3(
                        dcc.Link(
                            f"{folder}",
                            href=f"/live/{folder}",
                            target="_blank",  # to open in new tab
                        )
                    )
                    for folder in sorted(folders)
                ]
            ),
        ]
    )


def live(path=None):
    try:
        # read metadata and show in the live page
        with open(os.path.join(path, "meta.yml"), "r") as file:
            metadata = yaml.safe_load(file)
    except (FileNotFoundError, TypeError):
        return html.Div(children=[html.H2(f"Path {path} not available.")])

    # read routines from action runcard
    with open(os.path.join(path, "runcard.yml"), "r") as file:
        runcard = yaml.safe_load(file)

    content = [
        html.Div(
            [
                html.H2(path),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button(
                                    "Export",
                                    type="button",
                                    className="btn btn-sm btn-outline-secondary",
                                )
                            ],
                            className="btn-group me-2",
                        )
                    ],
                    className="btn-toolbar mb-2 mb-md-0",
                ),
            ],
            className="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom",
        ),
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

    navbar_routines = []
    for routine in runcard.get("actions").keys():
        routine_path = os.path.join(path, "data", routine)
        navbar_routines.append(
            html.Li(
                [
                    html.A(
                        [
                            routine,
                        ],
                        className="nav-link",
                        href=f"#{routine}",
                    )
                ],
                className="nav-item",
            ),
        )
        content.append(
            html.Div(
                [
                    html.H3(routine, id=routine),
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
        content.append(html.Br())

    return html.Div(
        html.Div(
            [
                html.Nav(
                    [
                        html.Div(
                            [
                                html.Ul(
                                    navbar_routines,
                                    className="nav flex-column",
                                ),
                                html.H6(
                                    [
                                        html.Span("Saved reports"),
                                        # html.A(className="link-secondary", href="#", aria-label="Add a new report">
                                        #    <span data-feather="plus-circle" class="align-text-bottom"></span>
                                    ],
                                    className="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted text-uppercase",
                                ),
                                html.Ul(
                                    [
                                        html.Li(
                                            [
                                                # <span data-feather="file-text" class="align-text-bottom"></span>
                                                html.A(
                                                    "Current month",
                                                    href="#",
                                                    className="nav-link",
                                                )
                                            ],
                                            className="nav-item",
                                        ),
                                        html.Li(
                                            [
                                                # <span data-feather="file-text" class="align-text-bottom"></span>
                                                html.A(
                                                    "Last quarter",
                                                    href="#",
                                                    className="nav-link",
                                                )
                                            ],
                                            className="nav-item",
                                        ),
                                    ],
                                    className="nav flex-column mb-2",
                                ),
                            ],
                            className="position-sticky pt-3 sidebar-sticky",
                        )
                    ],
                    id="sidebarMenu",
                    className="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse",
                ),
                html.Main(content, className="col-md-9 ms-sm-auto col-lg-10 px-md-4"),
            ],
            className="row",
        ),
        className="container-fluid",
    )
