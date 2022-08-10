# -*- coding: utf-8 -*-
import os

import dash
import yaml
from dash import dcc, html


def get_folders():
    for folder in os.listdir(os.getcwd()):
        if os.path.isdir(folder) and "meta.yml" in os.listdir(folder):
            yield folder


def home():
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
                        for folder in sorted(get_folders())
                    ],
                    className="list-group mx-auto justify-content-center",
                    style={"width": "50%"},
                ),
                className="container",
            ),
        ],
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

    content = [html.Br()]
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
                    # Empty spaces so that top bar does not cover the
                    # section title when using #links.
                    html.Br(),
                    html.Br(),
                    html.Br(),
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
                ],
                id=routine,
            )
        )
        content.append(html.Br())

    qcvv_version = metadata.get("versions").get("qcvv")
    return [
        html.Header(
            [
                html.A(
                    f"qcvv {qcvv_version}",
                    href="/",
                    className="navbar-brand col-md-3 col-lg-2 me-0 px-3 fs-6",
                ),
                html.Div(
                    html.Div(
                        html.A(html.H4(path), href="#", className="nav-link"),
                        className="nav-item text-nowrap",
                    ),
                    className="navbar-nav",
                ),
                html.Div(
                    html.Div(
                        html.H6(f"Run date: {metadata.get('date')}"),
                        className="nav-link text-nowrap",
                    ),
                    className="navbar-nav",
                ),
                html.Div(
                    html.Div(
                        html.A(html.H6("Export"), href="#", className="nav-link"),
                        className="nav-item text-nowrap",
                    ),
                    className="navbar-nav",
                ),
            ],
            className="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow",
        ),
        html.Div(
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
                                                        folder,
                                                        href=f"/live/{folder}",
                                                        className="nav-link",
                                                    )
                                                ],
                                                className="nav-item",
                                            )
                                            for folder in sorted(get_folders())
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
                    html.Main(
                        content, className="col-md-9 ms-sm-auto col-lg-10 px-md-4"
                    ),
                ],
                className="row",
            ),
            className="container-fluid",
        ),
    ]
