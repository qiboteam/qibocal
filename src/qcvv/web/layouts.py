# -*- coding: utf-8 -*-
import os

import dash
import yaml
from dash import dcc, html


def topbar():
    from qcvv import __version__

    return html.Header(
        [
            html.A(
                html.H6(f"qcvv {__version__}"),
                href="https://github.com/qiboteam/qcvv",
                target="_blank",
                className="navbar-nav nav-item nav-link px-3",
            ),
            html.Ul(
                [
                    html.Li(
                        html.A(html.H6("Home"), href="/", className="nav-link"),
                        className="nav-item col-6 col-lg-auto",
                    ),
                    html.Li(
                        html.A(
                            html.H6("GitHub"),
                            href="https://github.com/qiboteam/qcvv",
                            className="nav-link",
                        ),
                        className="nav-item col-6 col-lg-auto px-3",
                    ),
                ],
                className="navbar-nav flex-row flex-wrap ms-md-auto",
            ),
        ],
        className="navbar navbar-dark sticky-top p-0 shadow",
    )


def sidebar(path, routines):
    if path is None:
        menu = None
    else:
        menu = html.Ul(
            [
                html.Li(
                    [
                        html.A(
                            "Summary",
                            className="nav-link",
                            href=f"#summary",
                        )
                    ],
                    className="nav-item",
                ),
                html.Li(
                    [
                        html.A(
                            "Actions",
                            className="nav-link",
                            href=f"#actions",
                        )
                    ],
                    className="nav-item",
                ),
                html.Ul(
                    [
                        html.Li(
                            routines,
                            className="nav-item",
                        )
                    ]
                ),
            ],
            className="nav flex-column",
        )

    saved_reports = html.Ul(
        [
            html.Li(
                [
                    # <span data-feather="file-text" class="align-text-bottom"></span>
                    html.A(
                        folder,
                        href=f"/live/{folder}",
                        className="nav-link active" if folder == path else "nav-link",
                    )
                ],
                className="nav-item",
            )
            for folder in sorted(os.listdir(os.getcwd()))
            if os.path.isdir(folder) and "meta.yml" in os.listdir(folder)
        ],
        className="nav flex-column mb-2",
    )
    return html.Nav(
        [
            html.Div(
                [
                    html.H6(
                        [
                            html.Span("Index"),
                        ],
                        className="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted text-uppercase",
                    ),
                    menu,
                    html.H6(
                        [
                            html.Span("Saved reports"),
                            # html.A(className="link-secondary", href="#", aria-label="Add a new report">
                            #    <span data-feather="plus-circle" class="align-text-bottom"></span>
                        ],
                        className="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted text-uppercase",
                    ),
                    saved_reports,
                ],
                className="position-sticky pt-3 sidebar-sticky",
            )
        ],
        id="sidebarMenu",
        className="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse",
    )


def summary(metadata):
    if metadata is None:
        return html.H2("Please select a report from the list on the left.")

    return html.Div(
        [
            html.Br(),
            html.Br(),
            html.Br(),
            html.H2("Summary"),
            html.H5(f"Run date: {metadata.get('date')}"),
            html.Table(
                [
                    html.Thead(
                        [
                            html.Tr(
                                [
                                    html.Th("Library"),
                                    html.Th("Version"),
                                ]
                            )
                        ]
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Th(library),
                                    html.Th(version),
                                ]
                            )
                            for library, version in metadata.get("versions").items()
                        ]
                    ),
                ],
                className="table table-hover",
                style={"width": "20%"},
            ),
        ],
        id="summary",
    )


def page(path, metadata, routines_navbar, routines_content):
    return [
        topbar(),
        html.Div(
            html.Div(
                [
                    sidebar(path, routines_navbar),
                    html.Main(
                        [
                            html.Br(),
                            html.H1(path),
                            summary(metadata),
                            html.Div(routines_content, id="actions"),
                        ],
                        className="col-md-9 ms-sm-auto col-lg-10 px-md-4",
                    ),
                ],
                className="row",
            ),
            className="container-fluid",
        ),
    ]


def live(path=None):
    if path is None:
        return page(path, None, None, None)

    try:
        # read metadata and show in the live page
        with open(os.path.join(path, "meta.yml"), "r") as file:
            metadata = yaml.safe_load(file)
    except (FileNotFoundError, TypeError):
        return html.Div(children=[html.H2(f"Path {path} not available.")])

    # read routines from action runcard
    with open(os.path.join(path, "runcard.yml"), "r") as file:
        runcard = yaml.safe_load(file)

    routines_content = [html.Br(), html.Br(), html.Br(), html.H2("Actions")]
    routines_navbar = []
    for routine in runcard.get("actions").keys():
        routine_pretty = routine.replace("_", " ").title()
        routine_path = os.path.join(path, "data", routine)
        routines_navbar.append(
            html.Li(
                [
                    html.A(
                        routine_pretty,
                        className="nav-link",
                        href=f"#{routine}",
                    )
                ],
                className="nav-item",
            ),
        )
        routines_content.append(
            html.Div(
                [
                    # Empty spaces so that top bar does not cover the
                    # section title when using #links.
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.H3(routine_pretty),
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
        routines_content.append(html.Br())

    return page(path, metadata, routines_navbar, routines_content)
