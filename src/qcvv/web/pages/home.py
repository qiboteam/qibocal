# -*- coding: utf-8 -*-
import os

import dash
from dash import dcc, html

dash.register_page(__name__, path="/")


live = dash.page_registry.get("pages.live")

layout = html.Div(
    [
        html.H1("Multi-page app with Dash Pages"),
        html.Div(
            [
                html.Div(
                    dcc.Link(
                        f"{folder}",
                        href=live.get("path_template").replace("<path>", folder),
                    )
                )
                for folder in os.listdir(os.getcwd())
                if os.path.isdir(folder) and "meta.yml" in os.listdir(folder)
            ]
        ),
    ]
)
