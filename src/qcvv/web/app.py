# -*- coding: utf-8 -*-
from dash import Dash, Input, Output, callback, dcc, html

from qcvv.web import callbacks
from qcvv.web.layouts import home, live

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)


@callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(url):
    if url == "/":
        return home()
    elif url[:5] == "/live":
        path = url.split("/")[-1]
        return live(path)
    else:
        return html.H1("This page does not exist.")
