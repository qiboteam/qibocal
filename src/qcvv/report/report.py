# -*- coding: utf-8 -*-
import os
import pathlib

from jinja2 import Environment, FileSystemLoader


def parse_dash(element):
    if element._namespace == "dash_html_components":
        # Convert dash HTML components to html tags
        if isinstance(element.children, list):
            tag = element._type.lower()
            code = "".join(parse_dash(child) for child in element.children)
            return f"<{tag}>\n{code}</{tag}>\n"

        elif isinstance(element.children, str):
            tag = element._type.lower()
            return f"<{tag}>{element.children}</{tag}>\n"

        elif element.children is None:
            tag = element._type.lower()
            return f"<{tag}>\n"

        else:
            raise NotImplementedError(f"Failed to parse {element}.")

    elif element._type == "Graph":
        # Parse graphs using ``fig.write_html``
        from qcvv.live import get_graph

        fig = get_graph(0, element.id)
        fig.write_html("fig.html", include_plotlyjs=False, full_html=False)
        with open("fig.html", "r") as file:
            code = file.read()
        os.remove("fig.html")
        return code

    else:
        # Skip non-HTML (interactive) dash components
        return ""


def create_report(path):
    """Creates an HTML report for the data in the given path.

    The report is created based on the live plotting layout.
    """
    from qcvv.live import serve_layout

    layout = serve_layout(path)
    layout_html = parse_dash(layout)

    env = Environment(loader=FileSystemLoader(pathlib.Path(__file__).parent))
    template = env.get_template("template.html")
    report = template.render(body=layout_html)

    with open(f"{path}/report.html", "w") as file:
        file.write(report)
