# -*- coding: utf-8 -*-
import os
import pathlib
import tempfile

import yaml
from jinja2 import Environment, FileSystemLoader

from qcvv import __version__, calibrations, plots
from qcvv.cli.builders import ReportBuilder


def get_figure(folder, routine, method, qubit, format):
    figure = getattr(plots, method)(folder, routine.name, qubit, format)
    with tempfile.NamedTemporaryFile() as temp:
        figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
        fightml = temp.read().decode("utf-8")
    return fightml


def create_report(path):
    """Creates an HTML report for the data in the given path."""
    # TODO: Consider moving this method to the report builder
    filepath = pathlib.Path(__file__)

    with open(os.path.join(filepath.with_name("static"), "styles.css"), "r") as file:
        css_styles = f"<style>\n{file.read()}\n</style>"

    report = ReportBuilder(path)
    env = Environment(loader=FileSystemLoader(filepath.with_name("templates")))
    # env.globals.update(get_figure=get_figure)
    template = env.get_template("template.html")

    html = template.render(
        is_static=True,
        css_styles=css_styles,
        version=__version__,
        report=report,
    )

    with open(os.path.join(path, "report.html"), "w") as file:
        file.write(html)
