# -*- coding: utf-8 -*-
import os
import pathlib
import tempfile

import yaml
from jinja2 import Environment, FileSystemLoader

from qcvv import __version__, plots
from qcvv.data import Dataset


def get_figure(folder, routine, method, qubit, format):
    figure = getattr(plots, method)(folder, routine, qubit, format)
    with tempfile.NamedTemporaryFile() as temp:
        figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
        fightml = temp.read().decode("utf-8")
    return fightml


def create_report(path):
    """Creates an HTML report for the data in the given path."""
    filepath = pathlib.Path(__file__)

    with open(os.path.join(filepath.with_name("static"), "styles.css"), "r") as file:
        css_styles = f"<style>\n{file.read()}\n</style>"

    # read metadata and show in the live page
    with open(os.path.join(path, "meta.yml"), "r") as file:
        metadata = yaml.safe_load(file)

    # read routines from action runcard
    with open(os.path.join(path, "runcard.yml"), "r") as file:
        runcard = yaml.safe_load(file)

    # read plot configuration yaml
    with open(pathlib.Path(__file__).with_name("plots.yml"), "r") as file:
        plotters = yaml.safe_load(file)

    env = Environment(loader=FileSystemLoader(filepath.with_name("templates")))
    env.globals.update(get_figure=get_figure)
    template = env.get_template("template.html")
    report = template.render(
        is_static=True,
        css_styles=css_styles,
        version=__version__,
        path=path,
        metadata=metadata,
        runcard=runcard,
        plotters=plotters,
    )

    with open(os.path.join(path, "report.html"), "w") as file:
        file.write(report)
