# -*- coding: utf-8 -*-
import os
import pathlib

from jinja2 import Environment, FileSystemLoader

from qibocal import __version__
from qibocal.cli.builders import ReportBuilder


def create_report(path):
    """Creates an HTML report for the data in the given path."""
    filepath = pathlib.Path(__file__)

    with open(os.path.join(filepath.with_name("static"), "styles.css")) as file:
        css_styles = f"<style>\n{file.read()}\n</style>"

    report = ReportBuilder(path)
    env = Environment(loader=FileSystemLoader(filepath.with_name("templates")))
    template = env.get_template("template.html")

    html = template.render(
        is_static=True,
        css_styles=css_styles,
        version=__version__,
        report=report,
    )

    with open(os.path.join(path, "index.html"), "w") as file:
        file.write(html)
