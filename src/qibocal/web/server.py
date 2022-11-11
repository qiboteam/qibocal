import os
import pathlib

import yaml
from flask import Flask, render_template

from qibocal import __version__
from qibocal.cli.builders import ReportBuilder

server = Flask(__name__)


@server.route("/")
@server.route("/data/<path>")
def page(path=None):
    folders = [
        folder
        for folder in reversed(sorted(os.listdir(os.getcwd())))
        if os.path.isdir(folder) and "meta.yml" in os.listdir(folder)
    ]

    report = None
    if path is not None:
        try:
            report = ReportBuilder(path)
        except (FileNotFoundError, TypeError):
            pass

    return render_template(
        "template.html",
        version=__version__,
        folders=folders,
        report=report,
    )
