# -*- coding: utf-8 -*-
import os
import pathlib

import yaml
from flask import Flask, render_template

from qcvv import __version__
from qcvv.cli.builders import ReportBuilder

server = Flask(__name__)


@server.route("/")
@server.route("/data/<path>")
def page(path=None):
    folders = [
        folder
        for folder in sorted(os.listdir(os.getcwd()))
        if os.path.isdir(folder) and "meta.yml" in os.listdir(folder)
    ]
    if path is None:
        render_template(
            "template.html", version=__version__, folders=folders, path=None
        )

    try:
        report = ReportBuilder(path)
    except (FileNotFoundError, TypeError):
        return render_template(
            "template.html", version=__version__, folders=folders, path=None
        )

    return render_template(
        "template.html",
        version=__version__,
        folders=folders,
        path=path,
        title=path,
        report=report,
    )
