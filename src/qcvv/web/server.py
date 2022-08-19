# -*- coding: utf-8 -*-
import os
import pathlib

import yaml
from flask import Flask, render_template

from qcvv import __version__

server = Flask(__name__)


@server.route("/")
@server.route("/data/<path>")
def page(path=None):
    folders = [
        folder
        for folder in reversed(sorted(os.listdir(os.getcwd())))
        if os.path.isdir(folder) and "meta.yml" in os.listdir(folder)
    ]
    if path is None:
        render_template(
            "template.html", version=__version__, folders=folders, path=None
        )

    try:
        # read metadata and show in the live page
        with open(os.path.join(path, "meta.yml"), "r") as file:
            metadata = yaml.safe_load(file)
    except (FileNotFoundError, TypeError):
        return render_template(
            "template.html", version=__version__, folders=folders, path=None
        )

    # read routines from action runcard
    with open(os.path.join(path, "runcard.yml"), "r") as file:
        runcard = yaml.safe_load(file)

    # read plot configuration yaml
    with open(pathlib.Path(__file__).with_name("plots.yml"), "r") as file:
        plotters = yaml.safe_load(file)

    return render_template(
        "template.html",
        version=__version__,
        folders=folders,
        path=path,
        metadata=metadata,
        runcard=runcard,
        plotters=plotters,
    )
