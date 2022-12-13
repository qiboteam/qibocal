# -*- coding: utf-8 -*-
import os
import pathlib
import subprocess

import yaml
from flask import Flask, redirect, render_template, request

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

    report = []
    if path is not None:
        try:
            report.append(ReportBuilder(path))
        except (FileNotFoundError, TypeError):
            pass

    return render_template(
        "template.html",
        version=__version__,
        folders=folders,
        reports=report,
    )


@server.route("/", methods=["GET", "POST"])
@server.route("/data/<path>", methods=["GET", "POST"])
def qq_compare():
    if request.method == "POST":
        if request.form["qq-compare"] == "compare":
            list_of_folders = ""
            selected_folders = request.form.getlist("list_of_folders")
            for folder in selected_folders:
                list_of_folders = list_of_folders + folder + " "

            # execute shell command qq-compare
            command = "qq-compare " + list_of_folders
            try:
                command_output = subprocess.check_output([command], shell=True)

            except subprocess.CalledProcessError as e:
                #return "An error occurred while trying to generate qq-report.<br>Please check folder compatility."
                #return redirect("/", code=302)
                folders = [
                    folder
                    for folder in reversed(sorted(os.listdir(os.getcwd())))
                    if os.path.isdir(folder) and "meta.yml" in os.listdir(folder)
                ]
                report = []
                return render_template(
                    "template.html",
                    version=__version__,
                    folders=folders,
                    reports=report,
                    error="An error occurred while trying to generate qq-report. Please check folder compatility."
                )

            return redirect("/data/qq-compare", code=302)

        elif request.form["qq-compare"] == "combine":
            folders = [
                folder
                for folder in reversed(sorted(os.listdir(os.getcwd())))
                if os.path.isdir(folder) and "meta.yml" in os.listdir(folder)
            ]

            list_of_reports = []
            selected_folders = request.form.getlist("list_of_folders")
            for folder in selected_folders:
                try:
                    list_of_reports.append(ReportBuilder(folder))
                except (FileNotFoundError, TypeError):
                    pass

            return render_template(
                "template.html",
                version=__version__,
                folders=folders,
                reports=list_of_reports,
            )
