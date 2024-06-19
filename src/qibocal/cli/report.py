import io
import json
import pathlib
from typing import Union

import plotly.graph_objects as go
import yaml
from jinja2 import Environment, FileSystemLoader
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.history import History
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Completed
from qibocal.cli.utils import META, RUNCARD
from qibocal.config import log
from qibocal.web.report import STYLES, TEMPLATES, Report

ReportOutcome = tuple[str, list[go.Figure]]
"""Report produced by protocol."""


def generate_figures_and_report(
    node: Completed, target: Union[QubitId, QubitPairId, list[QubitId]]
) -> ReportOutcome:
    """Calling protocol plot by checking if fit has been performed.

    It operates on a completed `node` and a specific protocol `target`, generating
    a report outcome (cf. `ReportOutcome`).
    """
    if node.results is None:
        # plot acquisition data
        return node.task.operation.report(data=node.data, fit=None, target=target)
    if target not in node.results:
        # plot acquisition data and message for unsuccessful fit
        figures = node.task.operation.report(data=node.data, fit=None, target=target)[0]
        return figures, "An error occurred when performing the fit."
    # plot acquisition and fit
    return node.task.operation.report(data=node.data, fit=node.results, target=target)


def plotter(
    node: Completed, target: Union[QubitId, QubitPairId, list[QubitId]]
) -> tuple[str, str]:
    """Run plotly pipeline for generating html.

    Performs conversions of plotly figures in html rendered code for completed
    node on specific target.

    """
    figures, fitting_report = generate_figures_and_report(node, target)
    buffer = io.StringIO()
    html_list = []
    for figure in figures:
        figure.write_html(buffer, include_plotlyjs=False, full_html=False)
        buffer.seek(0)
        html_list.append(buffer.read())
    buffer.close()
    all_html = "".join(html_list)
    return all_html, fitting_report


def report(path: pathlib.Path, history: History = None):
    """Report generation.

    Generates the report for protocol dumped in `path`.
    Executor can be passed to generate report on the fly.
    """

    if (path / "index.html").exists():  # pragma: no cover
        log.warning(f"Regenerating {path}/index.html")
    # load meta
    meta = json.loads((path / META).read_text())
    # load runcard
    runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))

    if history is None:
        history = History.load(path)

    css_styles = f"<style>\n{pathlib.Path(STYLES).read_text()}\n</style>"

    env = Environment(loader=FileSystemLoader(TEMPLATES))
    template = env.get_template("template.html")
    html = template.render(
        is_static=True,
        css_styles=css_styles,
        path=path,
        title=path.name,
        report=Report(
            path=path,
            targets=runcard.targets,
            history=history,
            meta=meta,
            plotter=plotter,
        ),
    )

    (path / "index.html").write_text(html)
