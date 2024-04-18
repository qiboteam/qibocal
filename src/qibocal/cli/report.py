import json
import pathlib
import tempfile
from typing import Optional, Union

import plotly.graph_objects as go
import yaml
from jinja2 import Environment, FileSystemLoader
from qibo.backends import GlobalBackend
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.execute import Executor
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Completed
from qibocal.cli.utils import META, RUNCARD
from qibocal.config import log
from qibocal.web.report import STYLES, TEMPLATES, Report


def generate_figures_and_report(
    node: Completed, target: Union[QubitId, QubitPairId, list[QubitId]]
) -> tuple[str, list[go.Figure]]:
    """Calling protocol plot by checking if fit has been performed.

    Args:
        node (Completed): completed node
        target (Union[QubitId, QubitPairId, list[QubitId]]): protocol target

    Returns:
        tuple[str, list[go.Figure]]: report outcome
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
    """Run plotly pipeline for generating html

    Args:
        node (Completed): _description_
        target (Union[QubitId, QubitPairId, list[QubitId]]): _description_

    Returns:
        tuple[str, str]: plotly figures in html and tables
    """
    figures, fitting_report = generate_figures_and_report(node, target)
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        html_list = []
        for figure in figures:
            figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
            temp.seek(0)
            fightml = temp.read().decode("utf-8")
            html_list.append(fightml)

    all_html = "".join(html_list)
    return all_html, fitting_report


def report(path: pathlib.Path, executor: Optional[Executor] = None):
    """Report command

    Args:
        path (pathlib.Path): path where qibocal protocol is dumped
        executor (Optional[Executor], optional): Qibocal executor, if None it is loaded from path.
                                                 Defaults to None.
    """

    if path.exists():
        log.warning(f"Regenerating {path}/index.html")
    # load meta
    meta = json.loads((path / META).read_text())
    # load runcard
    runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))

    # set backend, platform and qubits
    GlobalBackend.set_backend(backend=meta["backend"], platform=meta["platform"])

    # load executor
    if executor is None:
        executor = Executor.load(runcard, path, targets=runcard.targets)
        # produce html
        list(executor.run(mode=ExecutionMode.report))

    with open(STYLES) as file:
        css_styles = f"<style>\n{file.read()}\n</style>"

    env = Environment(loader=FileSystemLoader(TEMPLATES))
    template = env.get_template("template.html")
    html = template.render(
        is_static=True,
        css_styles=css_styles,
        path=path,
        title=path.name,
        report=Report(
            path=path,
            targets=executor.targets,
            history=executor.history,
            meta=meta,
            plotter=plotter,
        ),
    )

    (path / "index.html").write_text(html)
