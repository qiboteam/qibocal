import pathlib
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from qibocal.auto.history import DummyHistory
from qibocal.auto.output import Output
from qibocal.web.compared_report import ComparedReport
from qibocal.web.report import STYLES, TEMPLATES


def compare_reports(folder: Path, path_1: Path, path_2: Path, force: bool):
    """Report comparison generation.

    Currently only two reports can be combined together. Only tasks with the same id can be merged.
    Tables display data from both reports side by side.
    Plots display data from both reports.

    Args:
        folder (pathlib.Path): path of the folder containing the combined report.
        path_1 (pathlib.Path): path of the first report to be compared.
        path_2 (pathlib.Path): path of the second report to be compared.
        force (bool): if set to true, overwrites folder (if it already exists).

    """
    combined_meta = Output.load(path_1).meta
    combined_meta.start()
    paths = [path_1, path_2]

    css_styles = f"<style>\n{pathlib.Path(STYLES).read_text()}\n</style>"

    env = Environment(loader=FileSystemLoader(TEMPLATES))
    template = env.get_template("template.html")
    combined_report = Output(history=DummyHistory(), meta=combined_meta)
    combined_report_path = combined_report.mkdir(folder, force)
    combined_meta.title = combined_report_path.name
    combined_meta.end()
    combined_report.meta = combined_meta

    html = template.render(
        is_static=True,
        css_styles=css_styles,
        path=folder,
        title=combined_report_path.name,
        report=ComparedReport(
            report_paths=paths,
            folder=folder,
            meta=combined_meta.dump(),
        ),
    )
    combined_report.dump(folder)
    (folder / "index.html").write_text(html)
