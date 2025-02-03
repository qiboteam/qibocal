from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from qibocal.auto.history import DummyHistory
from qibocal.auto.output import Output
from qibocal.web.compared_report import ComparedReport
from qibocal.web.report import STYLES, TEMPLATES, report_css_styles


def initialize_combined_report(
    report_path: Path, output_folder: Path, force: bool
) -> tuple[Output, Path]:
    """Initialisation of the output.

    Create the report directory and set up start-finish time, report title.

    Args:
        report_path (pathlib.Path): path of the folder containing one of the initial reports.
        output_folder (pathlib.Path): path of the folder containing the combined report.
        force (bool): if set to true, overwrites output_folder (if it already exists).
    """
    combined_meta = Output.load(report_path).meta
    combined_meta.start()
    combined_report = Output(history=DummyHistory(), meta=combined_meta)
    combined_report_path = combined_report.mkdir(output_folder, force)
    combined_meta.title = combined_report_path.name
    combined_meta.end()
    combined_report.meta = combined_meta
    return combined_report, combined_report_path


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
    paths = [path_1, path_2]
    env = Environment(loader=FileSystemLoader(TEMPLATES))
    template = env.get_template("template.html")
    combined_report, combined_report_path = initialize_combined_report(
        path_1, output_folder=folder, force=force
    )

    html = template.render(
        is_static=True,
        css_styles=report_css_styles(STYLES),
        path=combined_report_path,
        title=combined_report.meta.title,
        report=ComparedReport(
            report_paths=paths,
            folder=combined_report_path,
            meta=combined_report.meta.dump(),
        ),
    )
    combined_report.dump(combined_report_path)
    (combined_report_path / "index.html").write_text(html)
