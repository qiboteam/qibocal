import os
import pathlib

from jinja2 import Environment, FileSystemLoader

from qibocal import __version__
from qibocal.cli.report import ReportBuilder

WEB_DIR = pathlib.Path(__file__).parent
STYLES = WEB_DIR / "static" / "styles.css"
TEMPLATES = WEB_DIR / "templates"


def create_report(path, report: ReportBuilder):
    """Creates an HTML report for the data in the given path."""
    with open(STYLES) as file:
        css_styles = f"<style>\n{file.read()}\n</style>"

    env = Environment(loader=FileSystemLoader(TEMPLATES))
    template = env.get_template("template.html")
    html = template.render(
        is_static=True,
        css_styles=css_styles,
        version=__version__,
        report=report,
    )

    with open(os.path.join(path, "index.html"), "w") as file:
        file.write(html)
