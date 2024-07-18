import io
import pathlib
from pathlib import Path
from typing import Union

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from plotly.subplots import make_subplots
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.output import Output
from qibocal.auto.task import Completed, TaskId
from qibocal.cli.report import generate_figures_and_report
from qibocal.web.report import STYLES, TEMPLATES


class FakeHistory:

    def flush(self, output=None):
        pass

    def items(self):
        return tuple()


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
    combined_report = Output(history=FakeHistory(), meta=combined_meta)
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


class ComparedReport:

    def __init__(self, report_paths: list[Path], folder, meta):
        self.report_paths = report_paths
        self.folder = folder
        self.history = self.create_common_history()
        self.meta = meta

    def history_uids(self):
        experiment_ids = []
        for path in self.report_paths:
            report = Output.load(path)
            experiment_ids.append({x for x in report.history})

        common_experiment_ids = experiment_ids[0]
        for exp_id in experiment_ids[1:]:
            common_experiment_ids = common_experiment_ids & exp_id
        return common_experiment_ids

    def create_common_history(self):
        common_experiment_ids = self.history_uids()
        history = {}
        # create a dictionary of task_uid: List[Completed(...)]
        for common_experiment_id in common_experiment_ids:
            history[common_experiment_id] = []
            for path in self.report_paths:
                report = Output.load(path)
                history[common_experiment_id].append(
                    report.history[common_experiment_id]
                )
        return history

    @staticmethod
    def routine_name(routine: TaskId):
        """Prettify routine's name for report headers."""
        return routine.id.title()

    def routine_targets(self, task_id: TaskId):
        """Extract local targets parameter from Task.

        If not available use the global ones.
        """
        local_targets = self.history[task_id][0].task.targets
        return local_targets if len(local_targets) > 0 else self.meta["targets"]

    def merge_plots(self, plots: list[list[go.Figure]]) -> list[go.Figure]:
        """Merge plots from different reports.

        Scatter plots are plotted in the same figure.
        Heatmaps are vertically stacked.
        """
        if plots[0][0].data[0].type == "scatter":
            merged_plot_data = plots[0][0].data
            for plot in plots[1:]:
                merged_plot_data = merged_plot_data + plot[0].data
            fig = [go.Figure(data=merged_plot_data, layout=plots[0][0].layout)]
        elif plots[0][0].data[0].type == "heatmap":
            fig = make_subplots(rows=2, cols=2)
            for i, plot in enumerate(plots):
                fig.append_trace(
                    plot[0].data[0],
                    row=i + 1,
                    col=1,
                )
                fig.append_trace(
                    plot[0].data[1],
                    row=i + 1,
                    col=2,
                )
                # heatmap fit
                fig.append_trace(
                    plot[0].data[2],
                    row=i + 1,
                    col=1,
                )
            fig.update_layout(legend=plots[0][0].layout.legend)
            fig = [fig]

        return fig

    def plotter(
        self, nodes: list[Completed], target: Union[QubitId, QubitPairId, list[QubitId]]
    ) -> tuple[str, str]:
        tables = []
        plots = []
        for node in nodes:
            report_figures, fitting_report = generate_figures_and_report(node, target)
            tables.append(fitting_report)
            plots.append(report_figures)

        figures = self.merge_plots(plots)
        buffer = io.StringIO()
        html_list = []
        for figure in figures:
            figure.write_html(buffer, include_plotlyjs=False, full_html=False)
            buffer.seek(0)
            html_list.append(buffer.read())
        buffer.close()
        all_html = "".join(html_list)

        try:
            merged_table = None
            merge_columns = {"Qubit", "Parameters"}
            for i, table in enumerate(tables):
                a = pd.read_html(table)[0]
                a = a.rename(
                    columns={
                        col: f"{col}_{i}" for col in set(a.columns) - merge_columns
                    }
                )
                if merged_table is None:
                    merged_table = a
                else:
                    merged_table = pd.merge(merged_table, a, on=list(merge_columns))
        except ValueError:
            merged_table = pd.DataFrame(
                [], columns=["An error occurred when performing the fit."]
            )
        fitting_report = merged_table.to_html(
            classes="fitting-table", index=False, border=0, escape=False
        )
        return all_html, fitting_report
