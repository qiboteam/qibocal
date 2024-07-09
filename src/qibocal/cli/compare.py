import datetime
import json
from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from qibo.backends import GlobalBackend
from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import TaskId
from qibocal.cli.report import META, RUNCARD, ReportBuilder, generate_figures_and_report
from qibocal.cli.utils import generate_output_folder


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
    combined_meta = json.loads((path_1 / META).read_text())
    e = datetime.datetime.now(datetime.timezone.utc)
    combined_meta["start-time"] = e.strftime("%H:%M:%S")
    combined_meta["date"] = e.strftime("%Y-%m-%d")
    combined_report_path = generate_output_folder(folder, force)
    combined_meta["title"] = combined_report_path.name
    paths = [path_1, path_2]
    builders = []
    for path in paths:
        # load meta
        meta = json.loads((path / META).read_text())
        # load runcard
        runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))

        # set backend, platform and qubits
        GlobalBackend.set_backend(backend=meta["backend"], platform=meta["platform"])

        # load executor
        executor = Executor.load(runcard, path, targets=runcard.targets)
        # produce html
        builder = ReportBuilder(path, runcard.targets, executor, meta)
        builders.append(builder)
    comparison_report = CompareReportBuilder(builders, folder)
    comparison_report.run(combined_report_path)
    e = datetime.datetime.now(datetime.timezone.utc)
    combined_meta["end-time"] = e.strftime("%H:%M:%S")
    (combined_report_path / META).write_text(json.dumps(combined_meta, indent=4))


class CompareReportBuilder:

    def __init__(self, report_builders: List[ReportBuilder], path: Path):
        """Combine two reports.

        Args:
            report_builders (List[ReportBuilder]): List containing the two ReportBuilder
                objects to be combined.
            path (Path): Path of the combined report.
        """
        self.report_builders = report_builders
        self.metadata = self.combine_metadata(report_builders)
        self.path = self.title = path
        self.targets = report_builders[0].targets
        self.executor = report_builders[0].executor

    @staticmethod
    def combine_metadata(report_builders):
        metadata = report_builders[0].metadata

        combined_keys = ["platform", "date", "start-time", "end-time"]
        for key in combined_keys:
            report_dates = [report.metadata[key] for report in report_builders]
            metadata[key] = " | ".join(report_dates)
        return metadata

    def history_uids(self):
        experiment_ids = []
        for report_builder in self.report_builders:
            experiment_ids.append({x for x in report_builder.history.keys()})

        common_experiment_ids = experiment_ids[0]
        for exp_id in experiment_ids[1:]:
            common_experiment_ids = common_experiment_ids & exp_id
        return common_experiment_ids

    @property
    def history(self):
        # find common history
        common_experiment_ids = self.history_uids()
        history = {}
        # create a dictionary of task_uid: List[Completed(...)]
        for common_experiment_id in common_experiment_ids:
            history[common_experiment_id] = []
            for report_builder in self.report_builders:
                history[common_experiment_id].append(
                    report_builder.history[common_experiment_id]
                )
        return history

    def routine_name(self, routine, iteration):  # can be defined outside the class
        """Prettify routine's name for report headers."""
        name = routine.replace("_", " ").title()
        return f"{name} - {iteration}"

    def routine_targets(self, task_id: TaskId):  # ????
        """Get local targets parameter from Task if available otherwise use global one."""
        # local_qubits = self.history[task_id].task.targets
        # return local_qubits if len(local_qubits) > 0 else self.targets
        return self.targets

    def merge_plots(self, plots: List[List[go.Figure]]) -> str:
        """Merge plots from different reports.

        Scatter plots are plotted in the same figure.
        Heatmaps are vertically stacked.
        """
        if plots[0][0].data[0].type == "scatter":
            merged_plot_data = plots[0][0].data
            for plot in plots[1:]:
                merged_plot_data = merged_plot_data + plot[0].data
            fig = go.Figure(data=merged_plot_data, layout=plots[0][0].layout).to_html()
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
            fig = fig.to_html()

        return fig

    def single_qubit_plot(self, task_id: TaskId, qubit: QubitId):
        """Generate single qubit plot."""
        tables = []
        plots = []
        for report_builder in self.report_builders:
            _, table = report_builder.single_qubit_plot(task_id, qubit)
            node = report_builder.history[task_id]
            figures, _ = generate_figures_and_report(node, qubit)
            tables.append(table)
            plots.append(figures)

        try:
            merge_columns = {"Qubit", "Parameters"}
            merged_table = pd.read_html(tables[0])[0]
            merged_table = merged_table.rename(
                columns={
                    col: f"{col}_0" for col in set(merged_table.columns) - merge_columns
                }
            )
            for i, table in enumerate(tables[1:]):
                a = pd.read_html(table)[0]
                merged_table = pd.merge(merged_table, a, on=list(merge_columns)).rename(
                    columns={
                        col: f"{col}_{i+1}" for col in set(a.columns) - merge_columns
                    }
                )
        except ValueError:
            merged_table = pd.DataFrame(
                [], columns=["An error occurred when performing the fit."]
            )
        merged_plot = self.merge_plots(plots)
        return merged_plot, merged_table.to_html(
            classes="fitting-table", index=False, border=0, escape=False
        )

    def run(self, path):
        """Generation of html report."""
        from qibocal.web.report import create_report

        create_report(path, self)
