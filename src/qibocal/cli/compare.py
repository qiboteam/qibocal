import json
from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go
import yaml
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
    combined_report_path = generate_output_folder(folder, force)
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
    comparison_report = CompareReportBuilder(builders)
    comparison_report.run(combined_report_path)


class CompareReportBuilder:

    def __init__(self, report_builders: List[ReportBuilder]):
        self.report_builders = report_builders
        self.metadata = report_builders[0].metadata
        self.path = self.title = report_builders[0].path
        self.targets = report_builders[0].targets
        self.executor = report_builders[0].executor

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

        merged_table = pd.read_html(tables[0])[0].rename(columns={"Values": "Values_0"})
        for i, table in enumerate(tables[1:]):
            a = pd.read_html(table)[0]
            merged_table = pd.merge(merged_table, a, on=["Qubit", "Parameters"]).rename(
                columns={"Values": f"Values_{i + 1}"}
            )
        merged_plot_data = plots[0][0].data
        for plot in plots[1:]:
            merged_plot_data = merged_plot_data + plot[0].data
        merged_plot = go.Figure(data=merged_plot_data, layout=plot[0].layout)
        return merged_plot.to_html(), merged_table.to_html(
            classes="fitting-table", index=False, border=0, escape=False
        )

    def run(self, path):
        """Generation of html report."""
        from qibocal.web.report import create_report

        create_report(path, self)
