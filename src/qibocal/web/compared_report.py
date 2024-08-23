import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.history import History
from qibocal.auto.output import Output
from qibocal.auto.task import Completed, TaskId
from qibocal.cli.report import generate_figures_and_report


@dataclass
class ComparedReport:
    """Class for comparison of two qibocal reports."""

    report_paths: list[Path]
    """List of paths of qibocal reports to compare."""
    folder: Path
    """Generated report path."""
    meta: dict
    """Meta data."""
    history: dict = field(default_factory=dict)
    """History of protocols with same id in both reports."""

    def __post_init__(self):
        """Store only protocols with same id in all the reports."""
        self.history = self.create_common_history()

    def history_uids(self) -> set[TaskId]:
        """Find the set of TakId in common between the reports."""
        experiment_ids = []
        for path in self.report_paths:
            report = Output.load(path)
            experiment_ids.append({x for x in report.history})

        common_experiment_ids = experiment_ids[0]
        for exp_id in experiment_ids[1:]:
            common_experiment_ids = common_experiment_ids & exp_id
        return common_experiment_ids

    def create_common_history(self) -> dict[TaskId, list[History]]:
        """Obtain histories of common TaskIds from the reports."""
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
    def routine_name(routine: TaskId) -> str:
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
        merged_figures = []
        for fig0, fig1 in zip(plots[0], plots[1]):
            for trace_0, trace_1 in zip(fig0.data, fig1.data):
                trace_0.legendgroup = None
                trace_1.legendgroup = None
                trace_0.legend = "legend"
                trace_1.legend = "legend2"

            if any(isinstance(trace, go.Heatmap) for trace in fig0.data):
                # TODO: check if this is valid for all protocols
                fig = make_subplots(rows=2, cols=2)
                for j, trace in enumerate(fig0.data):
                    # TODO: remove harcoded
                    # usually with heatmap first should be signal col = 1
                    # then phase (col = 2)
                    # if there is another element it should be a fit done on the first column
                    if isinstance(trace, go.Heatmap):
                        flatten_trace = np.array(trace.z).flatten()
                        flatten_fig_data = np.array(fig1.data[j].z).flatten()
                        trace.zmin = min(min(flatten_trace), min(flatten_fig_data))
                        trace.zmax = max(max(flatten_trace), max(flatten_fig_data))
                    fig.add_trace(
                        trace,
                        row=1,
                        col=1 if j % 2 == 0 else 2,
                    )

                for j, trace in enumerate(fig1.data):
                    # same as before but for second report everything goes
                    # on second row
                    if isinstance(trace, go.Heatmap):
                        trace.showscale = False
                        flatten_trace = np.array(trace.z).flatten()
                        flatten_fig_data = np.array(fig1.data[j].z).flatten()
                        trace.zmin = min(min(flatten_trace), min(flatten_fig_data))
                        trace.zmax = max(max(flatten_trace), max(flatten_fig_data))
                    fig.add_trace(
                        trace,
                        row=2,
                        col=1 if j % 2 == 0 else 2,
                    )
                fig.update_layout(
                    dict(
                        legend={
                            "title": f"{self.report_paths[0].name}",
                            "xref": "container",
                            "yref": "container",
                            "y": 0.8,
                        },
                        legend2={
                            "title": f"{self.report_paths[1].name}",
                            "xref": "container",
                            "yref": "container",
                            "y": 0.2,
                        },
                        showlegend=None,
                    )
                )
                merged_figures.append(fig)

            else:
                for trace in fig1.data:
                    fig0.add_trace(trace)

                fig0.update_layout(
                    dict(
                        legend={
                            "title": f"{self.report_paths[0].name}",
                            "xref": "container",
                            "yref": "container",
                            "y": 0.8,
                        },
                        legend2={
                            "title": f"{self.report_paths[1].name}",
                            "xref": "container",
                            "yref": "container",
                            "y": 0.2,
                        },
                        showlegend=None,
                    )
                )

                # this fixes weird behavior for comparing classification protocols
                if any(isinstance(trace, go.Contour) for trace in fig0.data):
                    fig0.update_layout(
                        dict(
                            legend={
                                "font": {"size": None},
                                "x": None,
                                "xanchor": None,
                                "yanchor": None,
                                "orientation": None,
                            },
                        )
                    )
                    for trace in fig0.data:
                        if isinstance(trace, go.Scatter):
                            if trace.legend == "legend":
                                if trace.name == "Qubit Fit: state 0":
                                    trace.legendgroup = "legend0_q0"
                                else:
                                    trace.legendgroup = "legend0_q1"
                            else:
                                if trace.name == "Qubit Fit: state 0":
                                    trace.legendgroup = "legend1_q0"
                                else:
                                    trace.legendgroup = "legend1_q1"
                merged_figures.append(fig0)

        buffer = io.StringIO()
        html_list = []
        for figure in merged_figures:
            figure.write_html(buffer, include_plotlyjs=True, full_html=False)
            buffer.seek(0)
            html_list.append(buffer.read())
        buffer.close()
        return "".join(html_list)

    def plotter(
        self, nodes: list[Completed], target: Union[QubitId, QubitPairId, list[QubitId]]
    ) -> tuple[str, str]:
        tables = []
        plots = []
        for node in nodes:
            report_figures, fitting_report = generate_figures_and_report(node, target)
            tables.append(fitting_report)
            plots.append(report_figures)

        figures_html = self.merge_plots(plots)

        try:
            merged_table = None
            merge_columns = {"Qubit", "Parameters"}
            for i, table in enumerate(tables):
                a = pd.read_html(io.StringIO(table))[0]
                a = a.rename(
                    columns={
                        col: f"{col}\n{self.report_paths[i].name}"
                        for col in set(a.columns) - merge_columns
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
        return figures_html, fitting_report
