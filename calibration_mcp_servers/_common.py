"""Shared adapters for qibocal MCP servers."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from qibocal.auto.output import Output
from qibocal.cli.report import generate_figures_and_report
from qibocal.web.report import Report


def result(output_folder: str | Path) -> dict[str, str]:
    """Return a stable, JSON-compatible MCP response."""
    return {"output_folder": str(Path(output_folder).resolve())}


def report_content(output_folder: str | Path) -> dict[str, Any]:
    """Generate PNGs for every report figure and return the report folder."""
    path = Path(output_folder)
    output = Output.load(path)
    report = Report(
        path=path,
        history=output.history,
        meta=output.meta.dump(),
        plotter=lambda node, target: ("", ""),
    )
    artifacts_path = path / "agent_report"
    artifacts_path.mkdir(exist_ok=True)
    for task_id in report.history:
        node = report.history[task_id]
        for target in report.routine_targets(task_id):
            target_value: Any = target
            figures, _ = generate_figures_and_report(node, target_value)
            for index, figure in enumerate(figures):
                figure_name = f"{task_id}-{target}-figure-{index}.png"
                _export_png(figure, artifacts_path / figure_name)
    return {"output_folder": str(path.resolve())}


def _export_png(figure: go.Figure, path: Path) -> None:
    """Convert a Plotly report figure to Matplotlib and export it as PNG."""
    axis_names = sorted(
        {getattr(trace, "xaxis", None) or "x" for trace in figure.data},
        key=lambda name: int(name[1:] or 1),
    )
    matplotlib_figure, axes = plt.subplots(
        1,
        len(axis_names),
        figsize=(6 * len(axis_names), 5),
        squeeze=False,
    )
    axis_map = dict(zip(axis_names, axes[0], strict=True))

    for trace in figure.data:
        axis = axis_map[getattr(trace, "xaxis", None) or "x"]
        _draw_trace(axis, trace)

    for axis_name, axis in axis_map.items():
        suffix = axis_name[1:]
        plotly_xaxis = getattr(figure.layout, f"xaxis{suffix}")
        plotly_yaxis = getattr(figure.layout, f"yaxis{suffix}")
        axis.set_xlabel(plotly_xaxis.title.text or "")
        axis.set_ylabel(plotly_yaxis.title.text or "")
        if plotly_xaxis.range:
            axis.set_xlim(plotly_xaxis.range)
        if plotly_yaxis.range:
            axis.set_ylim(plotly_yaxis.range)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels)

    if figure.layout.title.text:
        matplotlib_figure.suptitle(figure.layout.title.text)
    matplotlib_figure.tight_layout()
    matplotlib_figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(matplotlib_figure)


def _draw_trace(axis: Any, trace: Any) -> None:
    """Draw one supported Plotly trace on a Matplotlib axis."""
    label = trace.name if trace.showlegend is not False else "_nolegend_"
    if isinstance(trace, go.Scatter):
        mode = trace.mode or "lines"
        color = trace.line.color or trace.marker.color
        if "lines" in mode:
            line_styles = {"dash": "--", "dot": ":", "dashdot": "-."}
            axis.plot(
                trace.x,
                trace.y,
                color=color,
                label=label,
                linewidth=trace.line.width,
                linestyle=line_styles.get(trace.line.dash, "-"),
                marker="o" if "markers" in mode else None,
                markersize=trace.marker.size,
            )
        elif "markers" in mode:
            axis.scatter(
                trace.x,
                trace.y,
                c=color,
                label=label,
                s=np.square(trace.marker.size or 6),
            )
        return
    if isinstance(trace, go.Heatmap):
        axis.pcolormesh(trace.x, trace.y, trace.z, shading="auto")
        return
    if isinstance(trace, go.Contour):
        axis.contour(trace.x, trace.y, trace.z)
        return
    if isinstance(trace, go.Bar):
        axis.bar(trace.x, trace.y, label=label, color=trace.marker.color)
        return
    msg = f"Unsupported Plotly trace type: {trace.type}"
    raise TypeError(msg)
