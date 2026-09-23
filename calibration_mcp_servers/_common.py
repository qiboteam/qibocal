"""Shared adapters for qibocal MCP servers."""

import asyncio
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import yaml

from qibocal.auto.output import Output
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Action
from qibocal.cli.report import generate_figures_and_report
from qibocal.web.report import Report


def make_output_path(parent_folder: str | Path, platform: str) -> Path:
    """Create and return a timestamped platform output directory."""
    now = datetime.now().astimezone()
    path = (
        Path(parent_folder)
        / platform
        / now.strftime("%Y%m%d")
        / now.strftime("%H:%M:%S")
    )
    path.mkdir(parents=True)
    return path


def make_runcard(
    experiments: list[dict[str, Any]],
    targets: list[Any] | None = None,
    platform: str = "mock",
    backend: str = "qibolab",
    update: bool = True,
) -> Runcard:
    """Build a qibocal runcard from JSON-compatible experiment definitions."""
    actions = []
    for index, experiment in enumerate(experiments, start=1):
        action = dict(experiment)
        action.setdefault("id", f"experiment-{index}")
        action.setdefault("update", update)
        actions.append(Action.cast(action))
    return Runcard(
        actions=actions,
        targets=targets,
        platform=platform,
        backend=backend,
        update=update,
    )


def write_runcard(runcard: Runcard, output_dir: Path) -> Path:
    """Dump a runcard to a standalone YAML file consumable by `qq run`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runcard_path = output_dir / "runcard.yml"
    with open(runcard_path, "w") as file:
        yaml.safe_dump(asdict(runcard), file)
    return runcard_path


def _qq_executable() -> str:
    """Resolve the `qq` entry point installed alongside the running interpreter.

    Using the `qq` from ``sys.executable``'s directory guarantees the subprocess
    runs with the same Python environment as the MCP server, avoiding a bare
    ``qq`` on ``PATH`` that may point at a different interpreter lacking
    ``qibocal``.
    """
    qq = Path(sys.executable).parent / "qq"
    if not qq.exists():
        raise RuntimeError(
            f"Could not find `qq` next to the running interpreter ({sys.executable})."
        )
    return str(qq)


async def run_qq(
    runcard_path: Path,
    output_path: Path,
    update: bool,
    partition: str | None,
) -> str:
    """Run `qq run`, optionally submitting it via Slurm `sbatch` and checking its exit status."""
    qq_command = [
        _qq_executable(),
        "run",
        str(runcard_path),
        "-o",
        str(output_path),
        "-f",
        "--update" if update else "--no-update",
    ]
    if partition is None:
        command = qq_command
    else:
        command = [
            "srun",
            "--time=01:00:00",
            f"--partition={partition}",
        ] + qq_command

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"'{' '.join(command)}' failed with exit code {process.returncode}:\n"
            f"{stderr.decode()}"
        )
    return stdout.decode()


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
        color = _matplotlib_color(trace.line.color or trace.marker.color)
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
        axis.bar(
            trace.x,
            trace.y,
            label=label,
            color=_matplotlib_color(trace.marker.color),
        )
        return
    msg = f"Unsupported Plotly trace type: {trace.type}"
    raise TypeError(msg)


def _matplotlib_color(color: Any) -> Any:
    """Convert Plotly's CSS rgb colors to Matplotlib-compatible values."""
    if not isinstance(color, str):
        return color

    match = re.fullmatch(
        r"rgba?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*"
        r"(\d+(?:\.\d+)?)(?:\s*,\s*(\d+(?:\.\d+)?))?\s*\)",
        color,
    )
    if match is None:
        return color

    red, green, blue, alpha = (float(value) for value in match.groups("1"))
    return (red / 255, green / 255, blue / 255, alpha)
