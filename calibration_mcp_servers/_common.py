"""Shared adapters for qibocal MCP servers."""

import uuid
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go

from qibocal.auto.output import Output
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Action
from qibocal.cli.report import generate_figures_and_report
from qibocal.cli.run import protocols_execution
from qibocal.cli.update import update
from qibocal.web.report import Report


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


def result(output_folder: str | Path) -> dict[str, str]:
    """Return a stable, JSON-compatible MCP response."""
    return {"output_folder": str(Path(output_folder).resolve())}


def run_full_workflow(
    experiments: list[dict[str, Any]],
    output_folder: str | Path | None = None,
    targets: list[Any] | None = None,
    platform: str = "mock",
    backend: str = "qibolab",
    force: bool = False,
) -> dict[str, Any]:
    """Run acquisition and fitting, report results, and defer platform updates."""
    if output_folder is None:
        output_folder = f"run-{uuid.uuid4().hex[:8]}"
    path = Path(output_folder)
    runcard = make_runcard(experiments, targets, platform, backend, update=False)
    protocols_execution(runcard, path, force, update=False)
    return {
        **report_content(path),
        "platform_update_pending": True,
        "platform_update_message": (
            "Platform updates were not applied. Ask the user for approval, then "
            "call update_platform with this output_folder."
        ),
    }


def update_platform(
    output_folder: str | Path,
    skip_qubits: list[str] | None = None,
) -> dict[str, str]:
    """Apply platform updates after explicit user approval."""
    update(Path(output_folder), cast(Any, skip_qubits))
    return result(output_folder)


def report_content(output_folder: str | Path) -> dict[str, Any]:
    """Build report HTML that MCP clients can render in the agent conversation."""
    path = Path(output_folder)
    output = Output.load(path)
    report = Report(
        path=path,
        history=output.history,
        meta=output.meta.dump(),
        plotter=lambda node, target: ("", ""),
    )
    sections = [
        f"<h1>Qibocal report: {path.name}</h1>",
        _versions_table(report.meta.get("versions", {})),
    ]
    artifacts_path = path / "agent_report"
    artifacts_path.mkdir(exist_ok=True)
    png_files = []
    export_errors = []
    for task_id in report.history:
        node = report.history[task_id]
        sections.append(f"<h2>{report.routine_name(task_id)}</h2>")
        for target in report.routine_targets(task_id):
            target_value: Any = target
            figures, fitting_report = generate_figures_and_report(node, target_value)
            sections.append(f"<h3>Target: {target}</h3>")
            sections.append(fitting_report)
            for index, figure in enumerate(figures):
                figure_name = f"{task_id}-{target}-figure-{index}.png"
                _export_png(
                    figure, artifacts_path / figure_name, png_files, export_errors
                )
                sections.append(
                    figure.to_html(
                        full_html=False,
                        include_plotlyjs="cdn" if index == 0 else False,
                    )
                )
            for index, table in enumerate(_report_tables(fitting_report)):
                table_name = f"{task_id}-{target}-table-{index}.png"
                _export_png(
                    table, artifacts_path / table_name, png_files, export_errors
                )
    return {
        "output_folder": str(path.resolve()),
        "report_html": "\n".join(sections),
        "report_artifacts_folder": str(artifacts_path.resolve()),
        "report_png_files": png_files,
        "report_export_errors": export_errors,
    }


def _export_png(
    figure: go.Figure,
    path: Path,
    png_files: list[str],
    export_errors: list[str],
) -> None:
    """Export one Plotly figure, retaining a clear error when Kaleido is unavailable."""
    try:
        figure.write_image(path)
    except (RuntimeError, ValueError) as error:
        export_errors.append(f"{path.name}: {error}")
    else:
        png_files.append(str(path.resolve()))


def _report_tables(report: str) -> list[go.Figure]:
    """Convert fitting-report HTML tables into image-ready Plotly tables."""
    try:
        tables = pd.read_html(StringIO(report))
    except ValueError:
        return []
    return [
        go.Figure(
            data=[
                go.Table(
                    header={"values": list(table.columns)},
                    cells={"values": [table[column] for column in table.columns]},
                )
            ]
        )
        for table in tables
    ]


def _versions_table(versions: dict[str, Any]) -> str:
    """Render report metadata as an HTML table."""
    rows = "".join(
        f"<tr><td>{library}</td><td>{version}</td></tr>"
        for library, version in versions.items()
    )
    return (
        "<h2>Versions</h2><table><thead><tr><th>Library</th>"
        f"<th>Version</th></tr></thead><tbody>{rows}</tbody></table>"
    )
