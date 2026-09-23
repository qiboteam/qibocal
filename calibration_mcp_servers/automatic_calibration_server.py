"""Stateful MCP server for autonomous qibocal calibration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP

from calibration_mcp_servers._common import (
    make_output_path,
    make_runcard,
    report_content,
    run_qq,
    write_runcard,
)
from qibocal.auto.output import PLATFORM, UPDATED_PLATFORM
from qibocal.cli.update import update as publish_platform

CATALOG = Path(__file__).with_name("PROTOCOL_CATALOG.md")

mcp = FastMCP("qibocal-automatic-calibration")


@dataclass
class CalibrationSession:
    """Private, on-disk state of a calibration session.

    Each `run_protocol` call is its own `qq run` subprocess, writing into a fresh
    step folder under `path`; no hardware connection is kept between calls.
    """

    platform_name: str
    targets: list[Any]
    path: Path
    partition: str | None
    steps: list[dict[str, Any]] = field(default_factory=list)


_active_session: CalibrationSession | None = None


def _session() -> CalibrationSession:
    if _active_session is None:
        raise RuntimeError("No calibration is active. Call start_calibration first.")
    return _active_session


@mcp.resource("qibocal://protocol-catalog")
def protocol_catalog() -> str:
    """Return the supported protocol catalog for calibration planning."""
    return CATALOG.read_text(encoding="utf-8")


@mcp.prompt()
def plan_automatic_calibration(request: str) -> str:
    """Instruct an AI agent to drive an adaptive calibration strategy."""
    return f"""Run an autonomous qibocal calibration for this request:

{request}

Read `qibocal://protocol-catalog`, then call `start_calibration` once. Build the
strategy dynamically and call `run_protocol` for one protocol at a time. Each call
runs `qq run` in its own step folder under the session's `data_folder`. After every
call, inspect its PNG figures and fitting results before choosing the next protocol,
its parameters, and its target qubits. Pass `targets` to run only the qubits that
need that step. For every protocol, `targets` must be a non-empty subset of the
session targets passed to `start_calibration`; using the entire session target set
is also valid. Omit `targets` to use that entire set. Use the default
`execution_mode="parallel"` to run all selected targets in one runcard and Slurm
job. Use `execution_mode="individual"` to run each selected target in its own
runcard, Slurm job, and output folder. Set `update=true` only when that protocol's
successful fit should update the platform: doing so immediately publishes the
step's calibration to the Qibolab platform registry, so later steps (and other
users of that platform) see it right away. Continue adapting until the calibration
goal is complete, then call `finish_calibration`."""


@mcp.tool()
async def start_calibration(
    parent_folder: str,
    targets: list[Any],
    platform: str,
    partition: str | None = None,
) -> dict[str, Any]:
    """Start one calibration session.

    ``partition`` selects the Slurm partition every subsequent `run_protocol` call
    is submitted to (via ``sbatch --wait --time=01:00:00``); when omitted those
    calls run on the local host.
    """
    global _active_session
    if _active_session is not None:
        raise RuntimeError(
            "A calibration is already active. Finish it before starting another."
        )

    path = make_output_path(parent_folder, platform)
    _active_session = CalibrationSession(
        platform_name=platform,
        targets=targets,
        path=path,
        partition=partition,
    )
    return {
        "data_folder": str(path.resolve()),
        "targets": targets,
        "platform": platform,
    }


@mcp.tool()
async def run_protocol(
    operation: str,
    parameters: dict[str, Any],
    update: bool = True,
    step_id: str | None = None,
    targets: list[Any] | None = None,
    execution_mode: Literal["parallel", "individual"] = "parallel",
) -> dict[str, Any]:
    """Run one protocol jointly or in a separate job for each selected target."""
    session = _session()
    step_id = step_id or operation

    step_targets = list(session.targets if targets is None else targets)
    targets_str = "-".join(str(target) for target in step_targets)

    if execution_mode == "parallel":
        step_path = session.path / f"{step_id}-{operation}-qubits-{targets_str}"
        runs = [
            await _run_targets(
                session, step_id, operation, parameters, step_targets, step_path, update
            )
        ]
        response: dict[str, Any] = {**runs[0]}
    else:
        runs = []
        for target in step_targets:
            target_path = session.path / f"{step_id}-{operation}-qubit-{target}"
            runs.append(
                await _run_targets(
                    session,
                    step_id,
                    operation,
                    parameters,
                    [target],
                    target_path,
                    update,
                )
            )
        response = {
            "output_folders": [run["output_folder"] for run in runs],
            "runs": runs,
        }

    session.steps.append(
        {
            "step_id": step_id,
            "operation": operation,
            "targets": step_targets,
            "execution_mode": execution_mode,
            "folders": [run["output_folder"] for run in runs],
        }
    )
    response.update(
        {
            "operation": operation,
            "step_id": step_id,
            "targets": step_targets,
            "execution_mode": execution_mode,
            "update_requested": update,
            "platform_published": update,
        }
    )
    return response


async def _run_targets(
    session: CalibrationSession,
    step_id: str,
    operation: str,
    parameters: dict[str, Any],
    targets: list[Any],
    output_path: Path,
    update: bool,
) -> dict[str, Any]:
    """Run one runcard and Slurm job for the given targets."""
    runcard = make_runcard(
        [
            {
                "id": step_id,
                "operation": operation,
                "parameters": parameters,
                "update": update,
            }
        ],
        targets=targets,
        platform=session.platform_name,
        update=update,
    )
    runcard_path = write_runcard(runcard, output_dir=output_path)
    try:
        await run_qq(
            runcard_path, output_path, update=update, partition=session.partition
        )
    finally:
        runcard_path.unlink(missing_ok=True)

    if update:
        publish_platform(output_path, skip_qubits=None)

    response: dict[str, Any] = {**report_content(output_path)}
    response.update(
        {
            "targets": targets,
            "platform_folder": str((output_path / PLATFORM).resolve()),
            "updated_platform_folder": str((output_path / UPDATED_PLATFORM).resolve()),
            "report_folder": str((output_path / "agent_report").resolve()),
        }
    )
    return response


@mcp.tool()
async def finish_calibration() -> dict[str, Any]:
    """Close the session and return a summary of its steps."""
    global _active_session
    session = _session()
    _active_session = None

    return {
        "data_folder": str(session.path.resolve()),
        "platform": session.platform_name,
        "steps": session.steps,
    }


if __name__ == "__main__":
    mcp.run()
