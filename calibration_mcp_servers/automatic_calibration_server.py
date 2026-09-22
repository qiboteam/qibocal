"""Stateful MCP server for autonomous qibocal calibration."""

import atexit
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from calibration_mcp_servers._common import report_content
from qibocal import Executor
from qibocal.auto.mode import AUTOCALIBRATION
from qibocal.auto.output import PLATFORM, UPDATED_PLATFORM, Output
from qibocal.auto.task import Action
from qibocal.cli.report import report
from qibocal.cli.update import update as publish_platform
from qibocal.config import log

CATALOG = Path(__file__).with_name("PROTOCOL_CATALOG.md")

mcp = FastMCP("qibocal-automatic-calibration")


@dataclass
class CalibrationSession:
    """Live executor and its private output state."""

    executor: Executor
    platform_name: str
    path: Path


_active_session: CalibrationSession | None = None


def _session() -> CalibrationSession:
    if _active_session is None:
        raise RuntimeError("No calibration is active. Call start_calibration first.")
    return _active_session


def _checkpoint(session: CalibrationSession, generate_report: bool = True) -> None:
    """Persist the live history and platform without closing the executor."""
    Output(session.executor.history, session.executor.meta).dump(session.path)
    Output.update_platform(session.executor.platform, session.path)
    if generate_report and session.executor.history._order:
        report(session.path, history=session.executor.history)
        report_content(session.path)


def _close_active_session() -> None:
    """Release hardware if the MCP process exits with an active calibration."""
    global _active_session
    if _active_session is None:
        return
    session = _active_session
    _active_session = None
    try:
        session.executor.platform.disconnect()
        session.executor.meta.end()
        _checkpoint(session, generate_report=False)
    except Exception:
        log.exception("Failed to close the active automatic calibration session.")


atexit.register(_close_active_session)


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
strategy dynamically and call `run_protocol` for one protocol at a time. After
every call, inspect its PNG figures and fitting results before choosing the next
protocol or changing its parameters. Set `update=true` only when that protocol's
successful fit should update the session's private platform. Continue adapting
until the calibration goal is complete, then call `finish_calibration`. The live
executor is reused throughout the process, `platform/` remains the initial
snapshot, and `new_platform/` is refreshed after every step. The Qibolab platform
registry is changed only by `finish_calibration(publish=true)`."""


@mcp.tool()
async def start_calibration(
    parent_folder: str,
    targets: list[Any],
    platform: str,
    force: bool = False,
) -> dict[str, Any]:
    """Start one calibration session and keep its executor connected."""
    global _active_session
    if _active_session is not None:
        raise RuntimeError(
            "A calibration is already active. Finish it before starting another."
        )

    path = Path(parent_folder) / platform / str(uuid.uuid4())
    executor = Executor.create(
        path=path,
        targets=targets,
        platform=platform,
        update=True,
    )
    executor.init(force=force)
    _active_session = CalibrationSession(
        executor=executor,
        platform_name=platform,
        path=path,
    )
    return {
        "data_folder": str(path.resolve()),
        "platform_folder": str((path / PLATFORM).resolve()),
        "updated_platform_folder": str((path / UPDATED_PLATFORM).resolve()),
        "targets": targets,
        "platform": platform,
    }


@mcp.tool()
async def run_protocol(
    operation: str,
    parameters: dict[str, Any],
    update: bool = True,
    step_id: str | None = None,
) -> dict[str, Any]:
    """Run, fit, optionally update, and checkpoint one calibration protocol."""
    session = _session()
    protocol = session.executor.protocols.get(operation)
    if protocol is None:
        raise ValueError(f"Unknown qibocal protocol: {operation}")

    action = Action.cast(
        {
            "id": step_id or operation,
            "operation": operation,
            "parameters": parameters,
            "update": update,
        }
    )
    completed = session.executor.run_protocol(
        protocol=protocol,
        parameters=action,
        mode=AUTOCALIBRATION,
        output=session.path,
    )
    _checkpoint(session)

    targets = completed.task.targets or session.executor.targets
    results = completed.results
    successful_targets = [
        str(target)
        for target in targets
        if results is not None
        and (tuple(target) if isinstance(target, list) else target) in results
    ]
    return {
        "data_folder": str(session.path.resolve()),
        "platform_folder": str((session.path / PLATFORM).resolve()),
        "updated_platform_folder": str((session.path / UPDATED_PLATFORM).resolve()),
        "report_folder": str((session.path / "agent_report").resolve()),
        "operation": operation,
        "step_id": str(completed.task.id),
        "successful_targets": successful_targets,
        "platform_updated_targets": successful_targets if update else [],
        "update_requested": update,
    }


@mcp.tool()
async def finish_calibration(publish: bool = True) -> dict[str, Any]:
    """Close the executor and optionally publish its final platform once."""
    global _active_session
    session = _session()
    session.executor.platform.disconnect()
    session.executor.meta.end()
    _checkpoint(session)
    _active_session = None

    if publish:
        publish_platform(session.path, skip_qubits=None)

    return {
        "data_folder": str(session.path.resolve()),
        "platform_folder": str((session.path / PLATFORM).resolve()),
        "updated_platform_folder": str((session.path / UPDATED_PLATFORM).resolve()),
        "published": publish,
        "platform": session.platform_name,
    }


if __name__ == "__main__":
    mcp.run()
