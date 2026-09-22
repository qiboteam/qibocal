"""Stdio MCP server for qibocal data acquisition."""

import uuid
from pathlib import Path
from typing import Any, cast

from fastmcp import FastMCP

from calibration_mcp_servers._common import report_content, result
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Action
from qibocal.cli.run import protocols_execution
from qibocal.cli.update import update

mcp = FastMCP("qibocal-acquisition")


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


@mcp.tool()
async def acquire_experiments(
    experiments: list[dict[str, Any]],
    parent_folder: str | None = None,
    platform: str = "mock",
    targets: list[Any] | None = None,
    backend: str = "qibolab",
    force: bool = False,
    update: bool = True,
) -> dict[str, Any]:
    """Run, fit, and report a list of qibocal experiments.

    Each experiment contains ``operation`` and optionally ``parameters``, ``targets``,
    ``update``, and ``id``. The operation must be registered in qibocal protocols.
    When ``update`` is true (the default), platform updates are applied after
    fitting; set it to false to defer updates until the user approves them.
    """

    base = Path(parent_folder) if parent_folder is not None else Path.cwd()
    path = base / platform / str(uuid.uuid4())
    runcard = make_runcard(experiments, targets, platform, backend, update=update)
    print(runcard)
    protocols_execution(runcard, path, force, update=update)
    response: dict[str, Any] = {**report_content(path)}
    response["platform_update_pending"] = True
    response["platform_update_message"] = (
        "Platform updates were not applied. Ask the user for approval, then "
        "call update_platform with the returned output_folder."
    )
    return response


def update_platform(
    output_folder: str | Path,
    skip_qubits: list[str] | None = None,
) -> dict[str, str]:
    """Apply platform updates after explicit user approval."""
    update(Path(output_folder), cast(Any, skip_qubits))
    return result(output_folder)


@mcp.tool()
def update_platform_after_approval(
    data_folder: str,
    skip_qubits: list[str] | None = None,
) -> dict[str, str]:
    """Apply platform updates after explicit user approval."""
    return update_platform(data_folder, skip_qubits)


if __name__ == "__main__":
    mcp.run()
