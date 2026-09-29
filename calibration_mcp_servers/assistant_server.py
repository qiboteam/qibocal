"""Stdio Qibocal assistant MCP server."""

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from calibration_mcp_servers.utils import (
    make_output_path,
    make_runcard,
    return_content,
    run_qq,
    update_qq,
    write_runcard,
)
from qibocal.cli.fit import fit
from qibocal.cli.report import report

mcp = FastMCP("qibocal-assistant")


#######################################################################
# Tools definitions
#######################################################################


@mcp.tool()
async def generate_fit(
    data_folder: str,
    output_folder: str | None = None,
    update: bool = False,
    force: bool = False,
) -> dict[str, str]:
    """Fit the acquired data."""
    fit_folder = Path(output_folder or data_folder)
    fit(
        Path(data_folder),
        update,
        fit_folder,
        force,
    )
    return return_content(fit_folder)


@mcp.tool()
def generate_report(data_folder: str) -> dict[str, str]:
    """Generate index.html and protocol reports for a qibocal output folder."""
    data_folder_path = Path(data_folder)
    report(data_folder_path)
    return return_content(data_folder_path)


@mcp.tool()
async def update_platform(
    data_folder: str,
) -> dict[str, str]:
    """Apply the updated platform from a qibocal output folder."""
    data_folder_path = Path(data_folder)
    await update_qq(data_folder_path)
    return return_content(data_folder_path)


@mcp.tool()
async def acquire_experiments(
    experiments: list[dict[str, Any]],
    parent_folder: str | None = None,
    platform: str = "mock",
    targets: list[Any] | None = None,
    backend: str = "qibolab",
    update: bool = True,
    partition: str | None = None,
) -> dict[str, Any]:
    """Run, fit, and report a list of qibocal experiments.

    Each experiment contains ``operation`` and optionally ``parameters``, ``targets``,
    ``update``, and ``id``. The operation must be registered in qibocal protocols.
    When ``update`` is true (the default), platform updates are applied after
    fitting; set it to false to defer updates until the user approves them.
    ``partition`` selects the Slurm partition the job is submitted to (via
    ``sbatch --wait --time=01:00:00``); when omitted the acquisition runs on the
    local host.
    """

    base = Path(parent_folder) if parent_folder is not None else Path.cwd()
    path = make_output_path(base, platform)
    runcard = make_runcard(experiments, targets, platform, backend, update=update)
    runcard_path = write_runcard(runcard, path)
    try:
        await run_qq(runcard_path, path, update, partition)
    finally:
        runcard_path.unlink(missing_ok=True)
    response: dict[str, Any] = {**return_content(path)}
    response["platform_update_pending"] = True
    response["platform_update_message"] = (
        "Platform updates were not applied. Ask the user for approval, then "
        "call update_platform with the returned output_folder."
    )
    return response


#######################################################################
# Start the MCP server (stdio transport).
#######################################################################


def assistant_start() -> None:
    mcp.run(transport="stdio")
