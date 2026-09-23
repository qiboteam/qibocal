"""Stdio MCP server for qibocal data acquisition."""

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from calibration_mcp_servers._common import (
    make_output_path,
    make_runcard,
    report_content,
    run_qq,
    write_runcard,
)

mcp = FastMCP("qibocal-execute")


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
    response: dict[str, Any] = {**report_content(path)}
    response["platform_update_pending"] = True
    response["platform_update_message"] = (
        "Platform updates were not applied. Ask the user for approval, then "
        "call update_platform with the returned output_folder."
    )
    return response


if __name__ == "__main__":
    mcp.run()
