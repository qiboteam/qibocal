"""Stdio MCP server for qibocal automatic calibration."""

import uuid
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from calibration_mcp_servers._common import make_runcard, result
from qibocal.cli.run import protocols_execution

mcp = FastMCP("qibocal-automatic-calibration")


@mcp.tool()
def run_automatic_calibration(
    experiments: list[dict[str, Any]],
    output_folder: str | None = None,
    targets: list[Any] | None = None,
    platform: str = "mock",
    backend: str = "qibolab",
    update: bool = True,
    force: bool = False,
) -> dict[str, str]:
    """Acquire, fit, and optionally apply updates for a list of experiments."""
    if output_folder is None:
        output_folder = f"run-{uuid.uuid4().hex[:8]}"
    runcard = make_runcard(experiments, targets, platform, backend, update)
    protocols_execution(runcard, Path(output_folder), force, update)
    return result(output_folder)


if __name__ == "__main__":
    mcp.run()
