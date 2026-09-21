"""Stdio MCP server for qibocal data acquisition."""

from typing import Any

from fastmcp import FastMCP

from calibration_mcp_servers._common import run_full_workflow, update_platform

mcp = FastMCP("qibocal-acquisition")


@mcp.tool()
def acquire_experiments(
    experiments: list[dict[str, Any]],
    output_folder: str | None = None,
    platform: str = "mock",
    targets: list[Any] | None = None,
    backend: str = "qibolab",
    force: bool = False,
) -> dict[str, Any]:
    """Run, fit, and report a list of qibocal experiments.

    Each experiment contains ``operation`` and optionally ``parameters``, ``targets``,
    ``update``, and ``id``. The operation must be registered in qibocal protocols.
    Platform updates are deferred until the user approves them.
    """
    return run_full_workflow(
        experiments, output_folder, targets, platform, backend, force
    )


@mcp.tool()
def update_platform_after_approval(
    data_folder: str,
    skip_qubits: list[str] | None = None,
) -> dict[str, str]:
    """Apply platform updates after explicit user approval."""
    return update_platform(data_folder, skip_qubits)


if __name__ == "__main__":
    mcp.run()
