"""Stdio MCP server for qibocal platform updates."""

from pathlib import Path

from fastmcp import FastMCP

from calibration_mcp_servers._common import result
from qibocal.cli.update import update

mcp = FastMCP("qibocal-platform-update")


@mcp.tool()
def update_platform(
    data_folder: str,
    skip_qubits: list[str] | None = None,
) -> dict[str, str]:
    """Apply the updated platform from a qibocal output folder."""
    update(Path(data_folder), skip_qubits)
    return result(data_folder)


if __name__ == "__main__":
    mcp.run()
