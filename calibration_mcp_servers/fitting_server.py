"""Stdio MCP server for qibocal fitting."""

from pathlib import Path

from fastmcp import FastMCP

from calibration_mcp_servers._common import result
from qibocal.cli.fit import fit

mcp = FastMCP("qibocal-fitting")


@mcp.tool()
def fit_data(
    data_folder: str,
    output_folder: str | None = None,
    update: bool = False,
    force: bool = False,
) -> dict[str, str]:
    """Fit the acquired data in a qibocal output folder."""
    fit(
        Path(data_folder),
        update,
        Path(output_folder) if output_folder is not None else None,
        force,
    )
    return result(output_folder or data_folder)


if __name__ == "__main__":
    mcp.run()
