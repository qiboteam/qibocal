"""Stdio MCP server for qibocal fitting and report generation."""

from pathlib import Path

from fastmcp import FastMCP

from calibration_mcp_servers._common import report_content
from qibocal.cli.fit import fit
from qibocal.cli.report import report

mcp = FastMCP("qibocal-fitting")


@mcp.tool()
async def fit_and_report(
    data_folder: str,
    output_folder: str | None = None,
    update: bool = False,
    force: bool = False,
) -> dict[str, str]:
    """Fit the acquired data and generate the interactive report."""
    fit(
        Path(data_folder),
        update,
        Path(output_folder) if output_folder is not None else None,
        force,
    )
    fitted_folder = output_folder or data_folder
    report(Path(fitted_folder))
    return report_content(fitted_folder)


if __name__ == "__main__":
    mcp.run()
