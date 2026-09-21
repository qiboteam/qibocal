"""Stdio MCP server for qibocal report generation."""

from pathlib import Path

from fastmcp import FastMCP

from calibration_mcp_servers._common import report_content
from qibocal.cli.report import report

mcp = FastMCP("qibocal-report")


@mcp.tool()
def generate_report(data_folder: str) -> dict[str, str]:
    """Generate index.html and protocol reports for a qibocal output folder."""
    report(Path(data_folder))
    return report_content(data_folder)


if __name__ == "__main__":
    mcp.run()
