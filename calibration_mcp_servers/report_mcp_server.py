"""Stdio MCP server for qibocal report generation using the MCP SDK."""

import asyncio
from pathlib import Path
from typing import Any

from mcp.server.lowlevel import Server
from mcp.types import Tool

from calibration_mcp_servers._common import report_content
from calibration_mcp_servers._mcp import dispatch, run_stdio
from qibocal.cli.report import report

server = Server("qibocal-report")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_report",
            description="Generate index.html and protocol reports for a qibocal output folder.",
            inputSchema={
                "type": "object",
                "properties": {"data_folder": {"type": "string"}},
                "required": ["data_folder"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list:
    def report_tool(data_folder: str) -> dict[str, str]:
        report(Path(data_folder))
        return report_content(data_folder)

    return dispatch(name, arguments, "generate_report", report_tool)


if __name__ == "__main__":
    asyncio.run(run_stdio(server, "qibocal-report"))
