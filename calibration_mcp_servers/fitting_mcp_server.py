"""Stdio MCP server for qibocal fitting using the MCP SDK."""

import asyncio
from pathlib import Path
from typing import Any

from mcp.server.lowlevel import Server
from mcp.types import Tool

from calibration_mcp_servers._common import result
from calibration_mcp_servers._mcp import dispatch, run_stdio
from qibocal.cli.fit import fit

server = Server("qibocal-fitting")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="fit_data",
            description="Fit the acquired data in a qibocal output folder.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_folder": {"type": "string"},
                    "output_folder": {"type": ["string", "null"]},
                    "update": {"type": "boolean", "default": False},
                    "force": {"type": "boolean", "default": False},
                },
                "required": ["data_folder"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list:
    def fit_tool(
        data_folder: str,
        output_folder: str | None = None,
        update: bool = False,
        force: bool = False,
    ) -> dict[str, str]:
        fit(
            Path(data_folder),
            update,
            Path(output_folder) if output_folder is not None else None,
            force,
        )
        return result(output_folder or data_folder)

    return dispatch(name, arguments, "fit_data", fit_tool)


if __name__ == "__main__":
    asyncio.run(run_stdio(server, "qibocal-fitting"))
