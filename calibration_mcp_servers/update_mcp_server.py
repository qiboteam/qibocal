"""Stdio MCP server for qibocal platform updates using the MCP SDK."""

import asyncio
from pathlib import Path
from typing import Any

from mcp.server.lowlevel import Server
from mcp.types import Tool

from calibration_mcp_servers._common import result
from calibration_mcp_servers._mcp import dispatch, run_stdio
from qibocal.cli.update import update

server = Server("qibocal-platform-update")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="update_platform",
            description="Apply the updated platform from a qibocal output folder.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_folder": {"type": "string"},
                    "skip_qubits": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                },
                "required": ["data_folder"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list:
    def update_tool(
        data_folder: str,
        skip_qubits: list[str] | None = None,
    ) -> dict[str, str]:
        update(Path(data_folder), skip_qubits)
        return result(data_folder)

    return dispatch(name, arguments, "update_platform", update_tool)


if __name__ == "__main__":
    asyncio.run(run_stdio(server, "qibocal-platform-update"))
