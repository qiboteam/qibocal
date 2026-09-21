"""Stdio MCP server for qibocal automatic calibration using the MCP SDK."""

import asyncio
import uuid
from pathlib import Path
from typing import Any

from mcp.server.lowlevel import Server
from mcp.types import Tool

from calibration_mcp_servers._common import make_runcard, result
from calibration_mcp_servers._mcp import dispatch, run_stdio
from qibocal.cli.run import protocols_execution

server = Server("qibocal-automatic-calibration")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="run_automatic_calibration",
            description="Acquire, fit, report, and optionally apply updates for experiments.",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiments": {"type": "array", "items": {"type": "object"}},
                    "output_folder": {"type": ["string", "null"]},
                    "targets": {"type": ["array", "null"]},
                    "platform": {"type": "string", "default": "mock"},
                    "backend": {"type": "string", "default": "qibolab"},
                    "update": {"type": "boolean", "default": True},
                    "force": {"type": "boolean", "default": False},
                },
                "required": ["experiments"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list:
    def automatic_calibration_tool(
        experiments: list[dict[str, Any]],
        output_folder: str | None = None,
        targets: list[Any] | None = None,
        platform: str = "mock",
        backend: str = "qibolab",
        update: bool = True,
        force: bool = False,
    ) -> dict[str, str]:
        if output_folder is None:
            output_folder = f"run-{uuid.uuid4().hex[:8]}"
        runcard = make_runcard(experiments, targets, platform, backend, update)
        protocols_execution(runcard, Path(output_folder), force, update)
        return result(output_folder)

    return dispatch(
        name,
        arguments,
        "run_automatic_calibration",
        automatic_calibration_tool,
    )


if __name__ == "__main__":
    asyncio.run(run_stdio(server, "qibocal-automatic-calibration"))
