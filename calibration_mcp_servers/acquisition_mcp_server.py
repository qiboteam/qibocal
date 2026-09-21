"""Stdio MCP server for qibocal data acquisition using the MCP SDK."""

import asyncio
from typing import Any

from mcp.server.lowlevel import Server
from mcp.types import Tool

from calibration_mcp_servers._common import run_full_workflow, update_platform
from calibration_mcp_servers._mcp import dispatch, run_stdio

server = Server("qibocal-acquisition")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="acquire_experiments",
            description="Run, fit, and report a list of qibocal experiments. Platform updates are deferred until approved.",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiments": {"type": "array", "items": {"type": "object"}},
                    "output_folder": {"type": ["string", "null"]},
                    "targets": {"type": ["array", "null"]},
                    "platform": {"type": "string", "default": "mock"},
                    "backend": {"type": "string", "default": "qibolab"},
                    "force": {"type": "boolean", "default": False},
                },
                "required": ["experiments"],
            },
        ),
        Tool(
            name="update_platform_after_approval",
            description="Apply platform updates after explicit user approval.",
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
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list:
    def workflow_tool(
        experiments: list[dict[str, Any]],
        output_folder: str | None = None,
        targets: list[Any] | None = None,
        platform: str = "mock",
        backend: str = "qibolab",
        force: bool = False,
    ) -> dict[str, Any]:
        return run_full_workflow(
            experiments, output_folder, targets, platform, backend, force
        )

    def update_tool(
        data_folder: str,
        skip_qubits: list[str] | None = None,
    ) -> dict[str, str]:
        return update_platform(data_folder, skip_qubits)

    if name == "acquire_experiments":
        return dispatch(name, arguments, name, workflow_tool)
    return dispatch(name, arguments, "update_platform_after_approval", update_tool)


if __name__ == "__main__":
    asyncio.run(run_stdio(server, "qibocal-acquisition"))
