"""Helpers for servers built with the official MCP Python SDK."""

import json
from collections.abc import Callable
from typing import Any

from mcp.server.lowlevel import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import NotificationOptions, TextContent


async def run_stdio(server: Server, name: str) -> None:
    """Run a low-level MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=name,
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def json_content(value: Any) -> list[TextContent]:
    """Encode a tool result as MCP text content."""
    return [TextContent(type="text", text=json.dumps(value))]


def dispatch(
    name: str,
    arguments: dict[str, Any] | None,
    expected: str,
    handler: Callable[..., Any],
) -> list[TextContent]:
    """Dispatch one named tool and reject unknown tools."""
    if name != expected:
        raise ValueError(f"Unknown tool: {name}")
    return json_content(handler(**(arguments or {})))
