"""Stdio MCP server for qibocal platform updates."""

import asyncio
from pathlib import Path

from fastmcp import FastMCP

from calibration_mcp_servers._common import _qq_executable, result

mcp = FastMCP("qibocal-update")


async def _update(path: Path) -> str:
    """Run ``qq update`` as a subprocess and check its exit status.

    Arguments:
        - path: Qibocal output folder.
    """
    command = [_qq_executable(), "update", str(path)]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"'{' '.join(command)}' failed with exit code {process.returncode}:\n"
            f"{stderr.decode()}"
        )
    return stdout.decode()


@mcp.tool()
async def update_platform(
    data_folder: str,
) -> dict[str, str]:
    """Apply the updated platform from a qibocal output folder."""
    await _update(Path(data_folder))
    return result(data_folder)


if __name__ == "__main__":
    mcp.run()
