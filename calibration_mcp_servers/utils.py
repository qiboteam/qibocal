"""Shared adapters for qibocal MCP servers."""

import asyncio
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from qibocal.auto.runcard import Runcard
from qibocal.auto.task import Action


def make_output_path(parent_folder: str | Path, platform: str) -> Path:
    """Create and return a timestamped platform output directory."""
    now = datetime.now().astimezone()
    path = (
        Path(parent_folder)
        / platform
        / now.strftime("%Y%m%d")
        / now.strftime("%H:%M:%S")
    )
    path.mkdir(parents=True)
    return path


def make_runcard(
    experiments: list[dict[str, Any]],
    targets: list[Any] | None = None,
    platform: str = "mock",
    backend: str = "qibolab",
    update: bool = True,
) -> Runcard:
    """Build a qibocal runcard from JSON-compatible experiment definitions."""
    actions = []
    for index, experiment in enumerate(experiments, start=1):
        action = dict(experiment)
        action.setdefault("id", f"experiment-{index}")
        action.setdefault("update", update)
        actions.append(Action.cast(action))
    return Runcard(
        actions=actions,
        targets=targets,
        platform=platform,
        backend=backend,
        update=update,
    )


def write_runcard(runcard: Runcard, output_dir: Path) -> Path:
    """Dump a runcard to a standalone YAML file consumable by `qq run`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runcard_path = output_dir / "runcard.yml"
    with open(runcard_path, "w") as file:
        yaml.safe_dump(asdict(runcard), file)
    return runcard_path


def _qq_executable() -> str:
    """Resolve the `qq` entry point installed alongside the running interpreter.

    Using the `qq` from ``sys.executable``'s directory guarantees the subprocess
    runs with the same Python environment as the MCP server, avoiding a bare
    ``qq`` on ``PATH`` that may point at a different interpreter lacking
    ``qibocal``.
    """
    qq = Path(sys.executable).parent / "qq"
    if not qq.exists():
        raise RuntimeError(
            f"Could not find `qq` next to the running interpreter ({sys.executable})."
        )
    return str(qq)


async def run_qq(
    runcard_path: Path,
    output_path: Path,
    update: bool,
    partition: str | None,
) -> str:
    """Run `qq run`, optionally submitting it via Slurm `sbatch` and checking its exit status."""
    qq_command = [
        _qq_executable(),
        "run",
        str(runcard_path),
        "-o",
        str(output_path),
        "-f",
        "--update" if update else "--no-update",
    ]
    if partition is None:
        command = qq_command
    else:
        command = [
            "srun",
            "--time=01:00:00",
            f"--partition={partition}",
        ] + qq_command

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


async def update_qq(path: Path) -> str:
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


def return_content(output_folder: str | Path) -> dict[str, str]:
    """Return a stable, JSON-compatible MCP response."""
    return {"output_folder": str(Path(output_folder).resolve())}
