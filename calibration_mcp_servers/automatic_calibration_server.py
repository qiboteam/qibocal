"""Stateful MCP server for autonomous qibocal calibration."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from qibolab import locate_platform

from calibration_mcp_servers._common import (
    make_output_path,
    make_runcard,
    report_content,
    run_qq,
    write_runcard,
)

CATALOG = Path(__file__).with_name("PROTOCOL_CATALOG.md")

mcp = FastMCP("qibocal-calibrate")


@dataclass
class CalibrationSession:
    """Private, on-disk state of a calibration session.

    Each `run_protocol` call is its own `qq run` subprocess, writing into a fresh
    step folder under `path`; no hardware connection is kept between calls.
    """

    platform_name: str
    targets: list[Any]
    path: Path
    partition: str | None
    step_counter: int = 0
    latest_step_output: Path | None = None

    def __post_init__(self) -> None:
        """Create a session-local platform copy that is isolated from the real registry."""
        self.step_counter = 0
        self.latest_step_output = None
        source = Path(locate_platform(self.platform_name))
        self.session_platform_path = self.path / "new_platform"
        if self.session_platform_path.exists():
            shutil.rmtree(self.session_platform_path)
        shutil.copytree(source, self.session_platform_path)

    def update_session_platform(self, updated_platform: Path) -> None:
        """Replace the session-local platform with an updated version."""
        shutil.rmtree(self.session_platform_path)
        shutil.copytree(updated_platform, self.session_platform_path)


_active_session: CalibrationSession | None = None


def _session() -> CalibrationSession:
    if _active_session is None:
        raise RuntimeError("No calibration is active. Call start_calibration first.")
    return _active_session


def _close_session(*, accept_latest_platform: bool) -> dict[str, Any] | None:
    """Close the active session, leaving the persisted step data in place."""
    global _active_session
    session = _active_session
    if session is None:
        return None

    try:
        if accept_latest_platform and session.latest_step_output is not None:
            last_platform = session.latest_step_output / "new_platform"
            if last_platform.exists():
                shutil.rmtree(session.session_platform_path)
                shutil.copytree(last_platform, session.session_platform_path)
        return {
            "data_folder": str(session.path.resolve()),
            "platform": session.platform_name,
            "step_counter": session.step_counter,
            "session_platform": str(session.session_platform_path.resolve()),
        }
    finally:
        _active_session = None


@mcp.resource("qibocal://protocol-catalog")
def protocol_catalog() -> str:
    """Return the supported protocol catalog for calibration planning."""
    return CATALOG.read_text(encoding="utf-8")


def _analyze_platform_architecture(platform_name: str) -> dict[str, Any]:
    """Inspect a platform's `parameters.json` to describe its QPU architecture.

    Reports the qubit names, the connectivity topology (from the two-qubit
    native gates), which qubits are flux tunable (have a flux channel in
    `configs`), and which couplers are present (flux channels named
    `coupler_*`).
    """
    parameters_path = Path(locate_platform(platform_name)) / "parameters.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))

    configs = parameters.get("configs", {})
    native_gates = parameters.get("native_gates", {})

    qubit_names = sorted(
        (
            name
            for name in native_gates.get("single_qubit", {})
            if any(channel.startswith(f"{name}/") for channel in configs)
        ),
        key=str,
    )
    flux_channels = {
        name: config for name, config in configs.items() if name.endswith("/flux")
    }
    qubit_flux = {
        name: config
        for name, config in flux_channels.items()
        if not name.startswith("coupler")
    }
    coupler_flux = {
        name: config
        for name, config in flux_channels.items()
        if name.startswith("coupler")
    }

    topology: dict[str, list[Any]] = {}
    for pair in native_gates.get("two_qubit", {}):
        a, _, b = str(pair).partition("-")
        topology.setdefault(a, []).append(b)
        topology.setdefault(b, []).append(a)

    return {
        "platform": platform_name,
        "parameters_file": str(parameters_path.resolve()),
        "qubits": qubit_names,
        "nqubits": len(qubit_names),
        "topology": {
            qubit: sorted(neighbors, key=str) for qubit, neighbors in topology.items()
        },
        "flux_tunable_qubits": sorted(qubit_flux, key=str),
        "couplers": sorted(coupler_flux, key=str),
        "has_couplers": bool(coupler_flux),
        "has_flux_tunable_qubits": bool(qubit_flux),
    }


@mcp.tool()
async def platform_architecture() -> dict[str, Any]:
    """Describe the QPU architecture of the active session's platform.

    Call this BEFORE planning the calibration strategy. It parses the
    platform's `parameters.json` and reports: the qubit names and count, the
    connectivity topology (which qubits are connected, from the two-qubit
    native gates), which qubits are flux tunable (they have a flux channel in
    `configs`), and which couplers are present (flux channels named
    `coupler_*`).
    """
    try:
        session = _session()
        return _analyze_platform_architecture(session.platform_name)
    except BaseException:
        _close_session(accept_latest_platform=False)
        raise


@mcp.prompt()
def plan_automatic_calibration(request: str) -> str:
    """Instruct an AI agent to drive an adaptive calibration strategy."""
    return f"""Run an autonomous qibocal calibration for this request:

{request}

This is a fully autonomous calibration run. After the initial user request, do
not ask for approval or additional instructions at every step. Start once with
`start_calibration`, then proceed without awaiting further user input until the
calibration goal is reached and the session is closed with `finish_calibration`.

Read `qibocal://protocol-catalog`, then call `start_calibration` once. Before
thinking about the strategy to adopt, call `platform_architecture` to understand
the QPU: the number of qubits and their names, the connectivity topology
(ancilla/coupler connections), and the kind of architecture of the whole QPU —
whether there are couplers and whether the qubits are flux tunable (this is
derived from the flux channels and coupler entries in the platform's
`parameters.json`). Use that architecture to decide which protocols apply and
which targets each protocol needs. Then build the strategy dynamically and call
`run_protocol` for one protocol at a time. Each call
runs `qq run` in its own step folder under the session's `data_folder`, and the
working platform for this session is stored in a dedicated local session platform
folder instead of the global Qibolab platform. After every call, inspect the PNG
figures in that step's output folder and judge whether the fitted curve matches the
measured signal and physical expectation. If the fit is good, accept the step and
update the session working platform; if the fit is poor or inconsistent, do NOT
accept the update, keep the previous working platform, and change the strategy
before the next step. Pass `targets` to run only the qubits that need that step.
For every protocol, `targets` must be a non-empty subset of the session targets
passed to `start_calibration`; using the entire session target set is also valid.
Omit `targets` to use that entire set. All protocol executions are parallel by
design: one runcard and one Slurm job are used for the selected targets, never a
sequential loop. Set `update=true` only when that protocol's successful fit should
be applied to the platform for subsequent steps; the platform is NOT published to
the Qibolab registry during the step. Continue adapting until the calibration goal
is complete, then explicitly call `accept_step_platform` only for steps whose PNGs
show a trustworthy fit, and finally call `finish_calibration`. When the
calibration is complete, use the existing `update_platform` tool with the session's
`data_folder` to publish the final calibrated platform to the Qibolab registry."""


@mcp.tool()
async def start_calibration(
    parent_folder: str,
    targets: list[Any],
    platform: str,
    partition: str | None = None,
) -> dict[str, Any]:
    """Start one calibration session.

    ``partition`` selects the Slurm partition every subsequent `run_protocol` call
    is submitted to (via ``sbatch --wait --time=01:00:00``); when omitted those
    calls run on the local host.
    """
    global _active_session
    if _active_session is not None:
        raise RuntimeError(
            "A calibration is already active. Finish it before starting another."
        )

    path = make_output_path(parent_folder, platform)
    _active_session = CalibrationSession(
        platform_name=platform,
        targets=targets,
        path=path,
        partition=partition,
    )
    return {
        "data_folder": str(path.resolve()),
        "targets": targets,
        "platform": platform,
        "working_platform": str(_active_session.session_platform_path.resolve()),
    }


@mcp.tool()
async def run_protocol(
    operation: str,
    parameters: dict[str, Any],
    targets: list[Any] | None = None,
) -> dict[str, Any]:
    """Run one protocol in parallel across the selected targets."""
    try:
        session = _session()

        step_id = session.step_counter
        session.step_counter += 1

        step_targets = list(session.targets if targets is None else targets)
        targets_str = "-".join(str(target) for target in step_targets)

        step_path = session.path / f"{step_id}-{operation}-qubits-{targets_str}"
        session.latest_step_output = step_path
        run = await _run_targets(
            session, step_id, operation, parameters, step_targets, step_path
        )

        run.update(
            {
                "operation": operation,
                "step_id": step_id,
                "targets": step_targets,
                "session_platform": str(session.session_platform_path.resolve()),
            }
        )
        return run
    except BaseException:
        _close_session(accept_latest_platform=False)
        raise


@mcp.tool()
async def accept_step_platform() -> dict[str, Any]:
    """Promote a successful step's updated platform to the next calibration iteration.

    This should only be called after visually checking the PNGs in the step output:
    if the fit does not match the measured signal, the previous platform must be
    kept and the calibration strategy revised instead of accepting the update.
    """
    try:
        session = _session()

        step_dir = session.latest_step_output
        if step_dir is None:
            raise RuntimeError("No step has been run yet.")

        next_platform = step_dir / "new_platform"
        if not next_platform.exists():
            raise FileNotFoundError(
                f"No updated platform was produced in {step_dir}; missing {next_platform}."
            )

        shutil.rmtree(session.session_platform_path)
        shutil.copytree(next_platform, session.session_platform_path)

        return {
            "accepted_platform": str(session.session_platform_path.resolve()),
            "source_step": str(step_dir.resolve()),
            "platform": session.platform_name,
        }
    except BaseException:
        _close_session(accept_latest_platform=False)
        raise


async def _run_targets(
    session: CalibrationSession,
    step_id: int,
    operation: str,
    parameters: dict[str, Any],
    targets: list[Any],
    output_path: Path,
) -> dict[str, Any]:
    """Run one runcard and Slurm job for the given targets."""
    runcard = make_runcard(
        [
            {
                "id": str(step_id),
                "operation": operation,
                "parameters": parameters,
                "update": True,
            }
        ],
        targets=targets,
        platform=session.platform_name,
        update=True,
    )
    runcard_path = write_runcard(runcard, output_dir=output_path)
    try:
        await run_qq(
            runcard_path,
            output_path,
            update=True,
            partition=session.partition,
        )
    finally:
        runcard_path.unlink(missing_ok=True)

    response: dict[str, Any] = {**report_content(output_path)}
    return response


@mcp.tool()
async def finish_calibration() -> dict[str, Any]:
    """Close the session and return a summary of its steps."""
    if _active_session is None:
        raise RuntimeError("No calibration is active. Call start_calibration first.")
    return _close_session(accept_latest_platform=True)  # type: ignore[return-value]


if __name__ == "__main__":
    try:
        mcp.run()
    finally:
        _close_session(accept_latest_platform=False)
