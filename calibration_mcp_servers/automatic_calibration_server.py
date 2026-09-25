"""Stateful MCP server for autonomous qibocal calibration."""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from qibolab.platform import Platform, load_hardware, locate_platform

from calibration_mcp_servers._common import (
    make_output_path,
    make_runcard,
    report_content,
    run_qq,
    write_runcard,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.calibration.calibration import CALIBRATION, Calibration
from qibocal.cli.update import merge_with_skipped_qubits

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
    latest_step_targets: list[Any] | None = None
    latest_operation: str | None = None
    manual_override_operation: str | None = None
    manual_override_targets: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Create a session-local platform copy that is isolated from the real registry."""
        self.step_counter = 0
        self.latest_step_output = None
        self.latest_step_targets = None
        self.latest_operation = None
        self.manual_override_operation = None
        self.manual_override_targets = []
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


def _load_calibration_platform(path: Path) -> CalibrationPlatform:
    hardware = load_hardware(path)
    platform = Platform.load(path, **vars(hardware))
    calibration_path = path / CALIBRATION
    calibration = (
        Calibration.model_validate_json(calibration_path.read_text())
        if calibration_path.exists()
        else Calibration()
    )
    return CalibrationPlatform(**vars(platform), calibration=calibration)


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
calibration goal is reached and the session is closed with `finish_calibration`,
or the calibration cannot be completed and the session is closed with
`abort_calibration`.

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
figures in that step's output folder and judge for each qubit whether the fitted
curve matches the measured signal and physical expectation. Build a list of the
qubits with good fits and pass it as `accepted_qubits` to `accept_step_platform`;
updates for the remaining qubits are discarded. If no fit is good, pass an empty
list, keep the previous working platform, and change the strategy before the next
step. Pass `targets` to run only the qubits that need that step.
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
`data_folder` to publish the final calibrated platform to the Qibolab registry.
If repeated refinements cannot produce trustworthy fits, required protocols or
platform information are unavailable, or continuing would produce an unreliable
calibration, call `abort_calibration` instead. Aborting preserves the collected
step data but does not accept the latest step or publish the session platform.
Report the reason for aborting and do not call `finish_calibration` or
`update_platform` afterward.

## Strategy Template

### From-scratch calibration (user explicitly asks to calibrate "from scratch" / "from the beginning")

Follow this ordered sequence. For each phase, run all applicable protocols for
the selected targets before moving to the next phase.

**Phase 1 — Resonator characterization**

**Phase 2 — Qubit characterization**

**Phase 3 — Signal experiments (drive calibration)**
Calibrate the drive pulse using signal experiments

**Phase 4 — Single-shot classification**

**Phase 5 — Classification and Readout optimization**
Optimize assignment fidelity and gate fidelity

Iterate within Phase 5: if fidelity is below target, go back to Phase 3 or 4
to refine the drive/readout calibration, then re-run the classification protocols.

### Incremental calibration (user does NOT specify "from scratch")

1. Call `platform_architecture` to understand the QPU.
2. Inspect the current platform state (read the session platform's
   `parameters.json` and any existing calibration data) to determine which
   quantities are already calibrated and which are missing or stale.
3. Identify the FIRST phase in the template above where information is missing
   or incomplete for the target qubits.
4. Start from that phase and proceed forward through the remaining phases,
   skipping any phase that is already fully calibrated.
5. If a specific quantity is missing (e.g., only `qubit_flux` is missing),
   run only the relevant protocol rather than the entire phase.

In both modes, adapt the strategy dynamically: if a fit is poor, revise
parameters or skip to an alternative protocol before proceeding.

## Parameter Selection: Quick-and-Dirty First

When choosing protocol parameters (frequency ranges, sweep steps, amplitude
ranges, pulse durations, etc.), ALWAYS prefer a quick and dirty run over an
exhaustive one:

- Use a **narrow initial range** centered on the expected value (from the
  platform state or a previous step) rather than a wide blind sweep.
- Use a **coarse step** (fewer data points) for the first pass.
- Keep `nshots` at the platform default or lower for exploratory runs.

After each run, inspect the PNG figures and judge whether the resolution is
fine enough to produce a trustworthy fit:

- If the peak/feature is **clearly resolved** and the fit converges well,
  accept the step and move on.
- If the feature is **barely visible**, the fit is noisy, or the peak position
  is uncertain, **repeat the same protocol** with a slightly finer resolution
  (smaller step, or a narrower range centered on the coarse peak).
- If the feature is **missing entirely**, widen the range and re-run.
- If the **fit is poor** (e.g., the Lorentzian/sine model does not converge,
  or the fitted value is unphysical) but you can **confidently read the
  quantity directly from the raw signal** (e.g., the peak is visually obvious
  even though the automated fit failed), do the following:
  1. Estimate the quantity from the signal (peak position, oscillation
     period, etc.).
  2. Look up the **Platform update fields** table for the protocol you just ran
     in `qibocal://protocol-catalog`. Do not guess a path from scratch: pick the
     row whose field matches the quantity you estimated. Then read the path to
     modify carefully:

     - If the path starts with `parameters`, edit the platform's
       `parameters.json` using the exact path under that file; infer any
       placeholder entries such as `target` or `qubit` from the actual qubit
       being calibrated.
     - If the path starts with `calibration`, edit the platform's
       `calibration.json` using the exact path under that file; again infer any
       placeholder entries such as `target` or `qubit` from the actual qubit.

     When writing the new value, cast it to the same Python type as the original
     value at that location (for example, preserve `float`, `int`, `bool`, or
     `list` structure instead of writing a string or a different numeric type).
     Also keep the literal path structure that the platform expects instead of
     inventing a new one.
  3. Re-run the **same protocol** with its sweep parameters re-centered on
     the new estimate (eventually with different range and step) so the fit has a chance
     to converge around the correct value.

Iterate this refine-and-recheck loop (at most 2–3 refinements per protocol)
until the fit is trustworthy, then accept the step. Never jump straight to a
high-resolution sweep without first confirming the feature exists with a
coarse pass."""


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

        if session.manual_override_operation is not None:
            requested_targets = list(session.targets if targets is None else targets)
            if operation != session.manual_override_operation or set(
                requested_targets
            ) != set(session.manual_override_targets):
                return {
                    "error": (
                        "A manual platform estimate requires the next run to repeat "
                        f"{session.manual_override_operation!r} for exactly "
                        f"{session.manual_override_targets!r} with focused parameters."
                    ),
                    "required_operation": session.manual_override_operation,
                    "required_targets": session.manual_override_targets,
                }

        step_id = session.step_counter
        session.step_counter += 1

        step_targets = list(session.targets if targets is None else targets)
        targets_str = "-".join(str(target) for target in step_targets)

        step_path = session.path / f"{step_id}-{operation}-qubits-{targets_str}"
        session.latest_step_output = step_path
        session.latest_step_targets = step_targets
        session.latest_operation = operation
        run = await _run_targets(
            session, step_id, operation, parameters, step_targets, step_path
        )

        session.manual_override_operation = None
        session.manual_override_targets = []

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
async def accept_step_platform(accepted_qubits: list[Any]) -> dict[str, Any]:
    """Promote successful qubit updates to the next calibration iteration.

    After visually checking each qubit's PNG, pass only the qubits whose fits match
    the measured signal and physical expectation. Updates for all other qubits in
    the latest step are discarded.
    """
    try:
        session = _session()

        step_dir = session.latest_step_output
        if step_dir is None:
            return {"error": "No step has been run yet."}
        step_targets = session.latest_step_targets
        if step_targets is None:
            return {"error": "The latest step has no recorded targets."}

        unknown_qubits = set(accepted_qubits) - set(step_targets)
        if unknown_qubits:
            return {
                "error": f"Accepted qubits were not part of the latest step: {sorted(unknown_qubits, key=str)}"
            }

        step_platform = step_dir / "new_platform"
        if not step_platform.exists():
            return {
                "error": f"No updated platform was produced in {step_dir}; missing {step_platform}."
            }

        current = _load_calibration_platform(session.session_platform_path)
        candidate = _load_calibration_platform(step_platform)
        skipped_qubits = [
            qubit for qubit in step_targets if qubit not in accepted_qubits
        ]
        updated = merge_with_skipped_qubits(current, candidate, skipped_qubits)
        updated.dump(session.session_platform_path)

        return {
            "accepted_platform": str(session.session_platform_path.resolve()),
            "source_step": str(step_dir.resolve()),
            "platform": session.platform_name,
            "accepted_qubits": accepted_qubits,
            "skipped_qubits": skipped_qubits,
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
async def abort_calibration() -> dict[str, Any]:
    """Abort the active session without accepting the latest step platform."""
    if _active_session is None:
        return {"error": "No calibration is active. Call start_calibration first."}
    return {**_close_session(accept_latest_platform=False), "aborted": True}  # type: ignore[arg-type]


@mcp.tool()
async def finish_calibration() -> dict[str, Any]:
    """Close the session and return a summary of its steps."""
    if _active_session is None:
        return {"error": "No calibration is active. Call start_calibration first."}
    return _close_session(accept_latest_platform=True)  # type: ignore[return-value]


if __name__ == "__main__":
    try:
        mcp.run()
    finally:
        _close_session(accept_latest_platform=False)
