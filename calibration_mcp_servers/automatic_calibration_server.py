"""Agent-driven stdio MCP server for qibocal automatic calibration."""

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from calibration_mcp_servers._common import run_full_workflow, update_platform

CATALOG = Path(__file__).with_name("PROTOCOL_CATALOG.md")

mcp = FastMCP("qibocal-automatic-calibration")


@mcp.resource("qibocal://protocol-catalog")
def protocol_catalog() -> str:
    """Return the supported protocol catalog for calibration planning."""
    return CATALOG.read_text(encoding="utf-8")


@mcp.prompt()
def plan_automatic_calibration(request: str) -> str:
    """Instruct an AI agent to plan and review a calibration workflow."""
    return f"""You are planning a qibocal calibration for this request:

{request}

Read the `qibocal://protocol-catalog` resource. Select one applicable protocol and
form an experiment dictionary with its required parameters. Ask for missing values
that cannot be inferred safely. Run only the first experiment with
`run_automatic_calibration`, then inspect every PNG in `report_png_files` and the
returned fitting tables. If the model fits the data, call
`update_platform_after_review`. If it does not, either revise the parameters or
choose another protocol and run a new first experiment. When a clear signal permits
a defensible parameter estimate despite a poor fit, explain the estimate before
applying an update. Never update the platform without reviewing the artifacts."""


@mcp.tool()
def run_automatic_calibration(
    natural_language_goal: str,
    experiments: list[dict[str, Any]],
    output_folder: str | None = None,
    targets: list[Any] | None = None,
    platform: str = "mock",
    backend: str = "qibolab",
    force: bool = False,
) -> dict[str, Any]:
    """Run the first strategy experiment, fit it, and export its report artifacts.

    The calling agent creates ``experiments`` from the user goal and protocol catalog.
    Only the first experiment is executed so its fit can guide the next decision.
    This tool never updates the platform.
    """
    if not experiments:
        raise ValueError("Provide one planned experiment from the protocol catalog.")

    outcome = run_full_workflow(
        experiments=[experiments[0]],
        output_folder=output_folder,
        targets=targets,
        platform=platform,
        backend=backend,
        force=force,
    )
    return {
        **outcome,
        "natural_language_goal": natural_language_goal,
        "executed_experiment": experiments[0],
        "next_actions": [
            "Inspect report_png_files and fitting tables before selecting a next step.",
            "For a good fit, call update_platform_after_review.",
            "For a poor fit, revise inputs or run another protocol.",
            "For a clear signal with a poor fit, document a defensible inferred value before updating.",
        ],
    }


@mcp.tool()
def update_platform_after_review(
    data_folder: str,
    skip_qubits: list[str] | None = None,
) -> dict[str, str]:
    """Apply fitted parameters only after the agent has reviewed report artifacts."""
    return update_platform(data_folder, skip_qubits)


if __name__ == "__main__":
    mcp.run()
