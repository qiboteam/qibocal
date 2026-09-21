# Qibocal MCP servers

Each module starts an MCP server over stdio when launched with Python:

```text
python -m calibration_mcp_servers.acquisition_server
python -m calibration_mcp_servers.fitting_server
python -m calibration_mcp_servers.report_server
python -m calibration_mcp_servers.automatic_calibration_server
python -m calibration_mcp_servers.update_server
```

The `_mcp_server.py` variants use the official low-level `mcp` SDK directly:

```text
python -m calibration_mcp_servers.acquisition_mcp_server
python -m calibration_mcp_servers.fitting_mcp_server
python -m calibration_mcp_servers.report_mcp_server
python -m calibration_mcp_servers.automatic_calibration_mcp_server
python -m calibration_mcp_servers.update_mcp_server
```

The acquisition tools run acquisition, fitting, and report generation in one call.
They return the report plots and tables in `report_html` and do not update the
platform automatically. After the user approves the changes, call
`update_platform_after_approval` with the returned output folder.

[`PROTOCOL_CATALOG.md`](PROTOCOL_CATALOG.md) lists every built-in qibocal
operation, its parameter descriptions, types, and required fields. Regenerate it
after protocol changes with:

```text
python calibration_mcp_servers/generate_protocol_catalog.py
```

The acquisition and automatic-calibration tools accept experiments in this form:

```json
[
  {
    "operation": "rabi_amplitude",
    "parameters": {"min_amp": 0.0, "max_amp": 1.0, "step_amp": 0.01, "rx90": false},
    "targets": [0]
  }
]
```

The operation name must be registered in `qibocal.protocols`. Paths are passed as
strings to MCP and resolved by qibocal on the server side.

## Example: running a Rabi amplitude experiment

Call the `acquire_experiments` tool on the acquisition server:

```json
{
  "experiments": [
    {
      "operation": "rabi_amplitude",
      "parameters": {
        "min_amp": 0.0,
        "max_amp": 1.0,
        "step_amp": 0.01,
        "rx90": true
      },
      "targets": [0]
    }
  ],
  "output_folder": "/tmp/qibocal_rabi",
  "platform": "mock",
  "force": true
}
```

The tool runs acquisition, fitting, and report generation, then returns:

- `output_folder` — the resolved output directory
- `report_html` — all plots and tables rendered as HTML
- `platform_update_pending` — `true`, indicating the platform has not been updated yet

After reviewing the report, if the user approves the calibration, call
`update_platform_after_approval`:

```json
{
  "data_folder": "/tmp/qibocal_rabi"
}
```

## Agent-guided automatic calibration

The automatic calibration server provides the `qibocal://protocol-catalog`
resource and the `plan_automatic_calibration` prompt. Give the prompt the user's
natural-language calibration request. The agent uses the catalog to select an
initial protocol and its parameters, runs it with `run_automatic_calibration`,
and inspects the returned `report_png_files` and fitting tables.

The first run does not update the platform. After reviewing the fit, the agent
either revises the parameters, tries another protocol, or documents a defensible
inferred value from a clear signal. It calls `update_platform_after_review` only
when the selected calibration result is approved.
