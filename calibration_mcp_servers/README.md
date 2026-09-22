# Qibocal MCP servers

Each module starts an MCP server over stdio when launched with Python:

```text
python -m calibration_mcp_servers.acquisition_server
python -m calibration_mcp_servers.fitting_server
python -m calibration_mcp_servers.report_server
python -m calibration_mcp_servers.automatic_calibration_server
python -m calibration_mcp_servers.update_server
```

## MCP server configuration

To use the servers from an MCP client (for example VS Code's `mcp.json`),
register one stdio server per module. Replace `/path/to/venv/bin/python` with
your Python interpreter and `/path/to/qibocal` with the path to this repository:

```json
{
	"servers": {
		"/qibocal-fit": {
			"type": "stdio",
			"command": "/path/to/venv/bin/python",
			"args": ["-m", "calibration_mcp_servers.fitting_server"],
			"cwd": "/path/to/qibocal"
		},
		"/qibocal-report": {
			"type": "stdio",
			"command": "/path/to/venv/bin/python",
			"args": ["-m", "calibration_mcp_servers.report_server"],
			"cwd": "/path/to/qibocal"
		},
		"/qibocal-update": {
			"type": "stdio",
			"command": "/path/to/venv/bin/python",
			"args": ["-m", "calibration_mcp_servers.update_server"],
			"cwd": "/path/to/qibocal"
		},
		"/qibocal-execute": {
			"type": "stdio",
			"command": "/path/to/venv/bin/python",
			"args": ["-m", "calibration_mcp_servers.acquisition_server"],
			"cwd": "/path/to/qibocal"
		},
		"/qibocal-calibrate": {
			"type": "stdio",
			"command": "/path/to/venv/bin/python",
			"args": ["-m", "calibration_mcp_servers.automatic_calibration_server"],
			"cwd": "/path/to/qibocal"
		}
	},
	"inputs": []
}
```

The acquisition tools run acquisition, fitting, and report generation in one call.
They generate the interactive `index.html` report and PNG files for every figure,
and return the `report_folder` containing the PNG artifacts. They do not update
the platform automatically. `parent_folder` is optional and identifies the parent
directory for a run. Each invocation creates the actual qibocal run root at
`parent_folder/<platform>/<uuid>/`; when `parent_folder` is omitted, the run
root is created at `<current working directory>/<platform>/<uuid>/`. That
generated directory contains qibocal's `data/` subfolder, runcard, history,
metadata, report, and the `agent_report/` folder with PNG figures.

[`PROTOCOL_CATALOG.md`](PROTOCOL_CATALOG.md) lists every built-in qibocal
operation, its parameter descriptions, types, and required fields. Regenerate it
after protocol changes with:

```text
python calibration_mcp_servers/generate_protocol_catalog.py
```

The operation name must be registered in `qibocal.protocols`. Paths are passed as
strings to MCP and resolved by qibocal on the server side.

## Example: running a Rabi amplitude experiment

Ask the agent in natural language:

```text
Run a Rabi amplitude experiment on qubit 0 with amplitude range (0.0, 1.0, 0.01)
on the mock platform, saving results to /tmp/qibocal_runs.
```

The agent calls the `acquire_experiments` tool on the acquisition server with
the corresponding parameters. The tool runs acquisition, fitting, and report
generation, then returns:

- `output_folder` — the generated qibocal run directory (`parent_folder/<platform>/<uuid>/`)
- `report_folder` — directory containing PNG files for all report figures
- `platform_update_pending` — `true`, indicating the platform has not been updated yet

After reviewing the report, if the user approves the calibration, ask:

```text
Apply the calibration update from /tmp/qibocal_runs.
```

The agent then calls `update_platform_after_approval` with the run folder.

## Agent-guided automatic calibration

The automatic calibration server provides the `qibocal://protocol-catalog`
resource and the `plan_automatic_calibration` prompt. The agent is the strategy
engine: it reads the catalog, dynamically plans a sequence of protocols, and
adapts the strategy after every step based on the observed results.

Start the process once with `start_calibration`, providing:

- `parent_folder` — where the run directory is created
- `targets` — the qubits to calibrate
- `platform` — the platform name

The server creates one `Executor` and keeps it connected for the entire process.
Each reasoning round calls `run_protocol` with one catalog operation, its
parameters, and an optional `update` flag. When `update=true`, successful fit
results update the executor's private in-memory platform. After every round the
current history, metadata, report, PNG figures, and platform state are persisted
under the same generated data folder. Following Qibocal's output convention,
`platform/` preserves the initial snapshot and `new_platform/` contains the
latest private platform state.

The agent must inspect each round's figures and fitting results before selecting
the next protocol or changing its parameters. When the strategy is complete, it
calls `finish_calibration`. Passing `publish=true` copies the final private
`new_platform/` into the configured Qibolab platform registry through Qibocal's
standard `update()` command exactly once. Passing `publish=false` closes the
executor while leaving the registry unchanged.

A minimal natural-language request can be passed to the prompt like this:

```text
Calibrate the pi pulse amplitude of qubit 0 on the qw5q_platinum platform,
saving results to /home/users/lorenzo.ballerio/test/qibocal_experiments.
Adapt the strategy after reviewing every fit and generated plot, then publish
the final platform only when the complete calibration is finished.
```
