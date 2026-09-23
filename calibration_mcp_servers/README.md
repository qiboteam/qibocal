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

The acquisition tools build a qibocal runcard from the requested experiments,
write the temporary YAML file into the output folder so it is visible from Slurm
compute nodes, and run it with the `qq run` CLI as a subprocess. The temporary
runcard is deleted after execution, including when `qq run` fails. One call
produces acquisition, fitting, and report generation. The tools generate the
interactive `index.html` report and PNG files for every figure, and return the
`report_folder` containing the PNG artifacts. They do not update the platform
automatically. `parent_folder` is optional and identifies the parent directory
for a run. Each invocation creates the actual qibocal run root at
`parent_folder/<platform>/<date>/<datetime>/`; when `parent_folder` is omitted, the run
root is created at `parent_folder/<platform>/<date>/<datetime>/`. That
generated directory contains qibocal's `data/` subfolder, history, metadata,
report, and the `agent_report/` folder with PNG figures.

An optional `partition` argument submits the `qq run` subprocess to a Slurm
partition instead of running it on the local host.

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

- `output_folder` — the generated qibocal run directory (`parent_folder/<platform>/<date>/<datetime>/`)
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

- `parent_folder` — where the session's run directory is created
- `targets` — the qubits to calibrate
- `platform` — the platform name
- `partition` (optional) — the Slurm partition every subsequent `run_protocol`
  call is submitted to; when omitted those calls run on the local host

Unlike the acquisition server, no hardware connection is kept open between tool
calls. The session has one datetime-based directory at
`parent_folder/<platform>/<date>/<datetime>/`. Each `run_protocol` call accepts an optional
`targets` list. It must be a non-empty subset of the targets passed to
`start_calibration`, including the full set; omitting it uses the full session
target set. This lets the agent retry one poorly calibrated qubit in one step and
return to all session qubits in a later step.

`run_protocol` supports two execution modes:

- `execution_mode="parallel"` (default) builds one single-action runcard and runs
	one `qq run` job for all selected targets. Its step folder is named
	`<step_id>-<operation>-qubits-<targets>`.
- `execution_mode="individual"` builds a separate runcard and runs a separate
	`qq run` job for each selected target. Each run has its own folder named
	`<step_id>-<operation>-qubit-<target>`. The response contains `runs` and
	`output_folders` entries for the per-qubit results.

Every run writes its temporary runcard into its output folder and deletes it when
execution finishes. Each run also generates an `agent_report/` directory with PNG
figures. When `update=true`, each successful run's fit result is immediately
published into the configured Qibolab platform registry (the same mechanism as
`update_platform_after_approval`), so later runs — which reconnect to the platform
from scratch — see it right away.

The agent must inspect each round's figures and fitting results before selecting
the next protocol, changing its parameters, choosing a target subset, or selecting
an execution mode. When the strategy is complete, it calls `finish_calibration`,
which closes the session and returns a summary of every logical step that ran,
including its `step_id`, `operation`, `targets`, `execution_mode`, and output
folders.

A minimal natural-language request can be passed to the prompt like this:

```text
Calibrate the pi pulse amplitude of qubit 0 on the qw5q_platinum platform,
saving results to /home/users/lorenzo.ballerio/test/qibocal_experiments.
Adapt the strategy after reviewing every fit and generated plot, and only set
update=true for protocols whose result should be published to the platform.
```

For example, an agent can run one protocol for all session qubits in parallel,
retry only qubit 1 in an individual job, and then run the next protocol for all
session qubits again. Each `run_protocol` call chooses its own `targets` and
`execution_mode` without changing the session target set.
