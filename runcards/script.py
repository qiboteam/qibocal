from pathlib import Path
from tempfile import TemporaryDirectory

from qibocal.auto.execute import Executor
from qibocal.protocols import t1_signal

t1_params = {
    "id": "t1_experiment",
    "targets": [0],  # we are defining here which qubits to analyze
    "operation": "t1_signal",
    "parameters": {
        "delay_before_readout_start": 0,
        "delay_before_readout_end": 20_000,
        "delay_before_readout_step": 50,
    },
}

with TemporaryDirectory() as tmp:
    exec = Executor.create(output=Path(tmp), platform="dummy")
    out = exec.run_protocol(t1_signal, parameters=t1_params)
