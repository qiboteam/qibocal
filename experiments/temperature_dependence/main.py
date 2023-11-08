from pathlib import Path

import numpy as np
from dummy_controller import DummyTemperatureController
from qibolab import create_platform

from qibocal.protocols.characterization import Operation

controller = DummyTemperatureController("192.168.0.197")
temperature_range = np.arange(0, 0.15, 0.01)
platform = create_platform("dummy")
qubits = {0: platform.qubits[0]}
path = Path("run0")
path.mkdir(parents=True, exist_ok=True)

for temperature in temperature_range:
    # create folder temperature
    temperature_path = path / f"T_{temperature}"
    temperature_path.mkdir(parents=True, exist_ok=True)

    # run readout characterization to obtain effective temperature
    protocol = Operation.readout_characterization.value
    params = protocol.parameters_type.load(dict(nshots=1024))
    data, time = protocol.acquisition(params=params, platform=platform, qubits=qubits)
    readout_path = temperature_path / "readout_characterization"
    readout_path.mkdir(parents=True, exist_ok=True)
    data.save(readout_path)
    results, time = protocol.fit(data)
    results.save(readout_path)

    # run qubit spectroscopy
    protocol = Operation.qubit_spectroscopy.value
    params = protocol.parameters_type.load(
        dict(
            freq_width=10_000_000,
            freq_step=100_000,
            drive_duration=5_000,
            drive_amplitude=0.05,
            nshots=2000,
        )
    )
    data, time = protocol.acquisition(params=params, platform=platform, qubits=qubits)
    qubit_spec_path = temperature_path / "qubit_spectroscopy"
    qubit_spec_path.mkdir(parents=True, exist_ok=True)
    data.save(qubit_spec_path)
    results, time = protocol.fit(data)
    results.save(qubit_spec_path)

    # run t1
    protocol = Operation.t1_msr.value
    params = protocol.parameters_type.load(
        dict(
            delay_before_readout_start=4,
            delay_before_readout_end=8_000,
            delay_before_readout_step=5_000,
            nshots=2000,
        )
    )
    data, time = protocol.acquisition(params=params, platform=platform, qubits=qubits)
    t1_path = temperature_path / "t1"
    t1_path.mkdir(parents=True, exist_ok=True)
    data.save(t1_path)
    results, time = protocol.fit(data)
    results.save(t1_path)

    # run ramsey
    protocol = Operation.ramsey_msr.value
    params = protocol.parameters_type.load(
        dict(
            delay_between_pulses_start=4,
            delay_between_pulses_end=2_000,
            delay_between_pulses_step=8,
            nshots=2000,
        )
    )
    data, time = protocol.acquisition(params=params, platform=platform, qubits=qubits)
    ramsey_path = temperature_path / "ramsey"
    ramsey_path.mkdir(parents=True, exist_ok=True)
    data.save(ramsey_path)
    results, time = protocol.fit(data)
    results.save(ramsey_path)
