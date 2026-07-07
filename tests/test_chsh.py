import importlib

import numpy as np

from qibocal.protocols.two_qubit_interaction.chsh.protocol import (
    CHSHParameters,
    _acquisition,
)

chsh_protocol = importlib.import_module(
    "qibocal.protocols.two_qubit_interaction.chsh.protocol"
)


def test_chsh_acquisition_defers_mitigation_until_frequencies_are_complete(
    platform, monkeypatch
):
    pair = list(platform.qubits)[:2]
    platform.calibration.set_readout_mitigation_matrix_element(
        pair,
        {tuple(pair): np.eye(4)},
    )

    monkeypatch.setattr(chsh_protocol, "build_native_gate_transpiler", lambda _: None)
    monkeypatch.setattr(chsh_protocol, "build_native_gate_compiler", lambda _: None)
    monkeypatch.setattr(
        chsh_protocol,
        "execute_circuits",
        lambda circuits, *args, **kwargs: [{"00": 1}] * len(circuits),
    )

    params = CHSHParameters(bell_states=[0], ntheta=2)
    params.nshots = 1
    params.relaxation_time = None

    data = _acquisition(params, platform, [tuple(pair)])

    assert set(data.frequencies[0][0]) == {"00", "01", "10", "11"}
    assert data.frequencies[0][0]["00"] == [1, 1]
    assert len(data.mitigated_frequencies[0]) == 4
    for basis in data.mitigated_frequencies[0]:
        for state in ("00", "01", "10", "11"):
            assert len(basis[state]) == params.ntheta
