import json

import pandas as pd
from qibo import gates
from qibo.models import Circuit

from qibocal.auto.operation import DATAFILE


def circ_to_json(circuit):
    circ_json = []
    # Look into circuit.moments for noise matters
    # when then get implemented in Qibo
    for gate in circuit.queue:
        circ_json.append(gate.to_json())
    return circ_json


# Make in a nicer general way
def json_tocircuit(circuit, nqubits):
    gatelist = []
    for gate_json in circuit:
        gate = json.loads(gate_json)
        if gate["name"] == "u3":
            gatelist.append(
                gates.U3(
                    gate["_target_qubits"][0],
                    gate["init_kwargs"]["theta"],
                    gate["init_kwargs"]["phi"],
                    gate["init_kwargs"]["lam"],
                )
            )
        if gate["name"] == "id":
            gatelist.append(gates.I(gate["_target_qubits"][0]))
        if gate["name"] == "rz":
            gatelist.append(
                gates.RZ(gate["_target_qubits"][0], gate["init_kwargs"]["theta"])
            )
        if gate["name"] == "rx":
            gatelist.append(
                gates.RX(gate["_target_qubits"][0], gate["init_kwargs"]["theta"])
            )
        if gate["name"] == "ry":
            gatelist.append(
                gates.RY(gate["_target_qubits"][0], gate["init_kwargs"]["theta"])
            )
        if gate["name"] == "z":
            gatelist.append(gates.Z(gate["_target_qubits"][0]))
        if gate["name"] == "x":
            gatelist.append(gates.X(gate["_target_qubits"][0]))
        if gate["name"] == "y":
            gatelist.append(gates.Y(gate["_target_qubits"][0]))

    nqubits = max(max(gate.qubits) for gate in gatelist) + 1
    new_circuit = Circuit(nqubits)
    new_circuit.add(gatelist)
    return new_circuit


class RBData(pd.DataFrame):
    """A pandas DataFrame child. The output of the acquisition function."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def save(self, path):
        """Overwrite because qibocal action builder calls this function with a directory."""
        save_copy = self.copy()
        for index, circuit in enumerate(self.circuit):
            save_copy.at[index, "circuit"] = circ_to_json(circuit)

        save_copy.to_json(path / DATAFILE, default_handler=str)

    # When loading add inverse gate or measurament if needed.
    # I skipped them as they may not be needed.
    @classmethod
    def load(cls, path):
        new_data = cls(pd.read_json(path / DATAFILE))

        nqubits = len(new_data.samples[0][0])
        for index, circuit in enumerate(new_data.circuit):
            new_data.at[index, "circuit"] = json_tocircuit(circuit, nqubits)

        return new_data
