import json
from dataclasses import dataclass

from qibo import Circuit
from qibolab.platform import Platform
from qibolab.qubits import QubitPairId

from qibocal.auto.operation import Routine
from qibocal.protocols.randomized_benchmarking.standard_rb import (
    StandardRBParameters,
    _plot,
)

from .utils import RB2QData, StandardRBResult, fit, twoq_rb_acquisition

FILE_CLIFFORDS = "2qubitCliffs.json"
FILE_INV = "2qubitCliffsInv.npz"
targets = ("D1", "D2")
root_path = "test_report"
platform = "qw11q"


@dataclass
class StandardRB2QParameters(StandardRBParameters):
    """Parameters for the standard 2q randomized benchmarking protocol."""

    file: str = FILE_CLIFFORDS
    """File with the cliffords to be used."""
    file_inv: str = FILE_INV
    """File with the cliffords to be used in an inverted dict."""


def _acquisition(
    params: StandardRB2QParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> RB2QData:
    """Data acquisition for two qubit Standard Randomized Benchmarking."""

    xxx = twoq_rb_acquisition(params, platform, targets)
    # print("ACQUISITION PRINT LOL")
    # print(xxx)

    return xxx  # twoq_rb_acquisition(params, platform, targets)


def save_circuits(bibi_circuits):
    print("\n\nSaving circuits to json\n\n")

    for i in range(
        0,
        len(
            bibi_circuits,
        ),
    ):
        temp_dict = bibi_circuits[i].raw
        # if i == 0:
        # print(temp_dict)
        with open("my_result_" + str(i) + ".json", "w") as f:
            json.dump(temp_dict, f)


def load_circuits(bibi_circuits, N):
    print("\n\nLoading circuits from json\n\n")

    for i in range(0, N):
        with open("my_result_" + str(i) + ".json") as f:
            cd = json.load(f)
            c = Circuit(2)
            c = c.from_dict(cd)
            bibi_circuits.append(c)


def _fit(data: RB2QData) -> StandardRBResult:
    qubits = data.pairs
    results = fit(qubits, data)

    from .utils import bibi_circuits

    bibi_circuits.sort(key=lambda c: len(c.gates_of_type("cz")))

    i = 0
    for c in bibi_circuits:
        print("Summary of circuit number " + str(i))
        print(c.summary())
        # print("Counting CZ gates")
        print(len(c.gates_of_type("cz")))
        print(c)
        i = i + 1

    print("TESTING SAVE/LOAD OF CIRCUITS\n")

    save_circuits(bibi_circuits)
    bibi_circuits_loaded = []
    load_circuits(bibi_circuits_loaded, len(bibi_circuits))

    for c in bibi_circuits_loaded:
        print(len(c.gates_of_type("cz")))

    print("END OF TESTING")

    return results


standard_rb_2q = Routine(_acquisition, _fit, _plot)
