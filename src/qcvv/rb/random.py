import numpy as np
from qibo import gates, models
from qibo.noise import NoiseModel
from qcvv.rb.clifford import OneQubitGate


def measure(queue, nshots=100, isHardware=False):
    """Perform sampling for all circuits
    Parameters:
        queue (Experiment): Total number of qubits in the circuit.
        nshots (int): Number of shots for each circuit

    Returns:
        Experiment with measurements performed.
    """
    queue.nshots = nshots
    for i in queue:
        i.execute(nshots=nshots, isHardware=isHardware)
    return queue

class Data():
    """Simple data structure containing a circuit, number of qubits,
       length and samples.
    """
    def __init__(self, nqubits=1, circuit=None, length=None):
        self.nqubits = nqubits
        self.circuit = circuit
        if length == None:
            self.length = int(len(circuit.queue) / 2 - 1)
        else:
            self.length = length
        self.samples = None
        self.probabilities=None

    def execute(self, nshots=None, isHardware=False):
        if isHardware:
            self.probabilities = self.circuit(nshots=nshots).probabilities()
        else:
            self.samples = self.circuit(nshots=nshots).samples()

class Experiment(list):
    """List that holds the queue of :class:`qiborb.random.Data`"""
    def __init__(self, nqubits = 1, nshots=1024):
        self.nqubits = nqubits
        self.lengths = set()
        self.nshots = nshots

    def append(self, circuit, length):
        data = Data(nqubits=circuit.nqubits, circuit=circuit, length=length)
        self.lengths.add(data.length)
        assert self.nqubits == data.nqubits, "Wrong number of qubits!"
        super().append(data)


class CircuitGenerator():
    """Class for generating random circuit"""
    def __init__(self, nqubits = 1, length=10, invert=True, group="Clifford", noiseModel=None):
        self.nqubits = nqubits
        self.length = length
        self.group = group
        self.gate = None
        if group == "Clifford" and nqubits == 1:
            from qcvv.rb.clifford import OneQubitGate
            self.gate = OneQubitGate
            #self.gates = [gates.X, gates.Y, gates.H] # gates.S, gates.SDG,
        else:
            raise RuntimeError("Unknown set of gates.")
        self.invert = invert
        self.noiseModel = noiseModel

    def __call__(self, length=None):
        if length == None:
            length = self.length

        circuit = models.Circuit(self.nqubits)
        for _ in range(length):
            circuit.add(self.gate())
        if self.invert:
            #inverse = circuit.invert().fuse().queue[0].matrix
            #circuit.add(inverse(matrix))
            circuit.add(circuit.invert().fuse().queue[0])
        circuit.add(gates.M(0))
        if self.noiseModel is None:
            yield circuit
        else:
            noise = self.noiseModel.apply(circuit)
            yield noise