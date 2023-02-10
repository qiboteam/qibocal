from qibo import gates, models

from qibocal.calibrations.protocols.XIdrb import *

# nqubits = 2
# circuit = models.Circuit(nqubits)
# circuit.add([gates.X(0), gates.Z(1), gates.Y(0), gates.H(1)])
# circuit.add(gates.M(0,1))

# print(circuit.draw())
# for count in range(nqubits):
#     helper_circuit = models.Circuit(1)
#     for gate in circuit.queue[count:-1:nqubits]:
#         print(gate)
#         helper_circuit.add(gate.on_qubits({0:0, 1:0}))
#     print(helper_circuit.unitary())
#     # print(helper_circuit().execution_result)

# fused_circuit = circuit.fuse(max_qubits=1)
# print(fused_circuit.queue[0].matrix)
# print(fused_circuit.queue[1].matrix)



nqubits = 1
depths = [0, 3, 5, 10, 15]
runs = 20
nshots = 100
noise_params = [0.01, 0.05, 0.05]

# factory = XIdFactory(nqubits, depths, runs)

perform(nqubits, depths, runs, nshots, noise_params=noise_params)
