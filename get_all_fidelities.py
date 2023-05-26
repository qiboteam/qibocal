import os
import pickle

import qibo

from qibocal.protocols.characterization.RB import standard_rb, xid_rb

# Set backend and platform.
runcard = "/home/users/jadwiga.wilkens/qibolab/src/qibolab/runcards/qw5q_gold_qblox.yml"
backend_name = "qibolab"
platform_name = "qw5q_gold_qblox"
qibo.set_backend(backend=backend_name, platform=platform_name, runcard=runcard)
backend = qibo.backends.GlobalBackend()
platform = backend.platform
platform.connect()
platform.setup()
platform.start()
# Set directory where to store results.
directory = "rb_results2"
if not os.path.isdir(directory):
    os.mkdir(directory)


nqubits = 5
niter = 5
nshots = 1024
# all_qubits = [[k] for k in range(4)]
all_qubits = [[1]]
standardrb_results = []

depths = [1, 3, 5, 7, 10]  # ,15,20,30,40,50,80]
for qubits in all_qubits:
    params = standard_rb.RBParameters(nqubits, qubits, depths, niter, nshots)
    data = standard_rb.acquire(params)
    result = standard_rb.extract(data)
    result.fit()
    result.calculate_fidelities()
    standardrb_results.append(result)
    with open(f"{directory}/standard_qubit{qubits[0]}.pkl", "wb") as f:
        pickle.dump(result, f)
    print(result.fidelity_dict)

depths = [1, 3, 5, 7, 9]
for qubits in all_qubits:
    params = xid_rb.RBParameters(nqubits, qubits, depths, niter, nshots)
    data = xid_rb.acquire(params)
    result = xid_rb.extract(data)
    result.fit()
    standardrb_results.append(result)
    with open(f"{directory}/xid_qubit{qubits[0]}.pkl", "wb") as f:
        pickle.dump(result, f)

platform.stop()
platform.disconnect()
