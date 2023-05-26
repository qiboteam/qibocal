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
directory = "rb_results4"
if not os.path.isdir(directory):
    os.mkdir(directory)


nqubits = 5
niter = 5
nshots = 1024
# all_qubits = [[k] for k in range(4)]
all_qubits = [[2]]
standardrb_results = []

depths = [1, 5, 10, 15, 20]
for qubits in all_qubits:
    params = standard_rb.RBParameters(nqubits, qubits, depths, niter, nshots)
    scan = standard_rb.StandardRBScan(params.nqubits, params.depths, params.qubits)
    data_list = []
    # Iterate through the scan and execute each circuit.
    for c in scan:
        # The inverse and measurement gate don't count for the depth.
        depth = (c.depth - 2) if c.depth > 1 else 0
        platform.stop()
        platform.disconnect()

        qibo.set_backend(backend=backend_name, platform=platform_name, runcard=runcard)
        backend = qibo.backends.GlobalBackend()
        platform = backend.platform
        platform.connect()
        platform.setup()
        platform.start()

        samples = c.execute(nshots=params.nshots).samples()
        # Every executed circuit gets a row where the data is stored.
        data_list.append({"depth": depth, "samples": samples})
    data = standard_rb.RBData(data_list)
    data.attrs = params.__dict__
    # data = standard_rb.acquire(params)
    result = standard_rb.extract(data)
    result.fit()
    result.calculate_fidelities()
    standardrb_results.append(result)
    with open(f"{directory}/standard_qubit{qubits[0]}.pkl", "wb") as f:
        pickle.dump(result, f)
    print(result.fidelity_dict)

depths = list(range(1, 50, 10))
for qubits in all_qubits:
    params = xid_rb.RBParameters(nqubits, qubits, depths, niter, nshots)
    scan = xid_rb.XIdScan(params.nqubits, params.depths, params.qubits)
    data_list = []
    # Iterate through the scan and execute each circuit.
    for c in scan:
        # The inverse and measurement gate don't count for the depth.
        depth = (c.depth - 2) if c.depth > 1 else 0
        platform.stop()
        platform.disconnect()

        qibo.set_backend(backend=backend_name, platform=platform_name, runcard=runcard)
        backend = qibo.backends.GlobalBackend()
        platform = backend.platform
        platform.connect()
        platform.setup()
        platform.start()
        nx = len(c.gates_of_type("x"))
        samples = c.execute(nshots=params.nshots).samples()
        # Every executed circuit gets a row where the data is stored.
        data_list.append({"depth": depth, "samples": samples, "nx": nx})
    data = xid_rb.RBData(data_list)
    data.attrs = params.__dict__
    # data = xid_rb.acquire(params)
    result = xid_rb.extract(data)
    result.fit()
    # result.calculate_fidelities()
    standardrb_results.append(result)
    with open(f"{directory}/xid_qubit{qubits[0]}.pkl", "wb") as f:
        pickle.dump(result, f)
    # print(result.fidelity_dict)
# depths = [1, 3, 5, 7, 9]
# for qubits in all_qubits:
#     params = xid_rb.RBParameters(nqubits, qubits, depths, niter, nshots)
#     data = xid_rb.acquire(params)
#     result = xid_rb.extract(data)
#     result.fit()
#     standardrb_results.append(result)
#     with open(f"{directory}/xid_qubit{qubits[0]}.pkl", "wb") as f:
#         pickle.dump(result, f)

platform.stop()
platform.disconnect()
