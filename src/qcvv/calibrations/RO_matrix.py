# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qcvv.calibrations import utils
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def RO_matrix(
    platform: AbstractPlatform,
    qubit,
    niter,
    points=10,
):
    platform.reload_settings()
    nqubits = platform.settings["nqubits"]
    data = Dataset(name=f"data_q{qubit}")

    # Init RO_matrix[2^5][2^5] with 0
    RO_matrix = [[0 for x in range(2**nqubits)] for y in range(2**nqubits)]
    count = 0
    # for all possible states 2^5 --> |00000> ... |11111>
    for k in range(2**nqubits):
        # repeat multiqubit state sequence niter times
        for j in range(niter):
            if count % points == 0:
                yield data
            # covert the multiqubit state i into binary representation
            multiqubit_state = bin(int(k))[2:].zfill(nqubits)
            print(f"binary state: {multiqubit_state}")
            ro_serials = []
            # multiqubit_state = |00000>, |00001> ... |11111>
            for n in multiqubit_state:
                # n = qubit_0 value ... qubit_4 value of a given state
                seq = PulseSequence()
                if n == "1":
                    # Define sequence for qubit for Pipulse state
                    RX_pulse = platform.create_RX_pulse(qubit, start=0)
                    ro_pulse = platform.create_qubit_readout_pulse(
                        qubit, start=RX_pulse.duration
                    )
                    seq.add(RX_pulse)
                    seq.add(ro_pulse)

                if n == "0":
                    # Define sequence for qubit Identity state
                    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
                    seq.add(ro_pulse)

                ro_serials.append(ro_pulse.serial)

            # Iterate over list of RO results
            res = ""
            mean_gnd_states = platform.characterization["single_qubit"][qubit][
                "mean_gnd_states"
            ]
            mean_gnd = complex(mean_gnd_states)
            mean_exc_states = platform.characterization["single_qubit"][qubit][
                "mean_exc_states"
            ]
            mean_exc = complex(mean_exc_states)

            for qubit in range(nqubits):
                # FIXME: Esto ejecuta la secuencia en cada lectura!!!
                msr, phase, i, q = platform.execute_pulse_sequence(seq, nshots=1)[
                    ro_serials[qubit]
                ]
                # classify state of qubit n
                point = complex(i, q)
                res += str(utils.classify(point, mean_gnd, mean_exc))
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                }
                data.add(results)

            yield data
            # End of processing multiqubit state i
            # populate state i with RO results obtained
            print(f"RO_matrix[{k}][{int(res, 2)}] - classified states: {res}")
            RO_matrix[k][int(res, 2)] = RO_matrix[k][int(res, 2)] + 1
        # End of repeting RO for a given state i
    # end states
    print(np.array(RO_matrix) / niter)
