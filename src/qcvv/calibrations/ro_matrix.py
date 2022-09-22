# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qcvv import plots
from qcvv.calibrations import utils
from qcvv.data import Dataset
from qcvv.decorators import plot


# RO Matrix
@plot("Readout Matrix", plots.ro_matrix)
def RO_matrix(
    platform: AbstractPlatform,
    qubit,
    niter,
    points=10,
):
    platform.reload_settings()
    nqubits = platform.settings["nqubits"]

    raw_data = Dataset(
        name=f"raw_data",
        quantities={"state": "dimensionless", "qubit": "dimensionless"},
    )
    probabilities = Dataset(name=f"probabilities")

    from pint import UnitRegistry

    ureg = UnitRegistry()

    headers_dict = {}
    for int_state in range(2**nqubits):
        str_state = bin(int(int_state))[2:].zfill(nqubits)
        headers_dict[str_state] = pd.Series(dtype="pint[dimensionless]")

    probabilities.df = pd.DataFrame(headers_dict)

    # Init RO_matrix[2^5][2^5] with 0
    RO_matrix = [
        [np.float64(0) * ureg("dimensionless") for x in range(2**nqubits)]
        for y in range(2**nqubits)
    ]

    # Init drive and readout pulses
    drive_pulses = []
    readout_pulses = []
    for n in range(nqubits):
        qd_pulse = platform.create_RX_pulse(n, start=0)
        ro_pulse = platform.create_qubit_readout_pulse(n, start=qd_pulse.finish)
        drive_pulses.append(qd_pulse)
        readout_pulses.append(ro_pulse)

    count = 0
    # for all possible states 2^5 --> |00000> ... |11111>
    for int_state_prepared in range(2**nqubits):
        # repeat multiqubit state sequence niter times
        for j in range(niter):
            if count % points == 0:
                yield probabilities
                yield raw_data

            # covert the multiqubit state i into binary representation
            str_state_prepared = bin(int(int_state_prepared))[2:].zfill(nqubits)
            # str_state_prepared = |00000>, |00001> ... |11111>
            print(f"binary state prepared: {str_state_prepared}")

            seq = PulseSequence()
            for n in range(nqubits):
                if int_state_prepared & (2**n) != 0:
                    seq += drive_pulses[n]
            seq.add(*readout_pulses)

            # Iterate over list of RO results
            res = ""
            results = platform.execute_pulse_sequence(seq, nshots=1)

            int_state_read = 0
            for n in range(nqubits):
                msr, phase, i, q = results[readout_pulses[n].serial]
                # classify state of qubit n
                point = complex(i, q)
                mean_gnd_states = platform.characterization["single_qubit"][n][
                    "mean_gnd_states"
                ]
                mean_gnd = complex(mean_gnd_states)
                mean_exc_states = platform.characterization["single_qubit"][n][
                    "mean_exc_states"
                ]
                mean_exc = complex(mean_exc_states)
                int_state_read += (2**n) * utils.classify(point, mean_gnd, mean_exc)

                sample = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "state[dimensionless]": np.int64(int_state_prepared),
                    "qubit[dimensionless]": np.int64(n),
                }
                raw_data.add(sample)

            str_state_read = bin(int(int_state_read))[2:].zfill(nqubits)
            print(f"binary state read: {str_state_read}")
            print(f"int state read: {int_state_read}")

            print(f"RO_matrix[x][y]: [{int_state_prepared}][{int_state_read}]")

            RO_matrix[int_state_prepared][int_state_read] += (
                np.float64(1) / niter * ureg("dimensionless")
            )

            for n in range(2**nqubits):
                str_state_read = bin(int(n))[2:].zfill(nqubits)
                probabilities.df.loc[int_state_prepared, str_state_read] = RO_matrix[
                    int_state_prepared
                ][n]
            yield probabilities
            yield raw_data
            # End of processing multiqubit state i
            # populate state i with RO results obtained
            # print(f"RO_matrix[{int_state_prepared}][{int(res, 2)}] - classified states: {res}")
            # RO_matrix[int_state_prepared][int(res, 2)] = RO_matrix[int_state_prepared][int(res, 2)] + 1
        # End of repeting RO for a given state i
    # end states
    print(np.array(RO_matrix))
