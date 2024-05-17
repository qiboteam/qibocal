"""Run pulse on qubit to find their baseline value"""

import numpy as np

from qibolab import create_platform, ExecutionParameters, AveragingMode, AcquisitionType
from qibolab.pulses import PulseSequence

QUBIT = 2

opts = ExecutionParameters(
    nshots=1000,
    relaxation_time=200e3,
    acquisition_type=AcquisitionType.INTEGRATION,
    averaging_mode=AveragingMode.SEQUENTIAL
)

platform = create_platform("icarusq_iqm5q")
platform.connect()

sweep = np.arange(0, 5000, 100)
gnd = np.zeros(len(sweep))
exc = np.zeros(len(sweep))

pi_pulse = platform.create_RX_pulse(qubit=QUBIT, start=5)
#tgt_pi_pulse = platform.create_RX_pulse(qubit=TGT, start=5)
#pulse_length = max(crtl_pi_pulse.finish,tgt_pi_pulse.finish)
ro_pulse = platform.create_qubit_readout_pulse(qubit=QUBIT, start=pi_pulse.finish)

# Readout on |0>
ps = PulseSequence(*[ro_pulse])
for idx, t in enumerate(sweep):
    ro_pulse.start = pi_pulse.finish
    gnd[idx] = platform.execute_pulse_sequence(ps, opts)[ro_pulse.serial].magnitude

# Readout on |1>
ps.add(pi_pulse)
for idx, t in enumerate(sweep):
    pi_pulse.duration = t
    ro_pulse.start = pi_pulse.finish
    exc[idx] = platform.execute_pulse_sequence(ps, opts)[ro_pulse.serial].magnitude

np.save(f"./baseline_data/baseline_gnd_Q{QUBIT}", np.mean(gnd))
np.save(f"./baseline_data/baseline_exc_Q{QUBIT}", np.mean(exc))