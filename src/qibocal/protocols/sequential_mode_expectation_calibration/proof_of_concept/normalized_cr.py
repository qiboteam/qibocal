import numpy as np
import matplotlib.pyplot as plt
from normalize import min_max_normalize

from qibolab import create_platform, ExecutionParameters, AveragingMode, AcquisitionType
from qibolab.pulses import PulseSequence

CRTL = 2
TGT = 0

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

crtl_pi_pulse = platform.create_RX_pulse(qubit=CRTL, start=5)
#tgt_pi_pulse = platform.create_RX_pulse(qubit=TGT, start=5)
#pulse_length = max(crtl_pi_pulse.finish,tgt_pi_pulse.finish)
tgt_ro_pulse = platform.create_qubit_readout_pulse(qubit=TGT, start=crtl_pi_pulse.finish)
cr_pulse = platform.create_RX_pulse(qubit=TGT, start=crtl_pi_pulse.finish)
cr_pulse.channel = crtl_pi_pulse.channel
cr_pulse.amplitude = 1

ps = PulseSequence(*[cr_pulse, tgt_ro_pulse])
for idx, t in enumerate(sweep):
    cr_pulse.duration = t
    tgt_ro_pulse.start = cr_pulse.finish
    gnd[idx] = platform.execute_pulse_sequence(ps, opts)[tgt_ro_pulse.serial].magnitude

ps.add(crtl_pi_pulse)
for idx, t in enumerate(sweep):
    cr_pulse.duration = t
    tgt_ro_pulse.start = cr_pulse.finish
    exc[idx] = platform.execute_pulse_sequence(ps, opts)[tgt_ro_pulse.serial].magnitude

np.save(f"./raw_data/crtl_0_cr_{CRTL}{TGT}", gnd)
np.save(f"./raw_data/crtl_1_cr_{CRTL}{TGT}", exc)

gnd_baseline = np.load("./baseline_data/baseline_gnd_Q{TGT}.npy")  
exc_baseline = np.load("./baseline_data/baseline_exc_Q{TGT}.npy")     

gnd = min_max_normalize(gnd_baseline,exc_baseline, gnd)
exc = min_max_normalize(gnd_baseline,exc_baseline, exc)

np.save(f"./normalize_data/crtl_0_cr_{CRTL}{TGT}", gnd)
np.save(f"./normalize_data/crtl_1_cr_{CRTL}{TGT}", exc)

plt.plot(sweep, gnd, color="blue",marker='o', linestyle='-', label=r"$Q_{CRTL} = |0\rangle$")
plt.plot(sweep, exc, color="orange",marker='o', linestyle='-', label=r"$Q_{CRTL} = |1\rangle$")

plt.grid()
plt.xlabel("CR Pulse Duration [ns]")
plt.ylabel("Expectation Value")
plt.legend()
plt.title(f"Q{CRTL + 1} as control, Q{TGT + 1} as target")
plt.tight_layout()
plt.savefig(f"./plots/CR_{CRTL}{TGT}.png", dpi=300)
