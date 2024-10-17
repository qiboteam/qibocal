from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId, QubitPairId
from qibolab.pulses import Pulse, Rectangular, PulseType, Gaussian
from qibolab.native import NativePulse
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

STATES = ["I", "X"]
"""Setup states for the cross resonance gate calibration: {Identity, RX}."""

def cr_pulse_sequence(platform: Platform, pair: QubitPairId, setup: tuple, duration:int=0, amplitude:int = None):
    target, control = pair
    tgt_setup, ctr_setup = setup
    tgt_native_rx:NativePulse = platform.qubits[target].native_gates.RX.pulse(start=0)
    ctr_native_rx:NativePulse = platform.qubits[control].native_gates.RX.pulse(start=0)

    sequence = PulseSequence()
    next_start = 0
    if tgt_setup == 1:
        sequence.add(tgt_native_rx)
        next_start = tgt_native_rx.finish

    if ctr_setup == 1:
        sequence.add(ctr_native_rx)
        next_start = max(ctr_native_rx.finish, next_start)
    
    cr_pulse: Pulse = Pulse(start=next_start,
                    duration=duration,
                    amplitude=ctr_native_rx.amplitude,
                    frequency=tgt_native_rx.frequency,   # control frequency
                    relative_phase=0,
                    shape=Gaussian(5),
                    qubit=control,
                    channel= ctr_native_rx.channel ,type=PulseType.DRIVE
                    )

    if amplitude is not None:
        cr_pulse.amplitude = amplitude

    sequence.add(cr_pulse)

    #for qubit in pair:
    sequence.add(platform.create_qubit_readout_pulse(target, start=cr_pulse.finish))

    return sequence, cr_pulse