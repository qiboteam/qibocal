from dataclasses import asdict

from qibolab.instruments.qm import QMController
from qibolab.instruments.qm.config import QMConfig

NATIVE_OPS = {
    "x180": lambda q: (f"plus_i_{q}", f"plus_q_{q}"),
    "y180": lambda q: (f"minus_q_{q}", f"plus_i_{q}"),
    "x90": lambda q: (f"plus_i_half_{q}", f"plus_q_half_{q}"),
    "y90": lambda q: (f"minus_q_half_{q}", f"plus_i_half_{q}"),
    "-x90": lambda q: (f"minus_i_half_{q}", f"minus_q_half_{q}"),
    "-y90": lambda q: (f"plus_q_half_{q}", f"minus_i_half_{q}"),
}


def native_operations(qubit):
    return {op: f"{op}_{qubit}" for op in NATIVE_OPS.keys()}


def drive_waveform_components(qubit, mode, samples):
    return {
        f"plus_{mode}_{qubit}": {
            "type": "arbitrary",
            "samples": samples,
        },
        f"minus_{mode}_{qubit}": {
            "type": "arbitrary",
            "samples": -samples,
        },
        f"plus_{mode}_half_{qubit}": {
            "type": "arbitrary",
            "samples": samples / 2,
        },
        f"minus_{mode}_half_{qubit}": {
            "type": "arbitrary",
            "samples": -samples / 2,
        },
    }


def drive_waveforms(platform, qubit):
    pulse = platform.qubits[qubit].native_gates.RX.pulse(start=0)
    envelope_i, envelope_q = pulse.envelope_waveforms(sampling_rate=1)
    return drive_waveform_components(
        qubit, "i", envelope_i.data
    ) | drive_waveform_components(qubit, "q", envelope_q.data)


def flux_waveforms(platform, qubit):
    _waveforms = {}
    for (q1, q2), pair in platform.pairs.items():
        cz = pair.native_gates.CZ
        if cz is not None:
            seq, _ = cz.sequence()
            pulse = seq[0]
            if pulse.qubit == qubit:
                other = q2 if q1 == qubit else q1
                _waveforms[f"cz_{qubit}_{other}"] = {
                    "type": "constant",
                    "sample": pulse.amplitude,
                }
    return _waveforms


def waveforms(platform, qubits):
    _waveforms = {
        "zero": {
            "type": "constant",
            "sample": 0.0,
        },
    }
    _waveforms.update(
        {
            f"mz_{q}": {
                "type": "constant",
                "sample": platform.qubits[q].native_gates.MZ.amplitude,
            }
            for q in qubits
        }
    )
    for q in qubits:
        _waveforms.update(drive_waveforms(platform, q))
        _waveforms.update(flux_waveforms(platform, q))
    return _waveforms


def drive_pulses(platform, qubit):
    _pulses = {}
    for op, wf in NATIVE_OPS.items():
        i, q = wf(qubit)
        _pulses[f"{op}_{qubit}"] = {
            "operation": "control",
            "length": platform.qubits[qubit].native_gates.RX.duration,
            "waveforms": {
                "I": i,
                "Q": q,
            },
            "digital_marker": "ON",
        }
    return _pulses


def flux_pulses(platform, qubit):
    _pulses = {}
    for (q1, q2), pair in platform.pairs.items():
        cz = pair.native_gates.CZ
        if cz is not None:
            seq, _ = cz.sequence()
            pulse = seq[0]
            if pulse.qubit == qubit:
                other = q2 if q1 == qubit else q1
                _pulses[f"cz_{qubit}_{other}"] = {
                    "operation": "control",
                    "length": pulse.duration,
                    "waveforms": {
                        "single": f"cz_{qubit}_{other}",
                    },
                }
    return _pulses


def pulses(platform, qubits):
    _pulses = {
        f"mz_{q}": {
            "operation": "measurement",
            "length": platform.qubits[q].native_gates.MZ.duration,
            "waveforms": {
                "I": f"mz_{q}",
                "Q": "zero",
            },
            "integration_weights": {
                "cos": f"cosine_weights{q}",
                "sin": f"sine_weights{q}",
                "minus_sin": f"minus_sine_weights{q}",
            },
            "digital_marker": "ON",
        }
        for q in qubits
    }
    for q in qubits:
        _pulses.update(drive_pulses(platform, q))
        _pulses.update(flux_pulses(platform, q))
    return _pulses


def integration_weights(platform, qubits):
    _integration_weights = {}
    for q in qubits:
        _duration = platform.qubits[q].native_gates.MZ.duration
        _integration_weights.update(
            {
                f"cosine_weights{q}": {
                    "cosine": [(1.0, _duration)],
                    "sine": [(-0.0, _duration)],
                },
                f"sine_weights{q}": {
                    "cosine": [(0.0, _duration)],
                    "sine": [(1.0, _duration)],
                },
                f"minus_sine_weights{q}": {
                    "cosine": [(-0.0, _duration)],
                    "sine": [(-1.0, _duration)],
                },
            }
        )
    return _integration_weights


def register_element(config, qubit, time_of_flight, smearing):
    config.register_port(qubit.readout.port)
    config.register_port(qubit.feedback.port)
    mz_frequency = qubit.native_gates.MZ.frequency - qubit.readout.lo_frequency
    config.register_readout_element(qubit, mz_frequency, time_of_flight, smearing)
    config.register_port(qubit.drive.port)
    rx_frequency = qubit.native_gates.RX.frequency - qubit.drive.lo_frequency
    config.register_drive_element(qubit, rx_frequency)
    if qubit.flux is not None:
        config.register_port(qubit.flux.port)
        config.register_flux_element(qubit)


def generate_config(platform, qubits, targets=None):
    con = [
        instr
        for instr in platform.instruments.values()
        if isinstance(instr, QMController)
    ][0]
    config = QMConfig()
    for q in qubits:
        qubit = platform.qubits[q]
        register_element(config, qubit, con.time_of_flight, con.smearing)
        config.elements[f"readout{q}"]["operations"]["measure"] = f"mz_{q}"
        config.elements[f"drive{q}"]["operations"] = native_operations(q)
        if targets is not None and q == targets[0]:
            q1, q2 = targets
            config.elements[f"flux{q}"]["operations"]["cz"] = f"cz_{q1}_{q2}"
    config.pulses = pulses(platform, qubits)
    config.waveforms = waveforms(platform, qubits)
    config.integration_weights = integration_weights(platform, qubits)
    return asdict(config)
