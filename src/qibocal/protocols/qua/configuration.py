from dataclasses import asdict

from qibolab import Pulse, Rectangular
from qibolab.instruments.qm import QmController

__all__ = ["generate_config"]

NATIVE_OPS = {
    "x180": lambda q: (f"plus_i_{q}", f"plus_q_{q}"),
    "y180": lambda q: (f"minus_q_{q}", f"plus_i_{q}"),
    "x90": lambda q: (f"plus_i_half_{q}", f"plus_q_half_{q}"),
    "y90": lambda q: (f"minus_q_half_{q}", f"plus_i_half_{q}"),
    "-x90": lambda q: (f"minus_i_half_{q}", f"minus_q_half_{q}"),
    "-y90": lambda q: (f"plus_q_half_{q}", f"minus_i_half_{q}"),
}
MAX_OUTPUT = 0.5


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


def get_sampling_rate(platform, channel):
    config = platform.config(channel)
    if hasattr(config, "sampling_rate"):
        return int(config.sampling_rate / 1e9)
    return 1


def get_max_output(platform, channel):
    config = platform.config(channel)
    if hasattr(config, "output_mode") and config.output_mode == "amplified":
        return 2.5
    return MAX_OUTPUT


def drive_waveforms(platform, qubit):
    channel, pulse = platform.natives.single_qubit[qubit].RX[0]
    sampling_rate = get_sampling_rate(platform, channel)
    envelope_i = MAX_OUTPUT * pulse.i(sampling_rate=sampling_rate)
    components_i = drive_waveform_components(qubit, "i", envelope_i)
    envelope_q = MAX_OUTPUT * pulse.q(sampling_rate=sampling_rate)
    components_q = drive_waveform_components(qubit, "q", envelope_q)
    return components_i | components_q


def baked_duration(duration: int) -> int:
    return int(max((duration + 3.5) // 4 * 4, 16))


def flux_waveforms(platform, qubit):
    _waveforms = {}
    for (q1, q2), natives in platform.natives.two_qubit.items():
        cz = natives.CZ
        if cz is not None:
            channel, pulse = [t for t in cz if isinstance(t[1], Pulse)][0]
            sampling_rate = get_sampling_rate(platform, channel)
            max_output = get_max_output(platform, channel)
            if channel == platform.qubits[qubit].flux:
                other = q2 if q1 == qubit else q1
                samples = (max_output * pulse.i(sampling_rate=sampling_rate)).tolist()
                new_duration = baked_duration(pulse.duration)
                pad_len = sampling_rate * new_duration - len(samples)
                samples.extend(pad_len * [0.0])
                _waveforms[f"cz_{qubit}_{other}"] = {
                    "type": "arbitrary",
                    "samples": samples,
                }
    return _waveforms


def waveforms(platform, qubits):
    _waveforms = {
        "zero": {
            "type": "constant",
            "sample": 0.0,
        },
    }
    for q in qubits:
        acq_channel, readout = platform.natives.single_qubit[q].MZ[0]
        pulse = readout.probe
        channel = platform.channels[acq_channel].probe
        if isinstance(pulse.envelope, Rectangular):
            _waveforms[f"mz_{q}"] = {
                "type": "constant",
                "sample": MAX_OUTPUT * pulse.amplitude,
            }
        else:
            sampling_rate = get_sampling_rate(platform, channel)
            _waveforms[f"mz_{q}_i"] = {
                "type": "arbitrary",
                "samples": MAX_OUTPUT * pulse.i(sampling_rate=sampling_rate),
            }
            _waveforms[f"mz_{q}_q"] = {
                "type": "arbitrary",
                "samples": MAX_OUTPUT * pulse.q(sampling_rate=sampling_rate),
            }
    for q in qubits:
        _waveforms.update(drive_waveforms(platform, q))
        _waveforms.update(flux_waveforms(platform, q))
    return _waveforms


def drive_pulses(platform, qubit):
    _pulses = {}
    for op, wf in NATIVE_OPS.items():
        i, q = wf(qubit)
        duration = int(platform.natives.single_qubit[qubit].RX[0][1].duration)
        _pulses[f"{op}_{qubit}"] = {
            "operation": "control",
            "length": duration,
            "waveforms": {
                "I": i,
                "Q": q,
            },
            "digital_marker": "ON",
        }
    return _pulses


def flux_pulses(platform, qubit):
    _pulses = {}
    for (q1, q2), natives in platform.natives.two_qubit.items():
        cz = natives.CZ
        if cz is not None:
            channel, pulse = [t for t in cz if isinstance(t[1], Pulse)][0]
            if channel == platform.qubits[qubit].flux:
                other = q2 if q1 == qubit else q1
                _pulses[f"cz_{qubit}_{other}"] = {
                    "operation": "control",
                    "length": baked_duration(pulse.duration),
                    "waveforms": {
                        "single": f"cz_{qubit}_{other}",
                    },
                }
    return _pulses


def pulses(platform, qubits):
    _pulses = {}
    for q in qubits:
        pulse = platform.natives.single_qubit[q].MZ[0][1].probe
        if isinstance(pulse.envelope, Rectangular):
            mz_waveforms = {"I": f"mz_{q}", "Q": "zero"}
        else:
            mz_waveforms = {"I": f"mz_{q}_i", "Q": f"mz_{q}_q"}
        _pulses[f"mz_{q}"] = {
            "operation": "measurement",
            "length": int(pulse.duration),
            "waveforms": mz_waveforms,
            "integration_weights": {
                "cos": f"cosine_weights{q}",
                "sin": f"sine_weights{q}",
                "minus_sin": f"minus_sine_weights{q}",
            },
            "digital_marker": "ON",
        }
        _pulses.update(drive_pulses(platform, q))
        _pulses.update(flux_pulses(platform, q))
    return _pulses


def integration_weights(platform, qubits):
    _integration_weights = {}
    for q in qubits:
        _duration = platform.natives.single_qubit[q].MZ[0][1].probe.duration
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
    controller = platform._controller
    assert isinstance(controller, QmController)

    channel_configs = platform.parameters.configs
    for q in qubits:
        for channel in platform.qubits[q].channels:
            if channel in channel_configs:
                controller.configure_channel(channel, channel_configs)

    config = controller.config
    config.elements = {k: asdict(v) for k, v in config.elements.items()}
    for q in qubits:
        qubit = platform.qubits[q]
        config.elements[qubit.acquisition]["operations"]["measure"] = f"mz_{q}"
        config.elements[qubit.drive]["operations"] = native_operations(q)
        if targets is not None and q == targets[0]:
            q1, q2 = targets
            config.elements[qubit.flux]["operations"]["cz"] = f"cz_{q1}_{q2}"
    config.pulses = pulses(platform, qubits)
    config.waveforms = waveforms(platform, qubits)
    config.integration_weights = integration_weights(platform, qubits)
    return asdict(config)
