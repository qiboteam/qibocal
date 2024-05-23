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
        _waveforms.update(
            drive_waveforms(q, platform.qubits[q].native_gates.RX.amplitude)
        )
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
        _pulses.update(drive_pulses(q))
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


def generate_config(platform, qubits):
    return {
        "version": 1,
        "controllers": {
            "con4": {
                "analog_inputs": {
                    "1": {
                        "offset": 0.0,
                        "gain_db": 10,
                    },
                    "2": {
                        "offset": 0.0,
                        "gain_db": 10,
                    },
                },
                "digital_outputs": {},
                "analog_outputs": {
                    "1": {
                        "offset": 0.288,
                        "filter": {},
                    },
                    "2": {
                        "offset": -0.25,
                        "filter": {},
                    },
                    "3": {
                        "offset": 0.268,
                        "filter": {
                            "feedforward": [1.0684635881381783, -1.0163217174522334],
                            "feedback": [0.947858129314055],
                        },
                    },
                    "4": {
                        "offset": 0.292,
                        "filter": {
                            "feedforward": [1.0684635881381783, -1.0163217174522334],
                            "feedback": [0.947858129314055],
                        },
                    },
                    "5": {
                        "offset": -0.49,
                        "filter": {},
                    },
                },
            },
            "con3": {
                "analog_inputs": {
                    "1": {
                        "offset": 0.0,
                        "gain_db": 10,
                    },
                    "2": {
                        "offset": 0.0,
                        "gain_db": 10,
                    },
                },
                "digital_outputs": {
                    "1": {},
                },
                "analog_outputs": {
                    "1": {
                        "offset": 0.0,
                        "filter": {},
                    },
                    "2": {
                        "offset": 0.0,
                        "filter": {},
                    },
                },
            },
            "con2": {
                "analog_inputs": {
                    "1": {
                        "offset": 0.0,
                        "gain_db": 10,
                    },
                    "2": {
                        "offset": 0.0,
                        "gain_db": 10,
                    },
                },
                "digital_outputs": {
                    "5": {},
                    "1": {},
                    "7": {},
                },
                "analog_outputs": {
                    "5": {
                        "offset": 0.0,
                        "filter": {},
                    },
                    "6": {
                        "offset": 0.0,
                        "filter": {},
                    },
                    "1": {
                        "offset": 0.0,
                        "filter": {},
                    },
                    "2": {
                        "offset": 0.0,
                        "filter": {},
                    },
                    "7": {
                        "offset": 0.0,
                        "filter": {},
                    },
                    "8": {
                        "offset": 0.0,
                        "filter": {},
                    },
                },
            },
        },
        "octaves": {
            "octave3": {
                "RF_outputs": {
                    "1": {
                        "LO_frequency": 7550000000,
                        "gain": -20,
                        "LO_source": "internal",
                        "output_mode": "triggered",
                    },
                },
                "connectivity": "con3",
                "RF_inputs": {
                    "1": {
                        "LO_frequency": 7550000000,
                        "LO_source": "internal",
                        "IF_mode_I": "direct",
                        "IF_mode_Q": "direct",
                    },
                },
            },
            "octave2": {
                "RF_outputs": {
                    "3": {
                        "LO_frequency": 5100000000.0,
                        "gain": 20,
                        "LO_source": "internal",
                        "output_mode": "triggered",
                    },
                    "1": {
                        "LO_frequency": 5800000000.0,
                        "gain": 20,
                        "LO_source": "internal",
                        "output_mode": "triggered",
                    },
                    "4": {
                        "LO_frequency": 6500000000.0,
                        "gain": -10,
                        "LO_source": "internal",
                        "output_mode": "triggered",
                    },
                },
                "connectivity": "con2",
            },
        },
        "elements": {
            "flux0": {
                "singleInput": {
                    "port": ("con4", 1),
                },
                "intermediate_frequency": 0,
                "operations": {},
            },
            "flux1": {
                "singleInput": {
                    "port": ("con4", 2),
                },
                "intermediate_frequency": 0,
                "operations": {},
            },
            "flux2": {
                "singleInput": {
                    "port": ("con4", 3),
                },
                "intermediate_frequency": 0,
                "operations": {},
            },
            "flux3": {
                "singleInput": {
                    "port": ("con4", 4),
                },
                "intermediate_frequency": 0,
                "operations": {},
            },
            "flux4": {
                "singleInput": {
                    "port": ("con4", 5),
                },
                "intermediate_frequency": 0,
                "operations": {},
            },
            "resonator0": {
                "RF_inputs": {
                    "port": ("octave3", 1),
                },
                "RF_outputs": {
                    "port": ("octave3", 1),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con3", 1),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": -341526044.0,
                "operations": {"measure": "mz_0"},
                "time_of_flight": 224,
                "smearing": 0,
            },
            "resonator2": {
                "RF_inputs": {
                    "port": ("octave3", 1),
                },
                "RF_outputs": {
                    "port": ("octave3", 1),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con3", 1),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": -54617042.0,
                "operations": {"measure": "mz_2"},
                "time_of_flight": 224,
                "smearing": 0,
            },
            "resonator3": {
                "RF_inputs": {
                    "port": ("octave3", 1),
                },
                "RF_outputs": {
                    "port": ("octave3", 1),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con3", 1),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": 118471933.0,
                "operations": {"measure": "mz_3"},
                "time_of_flight": 224,
                "smearing": 0,
            },
            "qubit0": {
                "RF_inputs": {
                    "port": ("octave2", 3),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con2", 5),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": -331021444.0,
                "operations": native_operations(0),
            },
            "qubit2": {
                "RF_inputs": {
                    "port": ("octave2", 1),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con2", 1),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": -283494000.0,
                "operations": native_operations(2),
            },
            "qubit3": {
                "RF_inputs": {
                    "port": ("octave2", 4),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con2", 7),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": -175418000.0,
                "operations": native_operations(3),
            },
        },
        "pulses": pulses(platform, qubits),
        "waveforms": waveforms(platform, qubits),
        "digital_waveforms": {
            "ON": {
                "samples": [(1, 0)],
            },
        },
        "integration_weights": integration_weights(platform, qubits),
        "mixers": {},
    }
