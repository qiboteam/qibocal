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
        _waveforms.update(drive_waveforms(platform, q))
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
        _pulses.update(drive_pulses(platform, q))
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
            "con9": {
                "analog_inputs": {
                    "1": {
                        "offset": 0.0,
                        "gain_db": 0,
                    },
                    "2": {
                        "offset": 0.0,
                        "gain_db": 0,
                    },
                },
                "digital_outputs": {},
                "analog_outputs": {
                    "3": {
                        "offset": -0.13,
                        "filter": {},
                    },
                    "4": {
                        "offset": 0.134,
                        "filter": {},
                    },
                    "5": {
                        "offset": -0.438,
                        "filter": {},
                    },
                    "6": {
                        "offset": -0.002,
                        "filter": {},
                    },
                    "7": {
                        "offset": -0.031,
                        "filter": {},
                    },
                },
            },
            "con6": {
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
                    "3": {},
                    "7": {},
                    "9": {},
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
                    "3": {
                        "offset": 0.0,
                        "filter": {},
                    },
                    "4": {
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
                    "9": {
                        "offset": 0.0,
                        "filter": {},
                    },
                    "10": {
                        "offset": 0.0,
                        "filter": {},
                    },
                },
            },
            "con8": {
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
                    "9": {},
                    "5": {},
                },
                "analog_outputs": {
                    "9": {
                        "offset": 0.0,
                        "filter": {},
                    },
                    "10": {
                        "offset": 0.0,
                        "filter": {},
                    },
                    "5": {
                        "offset": 0.0,
                        "filter": {},
                    },
                    "6": {
                        "offset": 0.0,
                        "filter": {},
                    },
                },
            },
        },
        "octaves": {
            "octave5": {
                "RF_outputs": {
                    "1": {
                        "LO_frequency": 7450000000,
                        "gain": 0,
                        "LO_source": "internal",
                        "output_mode": "triggered",
                    },
                    "2": {
                        "LO_frequency": 5100000000,
                        "gain": 0,
                        "LO_source": "internal",
                        "output_mode": "triggered",
                    },
                    "4": {
                        "LO_frequency": 5700000000,
                        "gain": 0,
                        "LO_source": "internal",
                        "output_mode": "triggered",
                    },
                    "5": {
                        "LO_frequency": 5700000000,
                        "gain": 0,
                        "LO_source": "internal",
                        "output_mode": "triggered",
                    },
                },
                "connectivity": "con6",
                "RF_inputs": {
                    "1": {
                        "LO_frequency": 7450000000,
                        "LO_source": "internal",
                        "IF_mode_I": "direct",
                        "IF_mode_Q": "direct",
                    },
                },
            },
            "octave6": {
                "RF_outputs": {
                    "5": {
                        "LO_frequency": 6400000000,
                        "gain": 0,
                        "LO_source": "internal",
                        "output_mode": "triggered",
                    },
                    "3": {
                        "LO_frequency": 5700000000,
                        "gain": 0,
                        "LO_source": "internal",
                        "output_mode": "triggered",
                    },
                },
                "connectivity": "con8",
            },
        },
        "elements": {
            "fluxD1": {
                "singleInput": {
                    "port": ("con9", 3),
                },
                "intermediate_frequency": 0,
                "operations": {},
            },
            "fluxD2": {
                "singleInput": {
                    "port": ("con9", 4),
                },
                "intermediate_frequency": 0,
                "operations": {},
            },
            "fluxD3": {
                "singleInput": {
                    "port": ("con9", 5),
                },
                "intermediate_frequency": 0,
                "operations": {},
            },
            "fluxD4": {
                "singleInput": {
                    "port": ("con9", 6),
                },
                "intermediate_frequency": 0,
                "operations": {},
            },
            "fluxD5": {
                "singleInput": {
                    "port": ("con9", 7),
                },
                "intermediate_frequency": 0,
                "operations": {},
            },
            "readoutD1": {
                "RF_inputs": {
                    "port": ("octave5", 1),
                },
                "RF_outputs": {
                    "port": ("octave5", 1),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con6", 1),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": -312480000,
                "operations": {"measure": "mz_D1"},
                "time_of_flight": 224,
                "smearing": 0,
            },
            "readoutD2": {
                "RF_inputs": {
                    "port": ("octave5", 1),
                },
                "RF_outputs": {
                    "port": ("octave5", 1),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con6", 1),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": -70432000,
                "operations": {"measure": "mz_D2"},
                "time_of_flight": 224,
                "smearing": 0,
            },
            "readoutD3": {
                "RF_inputs": {
                    "port": ("octave5", 1),
                },
                "RF_outputs": {
                    "port": ("octave5", 1),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con6", 1),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": 40960000,
                "operations": {"measure": "mz_D3"},
                "time_of_flight": 224,
                "smearing": 0,
            },
            "readoutD4": {
                "RF_inputs": {
                    "port": ("octave5", 1),
                },
                "RF_outputs": {
                    "port": ("octave5", 1),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con6", 1),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": 256570000,
                "operations": {"measure": "mz_D4"},
                "time_of_flight": 224,
                "smearing": 0,
            },
            "readoutD5": {
                "RF_inputs": {
                    "port": ("octave5", 1),
                },
                "RF_outputs": {
                    "port": ("octave5", 1),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con6", 1),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": 186110000,
                "operations": {"measure": "mz_D5"},
                "time_of_flight": 224,
                "smearing": 0,
            },
            "driveD1": {
                "RF_inputs": {
                    "port": ("octave5", 2),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con6", 3),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": -142488654,
                "operations": native_operations("D1"),
            },
            "driveD2": {
                "RF_inputs": {
                    "port": ("octave5", 4),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con6", 7),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": -136254288,
                "operations": native_operations("D2"),
            },
            "driveD3": {
                "RF_inputs": {
                    "port": ("octave5", 5),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con6", 9),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": -47692427,
                "operations": native_operations("D3"),
            },
            "driveD4": {
                "RF_inputs": {
                    "port": ("octave6", 5),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con8", 9),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": -150769917,
                "operations": native_operations("D4"),
            },
            "driveD5": {
                "RF_inputs": {
                    "port": ("octave6", 3),
                },
                "digitalInputs": {
                    "output_switch": {
                        "port": ("con8", 5),
                        "delay": 57,
                        "buffer": 18,
                    },
                },
                "intermediate_frequency": -173219116,
                "operations": native_operations("D5"),
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
