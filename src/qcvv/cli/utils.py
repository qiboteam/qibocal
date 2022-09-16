# -*- coding: utf-8 -*-
from qcvv.cli.builders import ActionBuilderHighLevel, ActionBuilderLowLevel, load_yaml
from qcvv.config import raise_error

LOW_LEVEL_KEYS = ["platform", "qubits", "format", "actions"]
HIGH_LEVEL_KEYS = ["backend", "platform", "nqubits", "format", "actions"]


def allocate_builder(runcard, folder, force):
    params = list(load_yaml(runcard).keys())
    if params == LOW_LEVEL_KEYS:
        return ActionBuilderLowLevel(runcard, folder, force)
    elif params == HIGH_LEVEL_KEYS:
        return ActionBuilderHighLevel(runcard, folder, force)
    else:
        raise_error(TypeError, "Runcard invalid!")
