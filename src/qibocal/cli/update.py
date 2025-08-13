import json
import os
import pathlib
import shutil
from typing import Optional

from qibolab import locate_platform

from ..auto.operation import QubitId
from ..auto.output import META, UPDATED_PLATFORM
from ..calibration import CalibrationPlatform, create_calibration_platform
from ..config import log, raise_error


def update(path: pathlib.Path, skip_qubits: Optional[list[QubitId]]):
    """Perform copy of updated platform in QIBOLAB_PLATFORM

    Arguments:
        - input_path: Qibocal output folder.
    """
    new_platform_path = path / UPDATED_PLATFORM

    if not new_platform_path.exists():
        raise_error(FileNotFoundError, f"No updated runcard platform found in {path}.")

    platform_name = json.loads((path / META).read_text())["platform"]

    platform_path = locate_platform(platform_name)
    old_platform = create_calibration_platform(platform_name)
    for filename in os.listdir(new_platform_path):
        shutil.copy(
            new_platform_path / filename,
            platform_path / filename,
        )

    if skip_qubits is not None:
        new_platform = create_calibration_platform(platform_name)
        updated_platform = merge_with_skipped_qubits(
            old_platform, new_platform, skip_qubits
        )
        updated_platform.dump(platform_path)

    log.info(f"Platform {platform_name} configuration has been updated.")


def merge_with_skipped_qubits(
    old: CalibrationPlatform, new: CalibrationPlatform, skip_qubits: list[QubitId]
) -> CalibrationPlatform:
    """Create a new platform with updated qubits and calibration information."""
    qubits = {
        q: new.qubits[q] if q not in skip_qubits else old.qubits[q] for q in new.qubits
    }
    for q in skip_qubits:
        new.update(
            {
                f"native_gates.single_qubit.{q}": old.parameters.native_gates.single_qubit[
                    q
                ]
            }
        )
        for attr in new.qubits[q].model_fields_set:
            ch = getattr(new.qubits[q], attr)
            if isinstance(ch, dict):
                for ch_ in ch.values():
                    new.update({f"configs.{ch_}": old.config(ch_)})
            else:
                new.update({f"configs.{ch}": old.config(ch)})

    new.calibration.single_qubits = {
        q: new.calibration.single_qubits[q]
        if q not in skip_qubits
        else old.calibration.single_qubits[q]
        for q in new.calibration.single_qubits
    }
    return CalibrationPlatform(
        new.name,
        new.parameters,
        new.instruments,
        qubits,
        new.couplers,
        calibration=new.calibration,
    )
