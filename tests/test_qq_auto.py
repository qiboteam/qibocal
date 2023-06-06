import pathlib
import tempfile

import pytest

from qibocal.cli.auto_builder import AutoCalibrationBuilder

PATH_TO_RUNCARD = pathlib.Path(__file__).parent / "runcards/test_autocalibration.yml"
PATH_TO_RUNCARD_SIM = (
    pathlib.Path(__file__).parent / "runcards/test_autocalibration_sim.yml"
)


def test_qq_auto_dummy():
    """Test full calibration pipeline for autocalibration."""
    folder = tempfile.mkdtemp()
    builder = AutoCalibrationBuilder(PATH_TO_RUNCARD, folder, force=True)
    builder.run()
    builder.dump_platform_runcard()
    builder.dump_report()


def test_qq_auto_numpy():
    """Test full calibration pipeline for autocalibration."""
    folder = tempfile.mkdtemp()
    builder = AutoCalibrationBuilder(PATH_TO_RUNCARD_SIM, folder, force=True)
    builder.run()
    builder.dump_platform_runcard()
    builder.dump_report()
