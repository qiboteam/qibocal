import pathlib
import tempfile

import pytest

from qibocal.cli.auto_builder import AutoCalibrationBuilder

PATH_TO_RUNCARD = pathlib.Path(__file__).parent / "runcards/test_autocalibration.yml"


# FIXME: https://github.com/qiboteam/qibolab/issues/454
@pytest.mark.skip(reason="Waiting for dummy platform support.")
def test_qq_auto_dummy():
    """Test full calibration pipeline for autocalibration."""
    folder = tempfile.mkdtemp()
    builder = AutoCalibrationBuilder(PATH_TO_RUNCARD, folder, force=True)
    builder.run()
    builder.dump_platform_runcard()
    builder.dump_report()
