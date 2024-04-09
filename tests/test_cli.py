import pathlib

import pytest
from click.testing import CliRunner

from qibocal.cli._base import command

DUMMY_ACTION = pathlib.Path(__file__).parent / "runcards" / "dummy_action.yml"


def test_fit_command(tmp_path):
    """Test qq fit behavior."""

    tmp_dir_1 = tmp_path / "temp_dir_1"
    tmp_dir_2 = tmp_path / "temp_dir_2"
    runner = CliRunner()
    runner.invoke(
        command,
        ["acquire", str(DUMMY_ACTION), "-o", str(tmp_dir_1), "-f"],
        catch_exceptions=False,
    )

    # fit after acquisition same folder
    runner.invoke(command, ["fit", str(tmp_dir_1)], catch_exceptions=False)
    # raise error if fit on same folder without force
    with pytest.raises(RuntimeError):
        runner.invoke(command, ["fit", str(tmp_dir_1)], catch_exceptions=False)
    # do not raise error with force option
    runner.invoke(command, ["fit", str(tmp_dir_1), "-f"], catch_exceptions=False)

    # fit on separate folder
    runner.invoke(
        command, ["fit", str(tmp_dir_1), "-o", str(tmp_dir_2)], catch_exceptions=False
    )

    # raise error if separate folder exists
    with pytest.raises(RuntimeError):
        runner.invoke(
            command,
            ["fit", str(tmp_dir_1), "-o", str(tmp_dir_2)],
            catch_exceptions=False,
        )

    # do not raise error with force option
    runner.invoke(
        command,
        ["fit", str(tmp_dir_1), "-o", str(tmp_dir_2), "-f"],
        catch_exceptions=False,
    )
    # fit after acquisition different folder
