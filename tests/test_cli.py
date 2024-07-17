import json
import pathlib
import re

import pytest
from click.testing import CliRunner
from lxml.html import parse

from qibocal.cli._base import command

test_runcards_dir = pathlib.Path(__file__).parent / "runcards"
DUMMY_ACTION = test_runcards_dir / "dummy_action.yml"
DUMMY_COMPARE = test_runcards_dir / "dummy_compare.yml"


def test_fit_command(tmp_path):
    """Test qq fit behavior."""

    tmp_dir_1 = tmp_path / "temp_dir_1"
    tmp_dir_2 = tmp_path / "temp_dir_2"
    runner = CliRunner()
    runner.invoke(command, ["acquire", str(DUMMY_ACTION), "-o", str(tmp_dir_1), "-f"])

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


def test_compare_report_dates(tmp_path):
    report_dir_1 = tmp_path / "report_dir_1"
    report_dir_2 = tmp_path / "report_dir_2"
    compare_dir = tmp_path / "compare_dir"

    runner = CliRunner()
    runner.invoke(command, ["auto", str(DUMMY_COMPARE), "-o", str(report_dir_1), "-f"])
    runner.invoke(command, ["auto", str(DUMMY_COMPARE), "-o", str(report_dir_2), "-f"])

    runner.invoke(
        command,
        ["compare", str(report_dir_1), str(report_dir_2), "-o", str(compare_dir), "-f"],
    )
    doc = parse(compare_dir / "index.html").getroot()
    report_info_keys = ["date", "start-time", "end-time"]
    single_report_info = {x: [] for x in report_info_keys}
    for rep in [report_dir_1, report_dir_2]:
        report_meta = json.loads((rep / "meta.json").read_text())
        for key in single_report_info:
            single_report_info[key].append(report_meta[key])

    report_info = doc.get_element_by_id("report_info").text_content().split("\n")
    assert re.sub("^ *Run date: *", "", report_info[2]) == " | ".join(
        single_report_info["date"]
    )
    assert re.sub(r"^ *Start time \(UTC\): *", "", report_info[3]) == " | ".join(
        single_report_info["start-time"]
    )
    assert re.sub(r"^ *End time \(UTC\): *", "", report_info[4]) == " | ".join(
        single_report_info["end-time"]
    )
