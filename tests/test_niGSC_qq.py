import os
from pathlib import Path
from shutil import rmtree

from qibocal.cli._base import ActionBuilder

here = Path(__file__).parent


def test_command_niGSC():
    path_to_runcard = here / "niGSC.yml"
    test_folder = "test_and_delete/"
    builder = ActionBuilder(path_to_runcard, test_folder, force=None)
    builder.execute()
    builder.dump_report()
    paths_to_protocols = [
        "data/simulfilteredrb/",
        "data/standardrb/",
        "data/XIdrb/",
    ]
    paths_to_check = [f"{test_folder}{path}" for path in paths_to_protocols]
    for path in paths_to_check:
        assert os.path.isdir(path)
    inside = ["experiment_data.pkl", "fit_plot.pkl"]
    files_to_check = [[f"{path}{name}" for path in paths_to_check] for name in inside]
    for filename_list in files_to_check:
        for filename in filename_list:
            assert os.path.isfile(filename)
    # The circuits data is not stored for standard rb.
    assert os.path.isfile(f"{test_folder}/data/simulfilteredrb/circuits.pkl")
    assert os.path.isfile(f"{test_folder}/data/XIdrb/circuits.pkl")
    assert not os.path.isfile(f"{test_folder}/data/standardrb/circuits.pkl")
    assert os.path.isfile(f"{test_folder}index.html")
    assert os.path.isfile(f"{test_folder}meta.yml")
    assert os.path.isfile(f"{test_folder}runcard.yml")
    rmtree(test_folder)
