import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def cd(tmp_path_factory: pytest.TempdirFactory):
    path: Path = tmp_path_factory.mktemp("run")
    os.chdir(path)
