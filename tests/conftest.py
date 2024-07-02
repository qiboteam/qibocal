import os

import pytest


@pytest.fixture(autouse=True)
def cd(tmp_path):
    os.chdir(tmp_path)
