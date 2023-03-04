import pathlib
from typing import List

import pytest
import yaml
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder

from qibocal.auto.execute import Executor
from qibocal.auto.runcard import Runcard

cards = pathlib.Path(__file__).parent / "runcards"


@dataclass
class Validation:
    result: List[str]
    description: str


@dataclass
class TestCard:
    runcard: Runcard
    validation: Validation


@pytest.mark.parametrize("card", cards.glob("*.yaml"))
def test_execution(card: pathlib.Path):
    testcard = TestCard(**yaml.safe_load(card.read_text(encoding="utf-8")))
    executor = Executor.load(pydantic_encoder(testcard.runcard))
    executor.run()

    assert testcard.validation.result == [step[0] for step in executor.history]
