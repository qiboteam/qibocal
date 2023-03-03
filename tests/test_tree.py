import pathlib
from typing import List

import yaml
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder

from qibocal.tree.execute import Executor
from qibocal.tree.runcard import Runcard

cards = pathlib.Path(__file__).parent / "runcards"


@dataclass
class Validation:
    result: List[str]


@dataclass
class TestCard:
    runcard: Runcard
    validation: Validation


def test_execution():
    for card in cards.glob("*.yaml"):
        testcard = TestCard(**yaml.safe_load(card.read_text(encoding="utf-8")))
        executor = Executor.load(pydantic_encoder(testcard.runcard))
        print(executor)
