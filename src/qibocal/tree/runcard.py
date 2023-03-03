from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, Union

import yaml
from pydantic.dataclasses import dataclass

from .operation import OperationId

MAX_PRIORITY = int(1e9)
"""A number bigger than whatever will be manually typed.

But not so insanely big not to fit in a native integer.

"""

Id = NewType("Id", str)


@dataclass
class Action:
    id: Id
    operation: Optional[OperationId] = None
    main: Optional[Id] = None
    next: Optional[Union[List[Id], Id]] = None
    priority: Optional[int] = None
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.next is None:
            self.next = []
        elif isinstance(self.next, str):
            self.next = [self.next]

        if self.priority is None:
            self.priority = MAX_PRIORITY

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class Runcard:
    actions: List[Action]

    @classmethod
    def load(cls, card: Union[dict, Path]):
        content = (
            yaml.safe_load(card.read_text(encoding="utf-8"))
            if isinstance(card, Path)
            else card
        )
        return cls(**content)
