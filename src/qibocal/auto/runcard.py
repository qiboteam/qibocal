from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, Union

import yaml
from pydantic.dataclasses import dataclass

from .operation import OperationId

Id = NewType("Id", str)


@dataclass
class Action:
    id: Id
    operation: Optional[OperationId] = None
    main: Optional[Id] = None
    next: Optional[Union[List[Id], Id]] = None
    priority: Optional[int] = None
    parameters: Optional[Dict[str, Any]] = None

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
