from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic.dataclasses import dataclass


@dataclass
class Action:
    id: str
    operation: Optional[str] = None
    main: Optional[str] = None
    next: Optional[Union[List[str], str]] = None
    priority: Optional[int] = None
    pars: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if isinstance(self.next, str):
            self.next = [self.next]

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
