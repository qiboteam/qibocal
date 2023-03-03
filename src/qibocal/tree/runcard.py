from typing import Any, Dict, List, Optional, Union

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


@dataclass
class Runcard:
    actions: List[Action]
