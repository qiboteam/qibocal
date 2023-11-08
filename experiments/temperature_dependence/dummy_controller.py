from dataclasses import dataclass
from typing import Optional


@dataclass
class DummyTemperatureController:
    """Dummy class for temperature controller"""

    address: str
    _temperature: Optional[float] = None

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temp: float):
        self_temperature = temp
