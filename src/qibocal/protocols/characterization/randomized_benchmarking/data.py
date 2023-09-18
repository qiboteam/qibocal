import pandas as pd

from qibocal.auto.operation import DATAFILE


class RBData(pd.DataFrame):
    """A pandas DataFrame child. The output of the acquisition function."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def save(self, path):
        """Overwrite because qibocal action builder calls this function with a directory."""
        super().to_json(path / DATAFILE, default_handler=str)

    @classmethod
    def load(cls, path):
        return cls(pd.read_json(path / DATAFILE))
