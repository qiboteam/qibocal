import pandas as pd


class RBData(pd.DataFrame):
    """A pandas DataFrame child. The output of the acquisition function."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def save(self, path):
        """Overwrite because qibocal action builder calls this function with a directory."""
        super().to_json(f"{path}/{self.__class__.__name__}.json", default_handler=str)

    @classmethod
    def load(cls, path):
        return cls(pd.read_json(f"{path}/RBData.json"))
