from pandas import DataFrame, read_csv


class RBData(DataFrame):
    """A pandas DataFrame child. The output of the acquisition function."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def save(self, path):
        super().to_csv(f"{path}/{self.__class__.__name__}.csv")

    def load(self, path):
        return read_csv(f"{path}/{self.__class__.__name__}.csv")
