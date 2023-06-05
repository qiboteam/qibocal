from pandas import DataFrame


class RBData(DataFrame):
    """A pandas DataFrame child. The output of the acquisition function."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def to_csv(self, path):
        super().to_csv(f"{path}/{self.__class__.__name__}.csv")
