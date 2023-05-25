from pandas import DataFrame


class RBData(DataFrame):
    """A pandas DataFrame child. The output of the acquisition function."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # FIXME this is necessary because the auto builder calls .to_csv(path_to_directory).
        # But the DataFrame object from pandas needs a path to a file.
        self.save_func = self.to_csv
        self.to_csv = self.to_csv_helper

    def to_csv_helper(self, path):
        self.save_func(f"{path}/{self.__class__.__name__}.csv")
