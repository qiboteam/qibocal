import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def load_qubit(data_path: pathlib.Path, qubit):
    r"""Load the information of the qubit `qubit`
    stored in `data_path`.
    Args:
        data_path (path): Data path.
        qubit: Qubit ID.
    Returns:
        Pandas.DataFrame with `qubit`'s data.
    """
    data = pd.read_csv(data_path, skiprows=[1])
    data = data[data.qubit == qubit]
    return data


def generate_models(data, test_size=0.25):
    r"""Extract from data the values stored
    in the keys `i` and `q` and split them in training and test sets.

    Args:
        data (DataFrame): Qubit's info.
        test_size (float | Optional): The proportion of the dataset to include in the train split.

    Returns:

        - x_train: Training inputs.
        - y_train: Training outputs.
        - x_test: Test inputs.
        - y_test: Test outputs.
    """
    input_data = data[["i", "q"]].values
    output_data = data["state"].values
    return train_test_split(
        input_data, output_data, test_size=test_size, random_state=0, shuffle=True
    )


def plot_qubit(data, save_dir: pathlib.Path):
    r"""Plot the `data` and save it in `{save_dir}/data_processing.pdf`.

    Args:
        data (DataFrame): Input data with "i", "q" and "state" keys.
        save_dir (path): Save path.
    """
    _, axes = plt.subplots(1, 2, figsize=(14, 7))

    colors = ["#FF0000", "#0000FF"]
    sns.set_palette(sns.color_palette(colors))
    sns.scatterplot(
        x="i", y="q", data=data, hue="state", ax=axes[0], alpha=0.7, edgecolor="black"
    )
    qubit = int(data.iloc[0]["qubit"])
    axes[0].set_title(f"qubit {qubit}")
    sns.countplot(x=data.state, data=data, ax=axes[1])
    axes[1].set_title("states distribution")
    plt.tight_layout()
    plt.savefig(save_dir / "data_processing.pdf")
