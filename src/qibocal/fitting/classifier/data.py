import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def load_qubit(data_path, qubit):
    data = pd.read_csv(data_path, skiprows=[1])
    data = data[data.qubit == qubit]
    return data


def generate_models(data):
    # data = data.sample(frac=1)
    input_data = data[["i", "q"]].values * 10000  # WARNING: change unit measure
    output_data = data["state"].values
    # Split data into X_train, X_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, output_data, test_size=0.25, random_state=0, shuffle=True
    )
    return x_train, y_train, x_test, y_test


def plot_qubit(data, save_dir: pathlib.Path):
    _, axes = plt.subplots(1, 2, figsize=(14, 7))
    sns.scatterplot(x="i", y="q", data=data, hue="state", ec=None, ax=axes[0], s=1)
    qubit = int(data.iloc[0]["qubit"])
    axes[0].set_title(f"qubit {qubit}")
    sns.countplot(x=data.state, data=data, ax=axes[1])
    axes[1].set_title("states distribution")
    plt.savefig(save_dir / "data_processing.pdf")
